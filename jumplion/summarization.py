# builtin
import logging
import warnings
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass
# external
import numpy as np
import pandas as pd
import polars as pl
# local
from jumplion.arguments import Arguments
from jumplion import _iq_native
from jumplion.id_results_processing import get_prot_prec_pep_info
from jumplion.utils import progress_bar, indent_df, extract_unique_parts

args = Arguments()
ShowProgress = args.show_progress
InterProgress = args.inter_progress


def get_xic_quantity_df_long(xic_quant_df, 
                             prec_id_col,
                             run_col,
                             xic_id_col, 
                             xic_quant_col, 
                             save_data=True,
                             out_dir="",
                             ):
    
    xic_quantity_df_long = (
        xic_quant_df
        .with_columns([
            pl.col(xic_id_col).str.split(";"),
            pl.col(xic_quant_col).str.split(";"),
        ])
        .explode([xic_id_col, xic_quant_col])
        .with_columns([
            pl.col(xic_quant_col).cast(pl.Float64),
        ])
    )
    
    xic_quantity_df_long = (
        xic_quantity_df_long.select([
            pl.col(prec_id_col).cast(pl.Utf8).alias("target_id"),
            pl.col(run_col).cast(pl.Utf8).alias("run"),
            pl.col(xic_id_col).cast(pl.Utf8).alias("source_id"),
            pl.col(xic_quant_col).alias("source_quantity"),
        ])
    )
            
    if save_data:
        xic_quantity_df_long.write_parquet(Path(out_dir) / f"{prec_id_col}_{xic_quant_col}_mapping.parquet")
        
    return xic_quantity_df_long


def quantify_with_maxlfq(df_long, target_id_name):
    required_cols = ['target_id', 'run', 'source_id', 'source_quantity']
    for c in required_cols:
        if c not in df_long.columns:
            raise KeyError(f"Dataframe is missing {c!r}")
    
    # filtering before log2 transformation
    df_long = df_long.filter(pl.col("source_quantity") > 1.0)
    # MaxLFQ
    result = fast_maxlfq(df_long['target_id'], 
                         df_long['run'], 
                         df_long['source_id'], 
                         np.log2(df_long['source_quantity']))
    
    quantity_mtx = pl.from_numpy(
        result.estimate,
        schema=list(result.run_names),
        orient='row',                      # rows = targets, cols = runs
    ).insert_column(0, pl.Series(target_id_name, result.target_ids))
    
    return quantity_mtx

    
def quantify_precursors(xic_quant_df, 
                        use_raw_ms1,
                        use_raw_ms2,
                        prec_quant_level, 
                        out_dir=""):

    #xic_quant_df_sub = xic_quant_df.filter(pl.col("Evidence.Tier") != "prediction_only")
    xic_quant_df_sub = xic_quant_df
    
    #
    id_col="Precursor.Id"
    run_col="Run"
    # MS1
    if use_raw_ms1:
        ms1_quant_col="MS1.XIC.Quant"
    else:
        ms1_quant_col="MS1.XIC.Quant.Adjusted"
    xic_quant_df_l_ms1 = get_xic_quantity_df_long(xic_quant_df_sub, 
                                                    prec_id_col=id_col,
                                                    run_col=run_col,
                                                    xic_id_col="MS1.XIC.Info",
                                                    xic_quant_col=ms1_quant_col,
                                                    save_data=True,
                                                    out_dir=out_dir)
    # MS2
    if use_raw_ms2:
        ms2_quant_col="MS2.XIC.Quant"
    else:
        ms2_quant_col="MS2.XIC.Quant.Adjusted"
    xic_quant_df_l_ms2 = get_xic_quantity_df_long(xic_quant_df_sub, 
                                                    prec_id_col=id_col,
                                                    run_col=run_col,
                                                    xic_id_col="MS2.XIC.Info",
                                                    xic_quant_col=ms2_quant_col,
                                                    save_data=True,
                                                    out_dir=out_dir)
    # MS1 and MS2
    xic_quant_df_l_ms1ms2 = pl.concat([xic_quant_df_l_ms1, xic_quant_df_l_ms2], how="vertical")
    
    prec_quant_all_levels = []
    methods = ['MS1', 'MS2', 'MS1 and MS2']
    long_dfs = [xic_quant_df_l_ms1, xic_quant_df_l_ms2, xic_quant_df_l_ms1ms2]
    for method, df_long in zip(methods, long_dfs):
        #logging.info(f"Quantifying precursors based on {method} XICs")
        prec_quant_mtx = quantify_with_maxlfq(df_long, id_col)
        value_cols = [c for c in prec_quant_mtx.columns if c != id_col]
        prec_quant_mtx = prec_quant_mtx.with_columns(
            [(pl.lit(2.0) ** pl.col(c)).alias(c) for c in value_cols]
        )
        if method == "MS1 and MS2":
            method = "MS1_MS2"
        prec_quant_mtx = prec_quant_mtx.with_columns(pl.lit(method).alias("MS.Level"))
        prec_quant_all_levels.append(prec_quant_mtx)
    
    prec_quant_mat_all_levels = pl.concat(prec_quant_all_levels, how="vertical")
    prec_quant_mat_all_levels.write_csv(
        Path(out_dir) / "precursor_quantity_all_levels.tsv",
        separator="\t",
    )
    
    logging.info(f"Using {prec_quant_level}-level quantification as final quantities")
    prec_quant_mat_final = (
        prec_quant_mat_all_levels
        .filter(pl.col("MS.Level") == prec_quant_level.upper())
        .drop("MS.Level")
    )
    
    return prec_quant_mat_final


def quantify_proteins(prec_quant_table, top_n=3):
    """
    top_n : int or None
        Use only the top N precursors per protein group, ranked by mean
        intensity across runs (computed over runs where the precursor was
        detected). None = use all precursors.
    """
    logging.info("Quantifying proteins")

    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1 or None, got {top_n}")

    META_COLS = ['Precursor.Id', 'Protein.Ids', 'Protein.Group', 'Protein.Names', 'Genes']
    meta_cols = [c for c in META_COLS if c in prec_quant_table.columns]
    run_cols  = [c for c in prec_quant_table.columns if c not in meta_cols]
    logging.info(f"{prec_quant_table.height} precursors x {len(run_cols)} runs")

    # precursor -> protein mapping, pulled out of the combined table
    prec_prot = prec_quant_table.select(meta_cols).unique(subset='Precursor.Id')

    long_df = (
        prec_quant_table
        .unpivot(
            index=['Protein.Group', 'Precursor.Id'],
            on=run_cols,
            variable_name='Run',
            value_name='Quantity',
        )
        .drop_nulls('Quantity')
        .rename({
            'Protein.Group': 'target_id',
            'Precursor.Id':  'source_id',
            'Run':           'run',
            'Quantity':      'source_quantity',
        })
    )

    # keep top N precursors per protein, ordered by mean intensity
    if top_n is not None:
        top_prec = (
            long_df
            .group_by(['target_id', 'source_id'])
            .agg(pl.col('source_quantity').mean().alias('mean_quantity'))
            # source_id as tie-breaker so selection is deterministic
            .sort(['target_id', 'mean_quantity', 'source_id'],
                  descending=[False, True, False])
            .group_by('target_id', maintain_order=True)
            .head(top_n)
            .select(['target_id', 'source_id'])
        )
        n_before = long_df.select(pl.col('source_id').n_unique()).item()
        long_df = long_df.join(top_prec, on=['target_id', 'source_id'], how='semi')
        n_after = long_df.select(pl.col('source_id').n_unique()).item()
        logging.info(f"Top-{top_n} filter: kept {n_after} of {n_before} precursors")

    prot_quant_mtx = quantify_with_maxlfq(long_df, target_id_name='Protein.Group')
    value_cols = [c for c in prot_quant_mtx.columns if c != 'Protein.Group']
    prot_quant_mtx = prot_quant_mtx.with_columns(
        [(pl.lit(2.0) ** pl.col(c)).alias(c) for c in value_cols]
    )

    # number of peptides per protein
    prot_prec_pep_info = get_prot_prec_pep_info(prec_prot.to_pandas())
    prot_quantity_table = (
        pl.from_pandas(prot_prec_pep_info)
          .join(prot_quant_mtx, on='Protein.Group', how='inner')
    )

    prot_info = prec_prot.select(
        ['Protein.Group', 'Protein.Names', 'Genes']
    ).unique()

    return prot_info.join(prot_quantity_table, on='Protein.Group', how='inner')


def filter_one_hit_wonder(prot_quantity_table, min_prec_count=2, min_pep_count=2):
    # For one-hit wonder (protein with one precursor/peptide)
    prot_1_prec = prot_quantity_table.filter(pl.col("n.Precursors") == 1)["Protein.Group"]
    prot_1_pep = prot_quantity_table.filter(pl.col("n.Peptides") == 1)["Protein.Group"]
    logging.info(
        f"Found {len(prot_1_prec)} proteins with one precursor and {len(prot_1_pep)} proteins with one peptide"
    )

    # Further filtering
    logging.info(
        f"Filtering proteins based on the number of precursors (>={min_prec_count}) and peptides (>={min_pep_count})"
    )
    prot_quantity_table_f = prot_quantity_table.filter(
        (pl.col("n.Precursors") >= min_prec_count) &
        (pl.col("n.Peptides") >= min_pep_count)
    )

    return prot_quantity_table_f


################################## MaxLFQ  ####################################
"""
Python-native MaxLFQ, backed by a C++ extension ported from tvpham/iq
(https://github.com/tvpham/iq, BSD-3-Clause), with the R (.Call/SEXP)
layer removed.

Mirrors the interface and behavior of R's `iq::fast_MaxLFQ()`.
"""
@dataclass
class MaxLFQResult:
    estimate: np.ndarray          # (n_proteins, n_run) log2 abundance matrix
    target_ids: list[str]         # row labels, in estimate's row order
    run_names: list[str]       # column labels, in estimate's column order
    annotation: list[str]         # per-protein connectivity annotation (matches iq's $annotation)


# def fast_maxlfq(
#     target_list: Sequence[str],
#     run_list: Sequence[str],
#     ion_id: Sequence,
#     quant: Sequence[float],
# ) -> MaxLFQResult:
#     """
#     Estimate relative protein abundances with MaxLFQ.

#     Parameters
#     ----------
#     target_list  : one target id (precursor/protein) per observation (parallel to run_list/ion_id/quant)
#     run_list  : one sample id per observation
#     ion_id       : one fragment/ion/precursor id per observation
#     quant        : log2 intensity per observation 

#     Returns
#     -------
#     MaxLFQResult
#     """
#     target_list = np.asarray(target_list)
#     run_list = np.asarray(run_list)
#     ion_id = np.asarray(ion_id)
#     quant = np.asarray(quant, dtype=np.float64)

#     if not (len(target_list) == len(run_list) == len(ion_id) == len(quant)):
#         raise ValueError("target_list, run_list, ion_id, quant must have equal length")
#     if np.isnan(quant).any():
#         raise ValueError("NA value in quant")

#     # factorize to 0-based dense codes before .Call("iq_MaxLFQ", ...) in fast_MaxLFQ()
#     targets, target_codes = np.unique(target_list, return_inverse=True)
#     runs, run_codes = np.unique(run_list, return_inverse=True)
#     _, ion_codes = np.unique(ion_id, return_inverse=True)

#     raw = _iq_native.run_maxlfq(
#         target_codes.astype(np.int32).tolist(),
#         ion_codes.astype(np.int32).tolist(),
#         run_codes.astype(np.int32).tolist(),
#         quant.tolist(),
#     )

#     return MaxLFQResult(
#         estimate=np.asarray(raw["estimate"]),
#         target_ids=[str(targets[i]) for i in raw["row_protein_index"]],
#         run_names=[str(runs[j]) for j in raw["col_sample_index"]],
#         annotation=list(raw["annotation"]),
#     )

def _factorize(values, name):
    # object dtype holds pointers to the existing str objects: no padding, no copy of the text
    codes, uniques = pd.factorize(np.asarray(values, dtype=object), sort=True)
    if (codes < 0).any():
        raise ValueError(f"NA value in {name}")
    return codes.astype(np.int32, copy=False), np.asarray(uniques)


def fast_maxlfq(
    target_list: Sequence[str],
    run_list: Sequence[str],
    ion_id: Sequence,
    quant: Sequence[float],
) -> MaxLFQResult:
    """
    Estimate relative protein abundances with MaxLFQ.

    Parameters
    ----------
    target_list  : one target id (precursor/protein) per observation (parallel to run_list/ion_id/quant)
    run_list  : one sample id per observation
    ion_id       : one fragment/ion/precursor id per observation
    quant        : log2 intensity per observation 

    Returns
    -------
    MaxLFQResult
    """
    
    n = len(quant)
    if not (len(target_list) == len(run_list) == len(ion_id) == n):
        raise ValueError("target_list, run_list, ion_id, quant must have equal length")

    quant = np.asarray(quant, dtype=np.float64)
    if np.isnan(quant).any():
        raise ValueError("NA value in quant")

    # factorize to 0-based dense codes (sorted, same as np.unique(return_inverse=True))
    target_codes, targets = _factorize(target_list, "target_list")
    run_codes, runs = _factorize(run_list, "run_list")
    ion_codes, _ = _factorize(ion_id, "ion_id")

    raw = _iq_native.run_maxlfq(
        target_codes.tolist(),
        ion_codes.tolist(),
        run_codes.tolist(),
        quant.tolist(),
    )

    return MaxLFQResult(
        estimate=np.asarray(raw["estimate"]),
        target_ids=[str(targets[i]) for i in raw["row_protein_index"]],
        run_names=[str(runs[j]) for j in raw["col_sample_index"]],
        annotation=list(raw["annotation"]),
    )

################################## Normalization ##############################
def norm_dataframe(df, data_type, tag_column, value_columns, normalization, norm_species, norm_method='median', trim_percentage=5):
    if normalization:
        logging.info(f"Performing {data_type.lower()}-level normalization")
        if norm_species == 'ALL' or tag_column not in df.columns:
            logging.info(f"Use all {data_type.lower()}s for normalization")
            sub_df = df.clone()
        else:
            logging.info(f"Use {norm_species} {data_type.lower()}s for normalization")
            species_tags = df.select(tag_column).with_columns(
                pl.col(tag_column).str.contains(norm_species).alias("match")
            )["match"]
            if species_tags.sum() == 0:
                logging.warning('WARNING: No matching species found in your input. Using all proteins')
                sub_df = df.clone()
            else:
                sub_df = df.filter(species_tags)
    else:
        logging.info(f"{data_type.lower()}-level normalization turned off")
        sub_df = df.clone()
        
    mat = sub_df.select([
        pl.col(col).log(2).alias(col) for col in value_columns
    ])
    mat = mat.select(pl.all().fill_nan(None))
    loadBias = get_loading_bias(mat, value_columns, data_type, trim_percentage, trim_percentage)
    medians_df = mat.median().to_pandas().T.reset_index(drop=False)
    medians_df.columns = ['Runs', 'Medians']
    
    logging.info("Loading bias before normalization:\n%s", 
                 indent_df(loadBias, spaces=4, index=False))
    
    if normalization:
        # get norm factors
        mat_norm, normFactors = normalize(mat, trim_percentage, trim_percentage, norm_method)
        normFactors_df = pd.DataFrame(list(normFactors.items()), columns=['Runs', 'NormFactors'])
        medians_df = medians_df.merge(normFactors_df)
        medians_df['Runs'] = extract_unique_parts(medians_df['Runs'].to_list())
        logging.info("Median log2(Intensity) before normalization:\n%s", 
                     indent_df(medians_df, spaces=4, index=False))
        
        medians_norm_df = mat_norm.median().to_pandas().T.reset_index(drop=False)
        medians_norm_df.columns = ['Runs', 'Medians']
        medians_norm_df['Runs'] = extract_unique_parts(medians_norm_df['Runs'].to_list())
        logging.info("Median log2(Intensity) after normalization:\n%s", 
                     indent_df(medians_norm_df, spaces=4, index=False))
        
        # normalization of the whole df
        df_norm = df.with_columns([
            (pl.col(col) * (2 ** -normFactors[col])).alias(col)
            for col in value_columns
        ])
    else:
        medians_df['Runs'] = extract_unique_parts(medians_df['Runs'].to_list())
        logging.info("Median log2(Intensity) before normalization:\n%s", 
                     indent_df(medians_df, spaces=4, index=False))
        df_norm = df.clone()
    
    return df_norm

def trim_matrix(mat, trim_pct_upper, trim_pct_lower):
    q25 = mat.quantile(0.25)
    q75 = mat.quantile(0.75)
    cutoff = q25 - 1.5 * (q75 - q25)
    
    mat_np = mat.to_numpy()
    cutoff_np = cutoff.to_numpy()
    valid_rows = (mat_np > cutoff_np).all(axis=1)
    mat_f = pl.DataFrame(mat_np[valid_rows], schema=mat.columns)

    # Compute row-wise means
    mat_f_with_rowmean = mat_f.with_columns(
        pl.mean_horizontal(pl.all()).alias("row_mean")
    )

    # Subtract row means from each row
    centered_np = mat_f_with_rowmean.drop("row_mean").to_numpy() - mat_f_with_rowmean["row_mean"].to_numpy()[:, None]

    valid_idx = np.ones(centered_np.shape[0], dtype=bool)
    for i, col in enumerate(mat.columns):
        col_data = centered_np[:, i]
        low = np.percentile(col_data, trim_pct_lower)
        high = np.percentile(col_data, 100 - trim_pct_upper)
        valid_idx &= (col_data > low) & (col_data < high)

    return pl.DataFrame(centered_np[valid_idx], schema=mat.columns)

def get_loading_bias(mat, run_names, data_type, trim_pct_upper, trim_pct_lower):
    mat_f = trim_matrix(mat, trim_pct_upper, trim_pct_lower)
    n = mat_f.height
    sm = 2 ** mat_f.mean().to_numpy()[0]
    msm = sm.mean()
    avg = sm / msm * 100
    sdVal = mat_f.std().to_numpy()[0]
    sd = ((2 ** sdVal - 2 ** (-sdVal)) / 2) * 100
    sem = sd / np.sqrt(n)

    loadBias = pd.DataFrame({
        "Runs": extract_unique_parts(run_names),
        "Mean[%]": avg,
        "SD[%]": sd,
        "SEM[%]": sem,
        f"# {data_type}s": [n] * len(avg)
    })
    return loadBias

def normalize(mat, trim_pct_upper, trim_pct_lower, norm_method):
    mat_f = trim_matrix(mat, trim_pct_upper, trim_pct_lower)
    
    if norm_method == "mean":
        sm = mat_f.mean().row(0)
    elif norm_method == "median":
        sm = mat_f.median().row(0)
    else:
        raise ValueError("Normalization method not supported!")
        
    sm = np.array(sm)  # convert to NumPy array
    target = sm.mean()
    normFactors = sm - target
    
    # Subtract normFactors from mat (column-wise)
    mat_np = mat.to_numpy()
    res_np = mat_np - normFactors
    res = pl.DataFrame(res_np, schema=mat.columns)
    
    return res, dict(zip(mat.columns, normFactors))
