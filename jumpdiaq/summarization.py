# builtin
import textwrap
import logging
import warnings
from pathlib import Path
# external
import numpy as np
from numba import njit
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
# local
from jumpdiaq.arguments import Arguments
from jumpdiaq.utils import progress_bar, extract_unique_parts

args = Arguments()
ShowProgress = args.show_progress
InterProgress = args.inter_progress

def get_quantity_quality_dfs(report_df, 
                             id_col,
                             sample_col,
                             secondary_id_col, 
                             quantity_col, 
                             quality_col, 
                             save_data=True,
                             out_dir="",
                             ):
    
    # Split and explode quantity and quality columns
    report_df_long = (
        report_df
        .with_columns([
            pl.col(secondary_id_col).str.split(";"),
            pl.col(quantity_col).str.split(";"),
            pl.col(quality_col).str.split(";")
        ])
        .explode([secondary_id_col, quantity_col, quality_col])
        .with_columns([
            pl.col(quantity_col).cast(pl.Float64),
            pl.col(quality_col).cast(pl.Float64)
        ])
    )
    
    # quantity
    quantity_df = (
        report_df_long.select([
            pl.col(id_col).cast(pl.Utf8).alias("target_id"),
            pl.col(sample_col).cast(pl.Utf8).alias("samples"),
            pl.col(secondary_id_col).cast(pl.Utf8).alias("source_id"),
            pl.col(quantity_col).alias("source_quantity")
        ])
    )

    # quality
    quality_df = (
        report_df_long.select([
            pl.col(id_col).cast(pl.Utf8).alias("target_id"),
            pl.col(sample_col).cast(pl.Utf8).alias("samples"),
            pl.col(secondary_id_col).cast(pl.Utf8).alias("source_id"),
            pl.col(quality_col).alias("source_quality")
        ])
    )
            
    if save_data:
        quantity_df.write_parquet(Path(out_dir) / f"{id_col}_{quantity_col}_mapping.parquet")
        quality_df.write_parquet(Path(out_dir) / f"{id_col}_{quality_col}_mapping.parquet")
        
    return quantity_df, quality_df


def precursor_summarize(quantity_df: pl.DataFrame, 
                        quality_df: pl.DataFrame,
                        xic_quality_cutoff=0.0,
                        fill_missing_xic=True,
                        ):

    # Pivot input tables
    quantity_pivoted = quantity_df.pivot(
        values="source_quantity",
        index=["target_id", "source_id"],
        on="samples",
        aggregate_function="first"
    )
    quality_pivoted = quality_df.pivot(
        values="source_quality",
        index=["target_id", "source_id"],
        on="samples",
        aggregate_function="first"
    )

    sample_cols = [col for col in quantity_pivoted.columns if col not in {"target_id", "source_id"}]
    prec_ids = quantity_pivoted["target_id"].unique().to_list()

    # Group by precursor for fast lookup
    quantity_groups = quantity_pivoted.partition_by("target_id", as_dict=True)
    quality_groups = quality_pivoted.partition_by("target_id", as_dict=True)

    quantity_summed = np.full((len(prec_ids), len(sample_cols)), np.nan)
    quality_summed = np.full((len(prec_ids), len(sample_cols)), np.nan)

    for i, prec_id in enumerate(progress_bar(prec_ids, ShowProgress, InterProgress)):
        quantity_df_sub = quantity_groups.get((prec_id,))
        if quantity_df_sub is None:
            continue
    
        quality_df_sub = quality_groups.get((prec_id,))
        if quality_df_sub is None:
            continue
        
        quantity_mat = quantity_df_sub.select(sample_cols).to_numpy()
        quality_mat = quality_df_sub.select(sample_cols).to_numpy()
        
        quantity_mat_new = quantity_mat.copy()
        if fill_missing_xic:
            if xic_quality_cutoff <= 0.0:
                missing_mask = (quality_mat < xic_quality_cutoff) | (quality_mat == 0.0)
            else:
                missing_mask = quality_mat < xic_quality_cutoff
            quantity_mat_new[missing_mask] = np.nan
            # remove rows that contain all NaNs
            quantity_mat_new = quantity_mat_new[~np.all(np.isnan(quantity_mat_new), axis=1)]
            # fill missing value
            if quantity_mat_new.shape[0] == 0:
                continue
            else:
                quantity_mat_new = np.nan_to_num(quantity_mat_new, nan=np.nanmin(quantity_mat_new))
        #
        quantity_mat_new = np.log2(quantity_mat_new)
        quantity_summed[i, :] = median_summ(quantity_mat_new, topN_rep=3)
        #
        quality_mat_new = quality_mat.copy()
        quality_mat_new[quality_mat_new <= 0.0] = 0.01
        quality_summed[i, :] = topN(quality_mat_new, N=6, method="median")
        
    quantity_table = pl.DataFrame(quantity_summed, schema=sample_cols)
    quantity_table = quantity_table.with_columns(pl.mean_horizontal(pl.all()).alias("Average"))
    quantity_table = quantity_table.with_columns(pl.Series(name="Precursor.Id", values=prec_ids))
    
    quality_table = pl.DataFrame(quality_summed, schema=sample_cols)
    quality_table = quality_table.with_columns(pl.mean_horizontal(pl.all()).alias("Average"))
    quality_table = quality_table.with_columns(pl.Series(name="Precursor.Id", values=prec_ids))

    return quantity_table, quality_table


def extract_annotation(id_list, 
                       report_df: pl.DataFrame, 
                       id_col: str, 
                       annotation_cols: list[str] | None):
    if annotation_cols is None:
        annotation_cols = []

    all_columns = [id_col] + annotation_cols

    # Check all columns exist
    missing = [col for col in all_columns if col not in report_df.columns]
    if missing:
        raise ValueError(f"The input table has no column: {missing[0]}")

    # Filter rows where id_col is in id_list
    filtered = report_df.filter(pl.col(id_col).is_in(id_list))
    if filtered.is_empty():
        raise ValueError("Cannot find any of the given target IDs in the table.")

    # Select only relevant columns and drop duplicates
    tab = filtered.select(all_columns).unique()

    # Create an indexed DataFrame (simulate pandas .loc by join)
    id_df = pl.DataFrame({id_col: id_list})
    joined = id_df.join(tab, on=id_col, how="left")

    return joined


def quantify_from_xic_report(
        report_file, 
        id_col="Precursor.Id",
        sample_col="Run",
        secondary_id_col="MS2.XIC.Info",
        quality_col="MS2.XIC.Corr.Raw",
        quantity_col="MS2.XIC.Quant.Adjusted",
        annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
        out_dir="",
        fill_missing=True, 
        xic_quality_cutoff=0.0
    ):
    
    columns = [sample_col, id_col, secondary_id_col, quality_col, quantity_col] + annotation_cols
    report_df = pl.read_parquet(Path(report_file), columns=columns)
    logging.info(f"ID column: {id_col}, Quantity column: {quantity_col}")
    
    # 1. extract quantity and quality
    quantity_df, quality_df = get_quantity_quality_dfs(
                                report_df, 
                                id_col,
                                sample_col,
                                secondary_id_col,
                                quality_col,
                                quantity_col,
                                save_data=True,
                                out_dir=out_dir,
                            )
    
    # 2. summarize
    quantity_mtx, quality_mtx = precursor_summarize(quantity_df, 
                                                 quality_df, 
                                                 fill_missing, 
                                                 xic_quality_cutoff)

    # 3. add annotation
    extra_names = extract_annotation(quantity_mtx["Precursor.Id"].to_list(), 
                                     report_df, 
                                     id_col,
                                     annotation_cols)
    
    final_quantity_table = extra_names.join(quantity_mtx, on=id_col, how="inner")
    final_quality_table = extra_names.join(quality_mtx, on=id_col, how="inner")  
    
    return final_quantity_table, final_quality_table
    

def quantify_precursors(report_file, 
                        xic_quality_cutoff=0.0, 
                        fill_missing_xic=True,
                        quantity_source='ms1_ms2', 
                        ms2_priority_quality=1.0, 
                        out_dir=""):
    
    # MS1-level quantification
    logging.info("Quantifying precursors at MS1-level")
    ms1_quantity_table, ms1_quality_table = quantify_from_xic_report(
                                             report_file, 
                                             #sample_meta_file,
                                             id_col="Precursor.Id",
                                             sample_col="Run",
                                             secondary_id_col="MS1.XIC.Info",
                                             quality_col="MS1.XIC.Corr.Raw",
                                             quantity_col="MS1.XIC.Quant.Adjusted",
                                             annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
                                             out_dir=out_dir,
                                             fill_missing=fill_missing_xic,
                                             xic_quality_cutoff=xic_quality_cutoff)
    ms1_quantity_table = ms1_quantity_table.with_columns(pl.lit('MS1').alias('MS_type'))
    ms1_quality_table = ms1_quality_table.with_columns(pl.lit('MS1').alias('MS_type'))
    ms1_quantity_table = ms1_quantity_table.with_columns([
        (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
    ])
    ms1_quality_table = ms1_quality_table.with_columns([
        (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
    ])

    # MS2-level quantification
    logging.info("Quantifying precursors at MS2-level")
    ms2_quantity_table, ms2_quality_table = quantify_from_xic_report(
                                             report_file, 
                                             #sample_meta_file,
                                             id_col="Precursor.Id",
                                             sample_col="Run",
                                             secondary_id_col="MS2.XIC.Info",
                                             quality_col="MS2.XIC.Corr.Raw",
                                             quantity_col="MS2.XIC.Quant.Adjusted",
                                             annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
                                             out_dir=out_dir,
                                             fill_missing=fill_missing_xic,
                                             xic_quality_cutoff=xic_quality_cutoff)
    ms2_quantity_table = ms2_quantity_table.with_columns(pl.lit('MS2').alias('MS_type'))
    ms2_quality_table = ms2_quality_table.with_columns(pl.lit('MS2').alias('MS_type'))
    ms2_quantity_table = ms2_quantity_table.with_columns([
        (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
    ])
    ms2_quality_table = ms2_quality_table.with_columns([
        (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
    ])
    
    # Combine MS1 and MS2 quantification
    ms1_quality_table = ms1_quality_table.select(ms2_quality_table.columns)
    ms1_quantity_table = ms1_quantity_table.select(ms2_quantity_table.columns)
    ms1_ms2_quality_table = pl.concat([ms1_quality_table, ms2_quality_table], how="vertical")
    ms1_ms2_quantity_table = pl.concat([ms1_quantity_table, ms2_quantity_table], how="vertical")
    
    # Save to files
    ms1_ms2_quantity_table.write_parquet(Path(out_dir) / "precursor_quantity_ms1_ms2_summ_table.parquet")
    ms1_ms2_quality_table.write_parquet(Path(out_dir) / "precursor_quality_ms1_ms2_summ_table.parquet")
    
    if quantity_source == 'ms1':
        logging.info("Using only MS1-level precursor quantity for further analysis")
        return ms1_quantity_table, ms1_quality_table
    elif quantity_source == 'ms2':
        logging.info("Using only MS2-level precursor quantity for further analysis")
        return ms2_quantity_table, ms2_quality_table
    elif quantity_source == 'ms1_ms2':
        logging.info("Dynamically select MS1- or MS2-level quantity for each precursor based on quality scores")
        # Keep only the row with the highest Average per Precursor.Id, but prioritize MS2 when both > 0.8
        ms1_ms2_quality_table = (
            ms1_ms2_quality_table
            .with_columns([
                # Get MS1 and MS2 averages for each Precursor.Id (null if not exists)
                pl.col("Average").filter(pl.col("MS_type") == "MS1").first().over("Precursor.Id").alias("ms1_avg"),
                pl.col("Average").filter(pl.col("MS_type") == "MS2").first().over("Precursor.Id").alias("ms2_avg")
            ])
            .with_columns([
                # Check if both MS1 and MS2 exist and have Average > ms2_priority_quality
                ((pl.col("ms1_avg").is_not_null()) & (pl.col("ms1_avg") > ms2_priority_quality)).alias("ms1_good"),
                ((pl.col("ms2_avg").is_not_null()) & (pl.col("ms2_avg") > ms2_priority_quality)).alias("ms2_good")
            ])
            .filter(
                # If both MS1 and MS2 have Average > ms2_priority_quality, choose MS2
                pl.when(pl.col("ms1_good") & pl.col("ms2_good"))
                .then(pl.col("MS_type") == "MS2")
                # Otherwise, choose the one with max average
                .otherwise(pl.col("Average") == pl.col("Average").max().over("Precursor.Id"))
            )
            .drop(["ms1_avg", "ms2_avg", "ms1_good", "ms2_good"])
        )
        
        # Filter quantity table to keep only rows matching filtered quality table
        ms1_ms2_quantity_table = ms1_ms2_quantity_table.filter(
            pl.col("Precursor.Id_MS_type").is_in(
                ms1_ms2_quality_table.get_column("Precursor.Id_MS_type").implode()
            )
        )
        return ms1_ms2_quantity_table, ms1_ms2_quality_table


def protein_summarize(prec_quantity_table, 
                      prec_quality_table, 
                      sample_cols):
    # Group by protein group for fast lookup
    quantity_groups = prec_quantity_table.partition_by("Protein.Group", as_dict=True)
    quality_groups = prec_quality_table.partition_by("Protein.Group", as_dict=True)
    
    prot_ids = [prot_key[0] for prot_key in quantity_groups.keys()]
    quantity_summed = np.full((len(prot_ids), len(sample_cols)), np.nan)
    quality_summed = np.full((len(prot_ids), len(sample_cols)), np.nan)

    for i, prot_id in enumerate(progress_bar(prot_ids, ShowProgress, InterProgress)):
        quantity_df_sub = quantity_groups.get((prot_id,))
        if quantity_df_sub is None:
            continue
    
        quality_df_sub = quality_groups.get((prot_id,))
        if quality_df_sub is None:
            continue
        
        prec_ids = set(quantity_df_sub["Precursor.Id"].to_list()).intersection(
            quality_df_sub["Precursor.Id"].to_list()
        )
        if not prec_ids:
            continue
        prec_ids = list(prec_ids)
        
        quality_df_sub = quality_df_sub.filter(pl.col("Precursor.Id").is_in(prec_ids)).sort("Precursor.Id")
        quantity_df_sub = quantity_df_sub.filter(pl.col("Precursor.Id").is_in(prec_ids)).sort("Precursor.Id")
        
        quantity_mat = quantity_df_sub.select(sample_cols).to_numpy()
        quality_mat = quality_df_sub.select(sample_cols).to_numpy()
        
        # Summarization
        quantity_mat = quantity_mat[~np.all(np.isnan(quantity_mat), axis=1)]
        quantity_summed[i, :] = median_summ(quantity_mat, topN_rep=3)
        quality_mat = quality_mat[~np.all(np.isnan(quality_mat), axis=1)]
        quality_summed[i, :] = topN(quality_mat, N=3, method="mean")
        
    quantity_table = pl.DataFrame(quantity_summed, schema=sample_cols)
    quantity_table = quantity_table.with_columns(pl.Series(name="Protein.Group", values=prot_ids))
    
    quality_table = pl.DataFrame(quality_summed, schema=sample_cols)
    quality_table = quality_table.with_columns(pl.Series(name="Protein.Group", values=prot_ids))

    return quantity_table, quality_table


def quantify_proteins(prec_quantity_table, 
                      prec_quality_table,
                      min_prec_quality):
    logging.info("Quantifying proteins")
    # filtering based on average quality
    prec_quality_table = prec_quality_table.filter(pl.col("Average") >= min_prec_quality)
    # Extract unique sample column names (excluding primary_id and id)
    other_cols = {'Precursor.Id',
                  'Protein.Group',
                  'Protein.Names',
                  'Genes',
                  'Average',
                  'Additional.Quan.Info',
                  'MS_type',
                  'Precursor.Id_MS_type'}
    sample_cols = [col for col in prec_quantity_table.columns if col not in other_cols]
    prot_quantity_table, prot_quality_table = protein_summarize(prec_quantity_table, 
                                                             prec_quality_table, 
                                                             sample_cols)
    extra_names = extract_annotation(prot_quantity_table['Protein.Group'].to_list(), 
                                     prec_quantity_table, 
                                     id_col='Protein.Group',
                                     annotation_cols=['Protein.Names', 'Genes'])
    prot_quantity_table = extra_names.join(prot_quantity_table, on='Protein.Group', how="inner")   
    prot_quality_table = extra_names.join(prot_quality_table, on='Protein.Group', how="inner")  
    
    return prot_quantity_table, prot_quality_table
    

def quality_hist_plot(quality_table, value_columns, quality_cutoffs, out_dir, save_file_name):
    # Calculate appropriate figure size based on number of columns
    num_cols = quality_table.loc[:, value_columns].shape[1]
    nrows=int(np.ceil(num_cols/4))
    ncols=int(num_cols/nrows)
    fig_width = ncols * 2  # Base width of 10, scales with more columns
    fig_height = nrows * 2  # Base height of 6, scales with more columns
    
    # Create subplots with shared axes
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=(fig_width, fig_height), 
                             sharex=True, sharey=True)
    
    # Flatten axes array for easier iteration
    if num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot histograms
    for i, (col_name, col_data) in enumerate(quality_table.iloc[:, 3:].items()):
        if i < len(axes):
            col_data.hist(ax=axes[i], bins=20, color='gray', grid=False)
            # Add vertical dashed lines for quality cutoffs
            for cutoff_value in quality_cutoffs:
                axes[i].axvline(x=cutoff_value, color='red', linestyle='--', 
                                linewidth=1, alpha=0.8)
            
            # Wrap long titles
            wrapped_title = '\n'.join(textwrap.wrap(col_name, width=20))
            axes[i].set_title(wrapped_title, fontsize=10)
    
    # Hide any unused subplots
    for j in range(num_cols, len(axes)):
        fig.delaxes(axes[j])
    
    # Apply tight layout
    plt.tight_layout()
    plt.savefig(out_dir / save_file_name)
    plt.close()

################################# MaxLFQ algorithm ############################
# The maxLFQ function is implemented based on R codes from the iq R package with modifications.
# GitHub of iq R package: https://github.com/tvpham/iq/tree/master
@njit
def compute_medians(mat):
    """Compute median log-intensity differences for each pair of samples."""
    N = mat.shape[1]
    AtA = np.zeros((N, N))
    Atb = np.zeros(N)

    for i in range(N - 1):
        for j in range(i + 1, N):
            valid_mask = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            if np.any(valid_mask):
                r_ij = np.median(-mat[valid_mask, i] + mat[valid_mask, j])
                AtA[i, j] = AtA[j, i] = -1
                AtA[i, i] += 1
                AtA[j, j] += 1
                Atb[i] -= r_ij
                Atb[j] += r_ij

    return AtA, Atb


def solve_lsq(AtA, Atb, mat):
    """Solve the least squares problem for MaxLFQ estimation."""
    N = AtA.shape[0]
    A_ext = np.vstack((np.hstack((2 * AtA, np.ones((N, 1)))), np.append(np.ones(N), 0)))
    b_ext = np.append(2 * Atb, np.nanmean(mat) * N)
    solution = np.linalg.lstsq(A_ext, b_ext, rcond=None)[0]
    return solution[:N]

@njit
def spread(mat, g, i, cc):
    """Recursive function to label connected components."""
    g[i] = cc
    for r in range(mat.shape[0]):
        if not np.isnan(mat[r, i]):
            for k in range(mat.shape[1]):
                if not np.isnan(mat[r, k]) and np.isnan(g[k]):
                    spread(mat, g, k, cc)

def maxLFQ(mat):
    """Main function to compute MaxLFQ estimates from input matrix mat."""
    if np.all(np.isnan(mat)):
        return np.nan

    if mat.shape[0] == 1:
        return mat[0, :]

    N = mat.shape[1]
    g = np.full(N, np.nan)  # Group labels
    cc = 0  # Component counter

    # Identify connected components
    for i in range(N):
        if np.isnan(g[i]):
            cc += 1
            spread(mat, g, i, cc)

    w = np.full(N, np.nan)
    for i in range(1, cc + 1):
        indices = np.where(g == i)[0]
        if len(indices) == 1:
            if np.all(np.isnan(mat[:, indices])):
                w[indices] = np.nan
            else:
                w[indices] = np.nanmedian(mat[:, indices], axis=0)
        else:
            sub_X = mat[:, indices]
            AtA, Atb = compute_medians(sub_X)
            w[indices] = solve_lsq(AtA, Atb, sub_X)

    return w

##########################  #########################
def median_summ(mat, log2values=True, topN_rep=3):
    if not log2values:
        mat = np.log2(mat)
    if mat.shape[0] == 1:
        return mat[0, :]
    rowMeans = np.nanmean(mat, axis=1)
    repAbundance = np.mean(sorted(rowMeans, reverse=True)[0:topN_rep])
    mat = mat - rowMeans[:, np.newaxis]
    
    # Check which columns are all-NaN before computing median
    all_nan_cols = np.all(np.isnan(mat), axis=0)
    medians = np.full(mat.shape[1], np.nan)
    
    # Only compute median for columns that aren't all-NaN
    valid_cols = ~all_nan_cols
    if np.any(valid_cols):
        medians[valid_cols] = np.nanmedian(mat[:, valid_cols], axis=0)
        
    out = medians + repAbundance
    return out

def median_simp(mat):
    if mat.shape[0] == 1:
        return mat[0, :]
    
    # Check which columns are all-NaN before computing median
    all_nan_cols = np.all(np.isnan(mat), axis=0)
    medians = np.full(mat.shape[1], np.nan)
    
    # Only compute median for columns that aren't all-NaN
    valid_cols = ~all_nan_cols
    if np.any(valid_cols):
        medians[valid_cols] = np.nanmedian(mat[:, valid_cols], axis=0)
        
    return medians

##############################
def topN(mat, N=3, method="mean"):
    if mat.shape[0] == 1:
        return mat[0, :]

    v = np.nanmean(mat, axis=1)
    v_sorted_idx = np.argsort(v)[::-1]  # Sort descending
    selected_rows = mat[v_sorted_idx[:min(N, len(v))], :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "mean":
            out = np.nanmean(selected_rows, axis=0)
        elif method == "median":
            out = np.nanmedian(selected_rows, axis=0)
    
    out[np.isnan(out)] = np.nan
    return out

#####
def norm_dataframe(df, data_type, tag_column, value_columns, norm_species, norm_method='median', trim_percentage=5):
    logging.info(f"Performing {data_type}-level normalization")
    if norm_species == 'ALL':
        sub_df = df.clone()
    else:
        species_tags = df.select(tag_column).with_columns(
            pl.col(tag_column).str.contains(norm_species).alias("match")
        )["match"]
        if species_tags.sum() == 0:
            logging.warning('WARNING: No matching species found in your input. Using all proteins')
            sub_df = df.clone()
        else:
            
            sub_df = df.filter(species_tags)
    
    mat = sub_df.select(value_columns)
    loadBias = get_loading_bias(mat, value_columns, data_type, trim_percentage, trim_percentage)
    mat_norm, normFactors = normalization(mat, trim_percentage, trim_percentage, norm_method)
    
    #
    logging.info("Loading bias before normalization:\n%s", 
                 loadBias.to_string(index=False))
    medians_df = mat.median().to_pandas().T.reset_index(drop=False)
    medians_df.columns = ['Samples', 'Medians']
    normFactors_df = pd.DataFrame(list(normFactors.items()), columns=['Samples', 'NormFactors'])
    medians_df = medians_df.merge(normFactors_df)
    medians_df['Samples'] = extract_unique_parts(medians_df['Samples'].to_list())
    logging.info("Median log2(Intensity) before normalization:\n%s", 
                 medians_df.to_string(index=False))
    medians_norm_df = mat_norm.median().to_pandas().T.reset_index(drop=False)
    medians_norm_df.columns = ['Samples', 'Medians']
    medians_norm_df['Samples'] = extract_unique_parts(medians_norm_df['Samples'].to_list())
    logging.info("Median log2(Intensity) after normalization:\n%s", 
                 medians_norm_df.to_string(index=False))

    df_norm = df.clone()
    for col in value_columns:
        df_norm = df_norm.with_columns(
            (pl.col(col) - normFactors[col]).alias(col)
        )
    return df_norm, normFactors


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


def get_loading_bias(mat, sample_names, data_type, trim_pct_upper, trim_pct_lower):
    mat_f = trim_matrix(mat, trim_pct_upper, trim_pct_lower)
    n = mat_f.height
    sm = 2 ** mat_f.mean().to_numpy()[0]
    msm = sm.mean()
    avg = sm / msm * 100
    sdVal = mat_f.std().to_numpy()[0]
    sd = ((2 ** sdVal - 2 ** (-sdVal)) / 2) * 100
    sem = sd / np.sqrt(n)

    loadBias = pd.DataFrame({
        "Samples": extract_unique_parts(sample_names),
        "Mean[%]": avg,
        "SD[%]": sd,
        "SEM[%]": sem,
        f"# {data_type}s": [n] * len(avg)
    })
    return loadBias


def normalization(mat, trim_pct_upper, trim_pct_lower, norm_method):
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

####
def filter_based_on_quality(quantity_df, quality_df, tag_column, value_columns, quality_cutoffs=[0.5, 0.75]):
    # Align quality matrix to quantity_df using tag_column
    quality_mat = quality_df.set_index(tag_column).loc[quantity_df[tag_column], value_columns].reset_index(drop=True)

    quantity_dfs_filtered = []
    for cutoff in quality_cutoffs:
        # Create a mask: True where quality > cutoff
        mask = quality_mat > cutoff
        
        # Keep rows with at least one value passing the cutoff
        valid_rows = mask.any(axis=1)
        
        # Filter quantity_df and apply mask
        quantity_df_f = quantity_df.loc[valid_rows].copy()
        quantity_df_f[value_columns] = quantity_df_f[value_columns].where(mask.loc[valid_rows])
        
        quantity_dfs_filtered.append(quantity_df_f)

    return quantity_dfs_filtered