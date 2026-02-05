# builtin
import logging
import warnings
from pathlib import Path
# external
import numpy as np
import pandas as pd
import polars as pl
# local
from jumpdiaq.arguments import Arguments
from jumpdiaq.utils import progress_bar, extract_unique_parts, missing_value_imputation

args = Arguments()
ShowProgress = args.show_progress
InterProgress = args.inter_progress


def get_quantity_quality_dfs(id_report_df, 
                             id_col,
                             sample_col,
                             secondary_id_col, 
                             quantity_col, 
                             quality_col, 
                             save_data=True,
                             out_dir="",
                             ):
    
    # Split and explode quantity and quality columns
    id_report_df_long = (
        id_report_df
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
        id_report_df_long.select([
            pl.col(id_col).cast(pl.Utf8).alias("target_id"),
            pl.col(sample_col).cast(pl.Utf8).alias("samples"),
            pl.col(secondary_id_col).cast(pl.Utf8).alias("source_id"),
            pl.col(quantity_col).alias("source_quantity")
        ])
    )

    # quality
    quality_df = (
        id_report_df_long.select([
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
                        sample_meta: pd.DataFrame,
                        sample_column='Sample', 
                        group_column='Group',
                        topN_rep=3, 
                        rep_abundance='mean',
                        xic_quality_cutoff=0.0,
                        allowed_xic_missing=0.5,
                        min_n_xic=2,
                        fill_missing_xic=True,
                        fill_method='global_min',
                        fill_factor=1.5
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
        
        quantity_mat_fill = quantity_mat.copy()
        if fill_missing_xic:
            if xic_quality_cutoff <= 0.0:
                missing_mask = (quality_mat < xic_quality_cutoff) | (quality_mat == 0.0)
            else:
                missing_mask = quality_mat < xic_quality_cutoff
            missing_mask = missing_mask | (quantity_mat == 0.0)
            quantity_mat_fill[missing_mask] = np.nan
            # remove rows with x% or more NaNs
            row_nan_percentage = np.isnan(quantity_mat_fill).sum(axis=1) / quantity_mat_fill.shape[1]
            quantity_mat_fill = quantity_mat_fill[row_nan_percentage < allowed_xic_missing]
            # fill missing value
            if quantity_mat_fill.shape[0] < min_n_xic:
                continue
            else:
                quantity_mat_fill = missing_value_imputation(quantity_mat_fill, 
                                                             sample_cols, 
                                                             sample_meta,
                                                             sample_column, 
                                                             group_column,
                                                             fill_method,
                                                             fill_factor)
        #
        quantity_summed[i, :] = median_summ(quantity_mat_fill, topN_rep, rep_abundance)
        #
        quality_mat_new = quality_mat.copy()
        quality_mat_new[quality_mat == 0.0] = np.nan
        quality_mat_new = quality_mat_new[~np.all(np.isnan(quality_mat_new), axis=1)]
        quality_mat_new[quality_mat_new < 0.0] = 0.01
        quality_mat_new = np.nan_to_num(quality_mat_new, nan=0.01)
        quality_summed[i, :] = topN(quality_mat_new, N=6, method="median")
        
    quantity_table = pl.DataFrame(quantity_summed, schema=sample_cols)
    quantity_table = quantity_table.with_columns(pl.Series(name="Precursor.Id", values=prec_ids))
    
    quality_table = pl.DataFrame(quality_summed, schema=sample_cols)
    quality_table = quality_table.with_columns(pl.Series(name="Precursor.Id", values=prec_ids))

    return quantity_table, quality_table

def extract_annotation(id_list, 
                       xic_quant_df: pl.DataFrame, 
                       id_col: str, 
                       annotation_cols: list[str] | None):
    if annotation_cols is None:
        annotation_cols = []

    all_columns = [id_col] + annotation_cols

    # Check all columns exist
    missing = [col for col in all_columns if col not in xic_quant_df.columns]
    if missing:
        raise ValueError(f"The input table has no column: {missing[0]}")

    # Filter rows where id_col is in id_list
    filtered = xic_quant_df.filter(pl.col(id_col).is_in(id_list))
    if filtered.is_empty():
        raise ValueError("Cannot find any of the given target IDs in the table.")

    # Select only relevant columns and drop duplicates
    tab = filtered.select(all_columns).unique()

    # Create an indexed DataFrame (simulate pandas .loc by join)
    id_df = pl.DataFrame({id_col: id_list})
    joined = id_df.join(tab, on=id_col, how="left")

    return joined

def quantify_from_xic(
        xic_quant_file, 
        sample_meta_file,
        id_col="Precursor.Id",
        sample_col="Run",
        secondary_id_col="MS2.XIC.Info",
        quantity_col="MS2.XIC.Quant.Adjusted",
        quality_col="MS2.XIC.Corr.Raw",
        annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
        meta_sample_col="Sample",
        meta_group_col="Group",
        out_dir="",
        topN_rep=3, 
        rep_abundance='mean',
        xic_quality_cutoff=0.0, 
        allowed_xic_missing=0.5,
        min_n_xic=2,
        fill_missing_xic=True,
        fill_method='local_min', # or 'local_mean'
        fill_factor=1.5
    ):
    
    columns = [sample_col, id_col, secondary_id_col, quality_col, quantity_col] + annotation_cols
    xic_quant_df = pl.read_parquet(Path(xic_quant_file), columns=columns)
    sample_meta_df = pd.DataFrame()
    if fill_missing_xic and fill_method in ['local_mean', 'local_min']:
        logging.info(f"Reading sample meta info from {sample_meta_file}")
        sample_meta_df = pd.read_csv(sample_meta_file)
    logging.info(f"ID column: {id_col}, Quantity column: {quantity_col}")
    
    # 1. extract XIC quantity and quality
    xic_quantity_df, xic_quality_df = get_quantity_quality_dfs(xic_quant_df, 
                                                       id_col,
                                                       sample_col,
                                                       secondary_id_col,
                                                       quantity_col,
                                                       quality_col,
                                                       save_data=True,
                                                       out_dir=out_dir)
    
    # 2. summarize to precursor quantity and quality
    prec_quantity_mtx, prec_quality_mtx = precursor_summarize(xic_quantity_df, 
                                                              xic_quality_df, 
                                                              sample_meta_df,
                                                              meta_sample_col,
                                                              meta_group_col,
                                                              topN_rep, 
                                                              rep_abundance,
                                                              xic_quality_cutoff,
                                                              allowed_xic_missing,
                                                              min_n_xic,
                                                              fill_missing_xic, 
                                                              fill_method,
                                                              fill_factor)

    # 3. add annotation
    extra_names = extract_annotation(prec_quantity_mtx["Precursor.Id"].to_list(), 
                                     xic_quant_df, 
                                     id_col,
                                     annotation_cols)
    
    final_quantity_table = extra_names.join(prec_quantity_mtx, on=id_col, how="inner")
    final_quality_table = extra_names.join(prec_quality_mtx, on=id_col, how="inner")  
    
    return final_quantity_table, final_quality_table
    
def quantify_precursors(xic_quant_file, 
                        sample_meta_file,
                        use_raw_ms1=False,
                        use_raw_ms2=True,
                        xic_quality_cutoff=0.0, 
                        allowed_xic_missing=0.5,
                        min_n_xic=2,
                        fill_missing_xic=True,
                        fill_method='local_min',
                        fill_factor=1.5,
                        prec_quant_level='ms2',
                        out_dir=""):
    
    # Choose between MS1 and MS2 precursor quantification
    if prec_quant_level in ['ms1', 'ms1_ms2']:
        # MS1-level quantification
        logging.info("Quantifying precursors at MS1-level")
        if use_raw_ms1:
            ms1_quantity_col = "MS1.XIC.Quant.Raw"
        else:
            ms1_quantity_col = "MS1.XIC.Quant.Adjusted"
        ms1_quantity_table, ms1_quality_table = quantify_from_xic(
                                                    xic_quant_file,
                                                    sample_meta_file,
                                                    id_col="Precursor.Id",
                                                    sample_col="Run",
                                                    secondary_id_col="MS1.XIC.Info",
                                                    quantity_col=ms1_quantity_col,
                                                    quality_col="MS1.XIC.Corr.Raw",
                                                    annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
                                                    out_dir=out_dir,
                                                    topN_rep=3,
                                                    rep_abundance='sum',
                                                    xic_quality_cutoff=xic_quality_cutoff,
                                                    allowed_xic_missing=allowed_xic_missing,
                                                    min_n_xic=min_n_xic,
                                                    fill_missing_xic=fill_missing_xic, 
                                                    fill_method=fill_method,
                                                    fill_factor=fill_factor
                                                    )
        ms1_quantity_table = ms1_quantity_table.with_columns(pl.lit('MS1').alias('MS_type'))
        ms1_quality_table = ms1_quality_table.with_columns(pl.lit('MS1').alias('MS_type'))
        ms1_quantity_table = ms1_quantity_table.with_columns([
            (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
        ])
        ms1_quality_table = ms1_quality_table.with_columns([
            (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
        ])
        
        if prec_quant_level == 'ms1':
            logging.info("Using only MS1-level precursor quantity for further analysis")
            return ms1_quantity_table, ms1_quality_table
        
    if prec_quant_level in ['ms2', 'ms1_ms2']:
        # MS2-level quantification
        logging.info("Quantifying precursors at MS2-level")
        if use_raw_ms2:
            ms2_quantity_col = "MS2.XIC.Quant.Raw"
        else:
            ms2_quantity_col = "MS2.XIC.Quant.Adjusted"
        ms2_quantity_table, ms2_quality_table = quantify_from_xic(
                                                    xic_quant_file, 
                                                    sample_meta_file,
                                                    id_col="Precursor.Id",
                                                    sample_col="Run",
                                                    secondary_id_col="MS2.XIC.Info",
                                                    quantity_col=ms2_quantity_col,
                                                    quality_col="MS2.XIC.Corr.Raw",
                                                    annotation_cols=['Protein.Group', 'Protein.Names', 'Genes'],
                                                    out_dir=out_dir,
                                                    topN_rep=3,
                                                    rep_abundance='mean',
                                                    xic_quality_cutoff=xic_quality_cutoff,
                                                    allowed_xic_missing=allowed_xic_missing,
                                                    min_n_xic=min_n_xic,
                                                    fill_missing_xic=fill_missing_xic, 
                                                    fill_method=fill_method,
                                                    fill_factor=fill_factor
                                                    )
        ms2_quantity_table = ms2_quantity_table.with_columns(pl.lit('MS2').alias('MS_type'))
        ms2_quality_table = ms2_quality_table.with_columns(pl.lit('MS2').alias('MS_type'))
        ms2_quantity_table = ms2_quantity_table.with_columns([
            (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
        ])
        ms2_quality_table = ms2_quality_table.with_columns([
            (pl.col("Precursor.Id").cast(pl.Utf8) + "_" + pl.col("MS_type").cast(pl.Utf8)).alias("Precursor.Id_MS_type")
        ])
        
        if prec_quant_level == 'ms2':
            logging.info("Using only MS2-level precursor quantity for further analysis")
            return ms2_quantity_table, ms2_quality_table
        
    if prec_quant_level == 'ms1_ms2':
        logging.info("Using a hybrid of MS1- and MS2-level quantity for further analysis")
        # Combine MS1 and MS2 quantification
        ms1_quality_table = ms1_quality_table.select(ms2_quality_table.columns)
        ms1_quantity_table = ms1_quantity_table.select(ms2_quantity_table.columns)
        ms1_ms2_quality_table = pl.concat([ms1_quality_table, ms2_quality_table], how="vertical")
        ms1_ms2_quantity_table = pl.concat([ms1_quantity_table, ms2_quantity_table], how="vertical")
        
        # Save to files
        #ms1_ms2_quantity_table.write_parquet(Path(out_dir) / "precursor_quantity_ms1_ms2_summ_table.parquet")
        #ms1_ms2_quality_table.write_parquet(Path(out_dir) / "precursor_quality_ms1_ms2_summ_table.parquet")
        
        return ms1_ms2_quantity_table, ms1_ms2_quality_table


def protein_summarize(prec_quantity_table, 
                      prec_quality_table, 
                      sample_cols, 
                      min_prec_quality):
    # filtering based on average quality
    prec_quality_table = prec_quality_table.with_columns(pl.mean_horizontal(sample_cols).alias("Average"))
    prec_quality_table = prec_quality_table.filter(pl.col("Average") >= min_prec_quality)
    
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
        
        prec_id_col = "Precursor.Id_MS_type"
        prec_ids = set(quantity_df_sub[prec_id_col].to_list()).intersection(
            quality_df_sub[prec_id_col].to_list()
        )
        if not prec_ids:
            continue
        prec_ids = list(prec_ids)
        
        quantity_df_sub = quantity_df_sub.filter(pl.col(prec_id_col).is_in(prec_ids)).sort(prec_id_col)
        quality_df_sub = quality_df_sub.filter(pl.col(prec_id_col).is_in(prec_ids)).sort(prec_id_col)
        
        # Summarization
        if len(prec_ids) >= 3:
            quantity_mat = quantity_df_sub.select(sample_cols).to_numpy()
            quality_mat = quality_df_sub.select(sample_cols).to_numpy()
            
            quantity_mat = quantity_mat[~np.all(np.isnan(quantity_mat), axis=1)]
            quality_mat = quality_mat[~np.all(np.isnan(quality_mat), axis=1)]
            
            quantity_summed[i, :] = median_summ(quantity_mat, topN_rep=3, rep_abundance='mean')
            quality_summed[i, :] = topN(quality_mat, N=3, method="mean")
        else:
            max_quality_prec = (quality_df_sub.filter(pl.col("Average") == pl.col("Average").max())
                                .select(["Precursor.Id_MS_type"])).item()
            quantity_summed[i, :] = quantity_df_sub.filter(pl.col("Precursor.Id_MS_type") == max_quality_prec).select(sample_cols).to_numpy()[0]
            quality_summed[i, :] = quality_df_sub.filter(pl.col("Precursor.Id_MS_type") == max_quality_prec).select(sample_cols).to_numpy()[0]
        
    quantity_table = pl.DataFrame(quantity_summed, schema=sample_cols)
    quantity_table = quantity_table.with_columns(pl.Series(name="Protein.Group", values=prot_ids))
    
    quality_table = pl.DataFrame(quality_summed, schema=sample_cols)
    quality_table = quality_table.with_columns(pl.Series(name="Protein.Group", values=prot_ids))

    return quantity_table, quality_table

def quantify_proteins(prec_quantity_table, 
                      prec_quality_table,
                      min_prec_quality):
    logging.info("Quantifying proteins")
    # Extract unique sample column names (excluding primary_id and id)
    other_cols = {'Precursor.Id',
                  'Protein.Group',
                  'Protein.Names',
                  'Genes',
                  'MS_type',
                  'Precursor.Id_MS_type'}
    sample_cols = [col for col in prec_quantity_table.columns if col not in other_cols]
    prot_quantity_table, prot_quality_table = protein_summarize(prec_quantity_table, 
                                                                prec_quality_table, 
                                                                sample_cols,
                                                                min_prec_quality)
    extra_names = extract_annotation(prot_quantity_table['Protein.Group'].to_list(), 
                                     prec_quantity_table, 
                                     id_col='Protein.Group',
                                     annotation_cols=['Protein.Names', 'Genes'])
    prot_quantity_table = extra_names.join(prot_quantity_table, on='Protein.Group', how="inner")   
    prot_quality_table = extra_names.join(prot_quality_table, on='Protein.Group', how="inner")  
    
    return prot_quantity_table, prot_quality_table


def filter_one_hit_wonder(prot_prec_pep_info, 
                          prot_quantity_table, 
                          prot_quality_table, 
                          sample_meta_file, 
                          sample_column='Sample', 
                          group_column='Group',
                          min_1hitw_quality=0.35, 
                          min_prec_count=2,
                          min_pep_count=1):
    # For one-hit wonder (protein with one precursor), filter based on quality
    prot_1_prec = prot_prec_pep_info['Protein.Group'][prot_prec_pep_info['n.Precursors'] == 1]
    prot_2more_precs = prot_prec_pep_info['Protein.Group'][prot_prec_pep_info['n.Precursors'] >= 2]
    logging.info(f"Found {len(prot_1_prec)} one-hit wonder proteins (i.e., proteins with only one precursor)")
    
    # first filtering
    prot_quality_1prec = prot_quality_table.filter(pl.col('Protein.Group').is_in(prot_1_prec)).to_pandas()
    group_mean_cols = []
    if sample_meta_file:
        sample_meta_df = pd.read_csv(sample_meta_file)
        group_dict = sample_meta_df.groupby(group_column)[sample_column].apply(list).to_dict()
        for group_name, cols in group_dict.items():
            mean_col_name = f'{group_name}_mean'
            prot_quality_1prec[mean_col_name] = prot_quality_1prec[cols].mean(axis=1)
            group_mean_cols.append(mean_col_name)
    else:
        prot_quality_1prec['row_mean'] = prot_quality_1prec.iloc[:, 3:].mean(axis=1)
        
    if min_1hitw_quality > 0.0:
        if sample_meta_file:
            # require at least 2 group means to meet the quality threshold
            is_high_quality = np.sum(prot_quality_1prec[group_mean_cols] >= min_1hitw_quality, axis=1) >= 2
            prot_1_prec_f = prot_quality_1prec.loc[is_high_quality, 'Protein.Group']
        else:
            is_high_quality = (prot_quality_1prec['row_mean'] >= min_1hitw_quality)
            prot_1_prec_f = prot_quality_1prec.loc[is_high_quality, 'Protein.Group']
    else:
        prot_1_prec_f = prot_quality_1prec['Protein.Group']
    
    logging.info(f"{len(prot_1_prec) - len(prot_1_prec_f)} low quality (<{min_1hitw_quality}) one-hit wonders were removed")
    prots_accept_1 = pd.concat([prot_2more_precs, prot_1_prec_f])
    prot_quantity_table_1st_f = prot_quantity_table.filter(pl.col('Protein.Group').is_in(prots_accept_1))
    
    # Further filtering
    logging.info(f"Filtering proteins based on the number of precursors (>={min_prec_count}) and peptides (>={min_pep_count})")
    accepted = (prot_prec_pep_info['n.Precursors'] >= min_prec_count) & (prot_prec_pep_info['n.Peptides'] >= min_pep_count)
    prots_accept_2 = prot_prec_pep_info['Protein.Group'][accepted]
    prot_quantity_table_2nd_f = prot_quantity_table_1st_f.filter(pl.col('Protein.Group').is_in(prots_accept_2))
    
    return prot_quantity_table_1st_f, prot_quantity_table_2nd_f


############################ summarization methods  ###########################
def log2_safe(mat):
    """Safely apply log2 transformation, replacing non-positive values with NaN"""
    result = np.full_like(mat, np.nan, dtype=float)
    mask = mat > 0
    result[mask] = np.log2(mat[mask])
    return result

def median_summ(mat, topN_rep=3, rep_abundance='mean'):
    """
    Summarize replicate measurements using median after normalization
    
    Parameters:
    -----------
    mat : array-like
        Input matrix (rows=ions, columns=runs)
    topN_rep : int
        Number of top ions to use for abundance calculation
    
    Returns:
    --------
    out : array
        Summarized values
    """
    if mat.shape[0] == 1:
        return mat[0, :]
    
    # log2 transformation
    mat = log2_safe(mat)
    
    # Get relative ratios (suppress empty slice warning if expected)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        rowMeans = np.nanmean(mat, axis=1)
    
    # Handle case where all rowMeans are NaN
    valid_means = rowMeans[~np.isnan(rowMeans)]
    if len(valid_means) == 0:
        # All rows are NaN, return NaN array
        return np.full(mat.shape[1], np.nan)
    
    # Calculate representative abundance from top N replicates
    topN_rep = min(topN_rep, len(valid_means))  # Handle case where fewer valid rows than topN_rep
    used_means = sorted(valid_means, reverse=True)[:topN_rep]
    if rep_abundance == 'mean':
        repAbundance = np.log2(np.mean(2 ** np.array(used_means)))
    elif rep_abundance == 'sum':
        repAbundance = np.log2(np.sum(2 ** np.array(used_means)))
    
    # Center each row by its mean
    mat = mat - rowMeans[:, np.newaxis]
    
    # Compute median of each column (only for non-all-NaN columns)
    medians = np.full(mat.shape[1], np.nan)
    valid_cols = ~np.all(np.isnan(mat), axis=0)
    
    if np.any(valid_cols):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            medians[valid_cols] = np.nanmedian(mat[:, valid_cols], axis=0)
    
    # Final summarized values (transform back from log space)
    out = np.power(2, medians + repAbundance)
    
    return out

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

################################## Normalization ##############################
def norm_dataframe(df, data_type, tag_column, value_columns, normalization, norm_species, norm_method='median', trim_percentage=5):
    if normalization:
        logging.info(f"Performing {data_type.lower()}-level normalization")
        if norm_species == 'ALL':
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
    loadBias = get_loading_bias(mat, value_columns, data_type, trim_percentage, trim_percentage)
    medians_df = mat.median().to_pandas().T.reset_index(drop=False)
    medians_df.columns = ['Samples', 'Medians']
    
    # print out info
    def indent_df(df, spaces=4, index=False):
        """Return a DataFrame string with left indentation for logging."""
        indent = " " * spaces
        df_str = df.to_string(index=index)
        return "\n".join(indent + line for line in df_str.splitlines())
    
    logging.info("Loading bias before normalization:\n%s", 
                 indent_df(loadBias, spaces=4, index=False))
    
    if normalization:
        # get norm factors
        mat_norm, normFactors = normalize(mat, trim_percentage, trim_percentage, norm_method)
        normFactors_df = pd.DataFrame(list(normFactors.items()), columns=['Samples', 'NormFactors'])
        medians_df = medians_df.merge(normFactors_df)
        medians_df['Samples'] = extract_unique_parts(medians_df['Samples'].to_list())
        logging.info("Median log2(Intensity) before normalization:\n%s", 
                     indent_df(medians_df, spaces=4, index=False))
        
        medians_norm_df = mat_norm.median().to_pandas().T.reset_index(drop=False)
        medians_norm_df.columns = ['Samples', 'Medians']
        medians_norm_df['Samples'] = extract_unique_parts(medians_norm_df['Samples'].to_list())
        logging.info("Median log2(Intensity) after normalization:\n%s", 
                     indent_df(medians_norm_df, spaces=4, index=False))
        
        # normalization of the whole df
        df_norm = df.with_columns([
            (pl.col(col) * (2 ** -normFactors[col])).alias(col)
            for col in value_columns
        ])
    else:
        medians_df['Samples'] = extract_unique_parts(medians_df['Samples'].to_list())
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
