# builtin
import re
import logging
from pathlib import Path
# external
import numpy as np
import pandas as pd
#from Bio import SeqIO
# local
from jumplion.arguments import Arguments
from jumplion.precursor import get_aas
from jumplion.utils import remove_extension, get_file_name, indent_df, logging_histogram
from jumplion.calibration import refine_spline, calc_spline_vectorized
from jumplion.coordinate_validation import (ModelConfig, 
                                            run_rt_prediction,
                                            run_im_prediction)

args = Arguments()
SaveSupplements = args.save_supplements
MinRunsQuant = args.min_runs_quant
PredictRTIM = args.predict_rt_im
ValidateRTIM = False
MinRunsModelFrac = args.min_runs_model_frac
MinRunsCalFrac = args.min_runs_cal_frac
MaxQvalueCal = args.max_qvalue_cal
MaxMassCal = args.max_mass_cal

def process_id_results(id_file, 
                       lib_file, 
                       ms_files,
                       ms_file_type,
                       out_dir,
                       save_supplements=SaveSupplements,
                       min_runs_quant=MinRunsQuant,
                       predict_rt_im=PredictRTIM,
                       validate_rt_im=ValidateRTIM,
                       min_runs_model_frac=MinRunsModelFrac,
                       min_runs_cal_frac=MinRunsCalFrac,
                       max_qvalue_cal=MaxQvalueCal,
                       max_mass_cal=MaxMassCal):
    
    out_int_dir = Path(out_dir) / 'intermediate'
    out_pub_dir = Path(out_dir) / 'publication'
    
    #################################
    # 0. read id report file        #
    #################################
    required_columns = ['Run', 'Precursor.Id', 'RT', 'Q.Value', 'Precursor.Quantity']
    is_tims = ms_file_type == 'd'
    if is_tims:
        required_columns = required_columns + ['IM']
    
    logging.info(f"Reading identification report file: {id_file}")
    id_df_top1000 = pd.read_parquet(id_file).head(1000)
    id_df_top1000.to_csv(out_int_dir / "diann_report_top1000.tsv", sep = '\t', index = False)
    id_df = pd.read_parquet(id_file, columns=required_columns)
    id_df_top5 = id_df.head(5).copy()
    id_df_top5.loc[:,'Run'] = id_df_top5['Run'].str[0:10] + '...' + id_df_top5['Run'].str[-10:]
    logging.info(f"Loaded {id_df.shape[0]} rows and {id_df.shape[1]} columns, e.g.,\n%s", 
                 indent_df(id_df_top5.rename(columns=lambda c: f"{c} ({id_df[c].dtype})"), 
                           spaces=4, index=False))
    n_unique_prec = id_df["Precursor.Id"].nunique()
    n_unique_run = id_df["Run"].nunique()
    logging.info(f"ID report contains {n_unique_prec} unique precursors in {n_unique_run} runs")
    if id_df['RT'].isna().sum():
        logging.warn("NA found in the RT column.")
    
    logging.info(f"Filtering report based on input runs (n = {len(ms_files)})")
    run_names = [remove_extension(get_file_name(ms_file)) for ms_file in ms_files]
    id_df = id_df[id_df['Run'].isin(run_names)]
    logging.info(f"Resulting in {id_df.shape[0]} rows and {id_df.shape[1]} columns")

    ############################
    # 1. get rt and im matrix  #
    ############################
    rt_mat = id_df.pivot(index='Precursor.Id', columns='Run', values='RT')
    # check missing value percentage
    logging.info("Distribution of the number of runs a precursor is detected in:")
    n_runs_detected = rt_mat.shape[1] - rt_mat.isna().sum(axis=1)
    logging_histogram(n_runs_detected, rt_mat.shape[1])
    # 
    missing_status = pd.DataFrame(
        np.where(rt_mat.isna(), "missing", "reported"),
        index=rt_mat.index,
        columns=rt_mat.columns
    )
    missing_status.to_csv(out_int_dir / "missing_status.tsv", sep = '\t')
    
    if save_supplements:
        out_rt_dir = out_int_dir / 'rt_prediction'
        out_rt_dir.mkdir(parents=True, exist_ok=True)
        rt_mat.to_csv(out_rt_dir / "rt_mat_before_prediction.tsv", sep = '\t')
    if is_tims:
        im_mat = id_df.pivot(index='Precursor.Id', columns='Run', values='IM')
        im_mat = im_mat.loc[rt_mat.index, :]
        if save_supplements:
            out_im_dir = out_int_dir / 'im_prediction'
            out_im_dir.mkdir(parents=True, exist_ok=True)
            im_mat.to_csv(out_im_dir / "im_mat_before_prediction.tsv", sep = '\t')
    else:
        im_mat = None
    
    ############################
    # 2. load spectral library #
    ############################
    # read library file
    logging.info(f"Reading spectral library: {lib_file}")
    lib_df = pd.read_parquet(lib_file)
    # Remove decoys
    for col_index, header in enumerate(lib_df.columns):
        if re.sub(r"[\s\._-]", "", header.lower()) in [h.lower() for h in ("Decoy", "isDecoy")]:
            decoy_col_name = header
            decoy_values = ['1', 't', 'T']
            decoy_mask = lib_df[decoy_col_name].astype(str).isin(decoy_values)
            n_decoys = decoy_mask.sum()
            logging.warning(f"WARNING: the library contains {n_decoys} decoy precursors, which will be removed")
            lib_df = lib_df[~decoy_mask].copy()
            break
    if "Precursor.Id" in lib_df.columns:
        n_unique = lib_df["Precursor.Id"].nunique()
        logging.info(f"Library contains {n_unique} unique precursors")
        lib_n_frags = lib_df.groupby('Precursor.Id').size().reset_index(name='n_frags')
        n_frags_dist = pd.DataFrame(lib_n_frags['n_frags'].value_counts()).reset_index(names='# Fragments')
        logging.info("Distribution of the number of fragments for a precursor\n%s", 
                     indent_df(n_frags_dist, spaces=4, index=False))
    else:
        raise ValueError("The library dose not contain the Precursor.Id column.")
    
    # extract library RT and IM
    prec_cols = ['Precursor.Id', 'Modified.Sequence', 'RT', 'Precursor.Mz']
    if is_tims:
        lib_df_uniq_prec = lib_df[prec_cols + ['IM']].copy()
    else:
        lib_df_uniq_prec = lib_df[prec_cols].copy()
    lib_df_uniq_prec.drop_duplicates(subset=['Precursor.Id'], inplace=True, ignore_index=True)
    lib_df_uniq_prec = lib_df_uniq_prec.merge(lib_n_frags)
    
    # extract protein info 
    prot_cols = ['Protein.Group', 'Protein.Ids', 'Protein.Names', 'Genes']
    prec_prot_cols = ['Precursor.Id'] + prot_cols
    #
    prot_df = lib_df[prot_cols].copy()
    prot_df = prot_df.drop_duplicates(ignore_index=True)
    prot_df.to_csv(out_int_dir / "prot_info.tsv", sep='\t', index=False)
    #
    prec_prot_df = lib_df[prec_prot_cols].copy()
    prec_prot_df = prec_prot_df.drop_duplicates(ignore_index=True)
    n_duplicated = prec_prot_df['Precursor.Id'].duplicated().sum()
    if n_duplicated > 0:
        logging.info(f"Found cases in which one precursor was assigned to multiple proteins. Total number: {n_duplicated}")
        dups = prec_prot_df[prec_prot_df['Precursor.Id'].duplicated(keep=False), ['Precursor.Id', 'Protein.Names', 'Genes']]
        logging.info(indent_df(dups, spaces=4, index=False))
        prec_prot_df = prec_prot_df.drop_duplicates(subset=['Precursor.Id'], ignore_index=True)
    prec_prot_df.to_csv(out_int_dir / "prec_prot_relation.tsv", sep='\t', index=False)
    
    #######################
    # 3. extract DIA-NN precursor/protein quantities #
    #######################
    # extract precursor quantities
    quant_df = pd.read_parquet(id_file, columns=['Run', 
                                                 'Precursor.Id', 'Precursor.Quantity', 'Precursor.Normalised',
                                                 'Protein.Group', 'PG.MaxLFQ'])
    quant_df = quant_df[quant_df['Run'].isin(run_names)]
    #
    prec_quant_mat = quant_df.pivot(index='Precursor.Id', columns='Run', values='Precursor.Quantity')
    prec_quant_mat = prec_prot_df.merge(prec_quant_mat, on='Precursor.Id', how='inner')
    prec_quant_mat.to_csv(out_pub_dir / "prec_matrix_diann.tsv", sep = '\t', index=False)
    #
    prec_quant_norm_mat = quant_df.pivot(index='Precursor.Id', columns='Run', values='Precursor.Normalised')
    prec_quant_norm_mat = prec_prot_df.merge(prec_quant_norm_mat, on='Precursor.Id', how='inner')
    prec_quant_norm_mat.to_csv(out_pub_dir / "prec_matrix_diann_norm.tsv", sep = '\t', index=False)
    #
    prot_quant_df = quant_df[['Run', 'Protein.Group', 'PG.MaxLFQ']].drop_duplicates()
    prot_quant_mat = prot_quant_df.pivot(index='Protein.Group', columns='Run', values='PG.MaxLFQ')
    prot_quant_mat = prot_df.drop(columns='Protein.Ids').drop_duplicates().merge(prot_quant_mat, on='Protein.Group', how='inner')
    prot_quant_mat.to_csv(out_pub_dir / "prot_matrix_diann.tsv", sep = '\t', index=False)
    
    #######################
    # 3. RT/IM prediction #
    #######################
    if predict_rt_im:
        logging.info("Predicting precursor RTs (and IMs)") 
        
        config = ModelConfig(
            coordinate_label="RT",
            max_segments=30,
            min_run_fraction=min_runs_model_frac,
            adjustment_mode="clip",
        )
        
        _, _, _, rt_prediction = run_rt_prediction(
            rt_mat=rt_mat,
            lib_df_uniq_prec=lib_df_uniq_prec,
            refine_spline_fn=refine_spline,
            calc_spline_vectorized_fn=calc_spline_vectorized,
            out_dir=out_rt_dir,
            config=config,
            run_validation=validate_rt_im,
            save_plots=save_supplements and validate_rt_im,
        )
        rt_mat_pred = rt_prediction.final_coordinate_matrix
        
        if is_tims:
            config = ModelConfig(
                coordinate_label="IM",
                max_segments=5,
                min_run_fraction=min_runs_model_frac,
                adjustment_mode="median",
            )
            _, _, _, im_prediction = run_im_prediction(
                im_mat=im_mat,
                lib_df_uniq_prec=lib_df_uniq_prec,
                fit_fn=refine_spline,
                predict_fn=calc_spline_vectorized,
                out_dir=out_im_dir,
                config=config,
                run_validation=validate_rt_im,
                save_plots=save_supplements and validate_rt_im,
            )
            im_mat_pred = im_prediction.final_coordinate_matrix
    
        # combine together
        prec_rt_info = rt_mat_pred.add_suffix('_RT')
        if is_tims:
            prec_rt_info = pd.concat([prec_rt_info, im_mat_pred.add_suffix('_IM')], axis=1) 
    else:
        logging.info("Prediction of missing precursor RTs (and IMs) is not performed\n") 
        # combine together
        prec_rt_info = rt_mat.add_suffix('_RT')
        if is_tims:
            prec_rt_info = pd.concat([prec_rt_info, im_mat.add_suffix('_IM')], axis=1) 
    
    #############################################
    # 5. Choose precursors for mass calibration #
    #############################################
    min_runs_cal = int(np.ceil(min_runs_cal_frac * len(ms_files)))
    id_df_cal = id_df.loc[id_df['Q.Value'] < max_qvalue_cal, ['Precursor.Id', 'Run', 'Q.Value']]
    q_mat_cal = id_df_cal.pivot(index='Precursor.Id', columns='Run', values='Q.Value')
    q_mat_cal = q_mat_cal[(q_mat_cal.shape[1] - q_mat_cal.isna().sum(axis=1)) >= min_runs_cal]
    logging.info(f"{len(q_mat_cal)} precursors with Q-values < {max_qvalue_cal} are detected in at least (>=) {min_runs_cal} runs")
    q_means = q_mat_cal.mean(axis=1, numeric_only=True)
    q_means_topN = q_means.nsmallest(max_mass_cal)
    mass_cal_precs = q_means_topN.index.values
    logging.info(f"The top {len(mass_cal_precs)} (ordered by average Q-value) of them are selected for mass calibration\n")
    
    # mark precursors used for calibration
    lib_df_uniq_prec['mass_cal'] = 0
    lib_df_uniq_prec.loc[lib_df_uniq_prec['Precursor.Id'].isin(mass_cal_precs), 'mass_cal'] = 1
    lib_df_uniq_prec = lib_df_uniq_prec.set_index('Precursor.Id').reindex(prec_rt_info.index)
    prec_rt_info = pd.concat([prec_rt_info, lib_df_uniq_prec['mass_cal']], axis=1)    
    
    
    ###########################################
    # 6. Choose precursors for quantification #
    ###########################################
    if min_runs_quant == 0:
        min_runs_quant = 3
    non_missing_pct_lower = 0.0
    non_missing_pct_upper = 1.0
    min_runs_quant = max(min_runs_quant, len(ms_files) * non_missing_pct_lower)
    logging.info(f"Filtering precursors detected in at least (>=) {min_runs_quant} runs and with a non-missing percentaget between {non_missing_pct_lower*100}% and {non_missing_pct_upper*100}% for re-quantification")
    rt_mat_f = rt_mat[(n_runs_detected >= min_runs_quant) & (n_runs_detected < len(ms_files) * non_missing_pct_upper)]
    logging.info(f"{len(rt_mat_f)} precursors are included for re-quantification")
    
    # filter and save new library
    combined_precs = rt_mat_f.index.union(q_means_topN.index).values
    lib_df = lib_df[lib_df['Precursor.Id'].isin(combined_precs)]
    lib_file_filtered = out_int_dir / 'speclib_filtered.parquet'
    lib_df.to_parquet(lib_file_filtered, index=False)
    logging.info(f"Filtered spectral library saved to {lib_file_filtered}\n")
    
    prec_rt_info['re_quan'] = 0
    
    #prec_rt_info.loc[combined_precs, 're_quan'] = 1
    prec_rt_info.loc[rt_mat_f.index.values, 're_quan'] = 1
    prec_rt_info.to_csv(out_int_dir / "prec_rt_info.tsv", sep = '\t')
    

    
    return prec_rt_info, missing_status


def get_prot_prec_pep_info(df):
    df['Stripped.Sequence'] = df['Precursor.Id'].apply(get_aas)
    #
    prot_pep_info = df.drop(columns='Precursor.Id').drop_duplicates(ignore_index=True)
    prot_pep_info_grouped = prot_pep_info.groupby('Protein.Group', as_index=False).agg({'Stripped.Sequence': lambda x: ', '.join(x)})
    prot_pep_info_grouped.rename(columns={'Stripped.Sequence': 'Peptides'}, inplace=True)
    prot_pep_info_grouped["n.Peptides"] = prot_pep_info_grouped["Peptides"].apply(lambda x: len(x.split(', ')))
    #
    prot_prec_info = df.drop(columns='Stripped.Sequence').drop_duplicates(ignore_index=True)
    prot_prec_info_grouped = prot_prec_info.groupby('Protein.Group', as_index=False).agg({'Precursor.Id': lambda x: ', '.join(x)})
    prot_prec_info_grouped.rename(columns={'Precursor.Id': 'Precursors'}, inplace=True)
    prot_prec_info_grouped["n.Precursors"] = prot_prec_info_grouped["Precursors"].apply(lambda x: len(x.split(', ')))
    #
    prot_prec_pep_info = prot_prec_info_grouped.merge(prot_pep_info_grouped)
    
    return prot_prec_pep_info
    
    
# def calculate_protein_coverage(prot_prec_pep_info, fasta_files): 
#     prot_prec_pep_info.insert(1, 'Protein.Group.Rep', prot_prec_pep_info['Protein.Group'].str.split(pat=';', n=1, expand=True)[0])
#     #
#     prot_fas_info = extract_prot_info_from_FASTAs(fasta_files, prot_prec_pep_info['Protein.Group.Rep'].values.tolist())
#     #prot_fas_info.drop(columns=['Protein.Name','Database'], inplace = True)
#     prot_fas_info.rename(columns={"Protein.Id": "Protein.Group.Rep"}, inplace = True)
#     prot_fas_info = prot_fas_info.drop_duplicates(subset=['Protein.Group.Rep'], ignore_index=True)
#     #
#     prot_prec_pep_info = prot_fas_info.merge(prot_prec_pep_info, how = 'right')
#     prot_prec_pep_info.drop(columns=['Protein.Group.Rep'], inplace = True)
#     #
#     (prot_prec_pep_info['Peptide.Positions'], 
#      prot_prec_pep_info['Coverage.Percentage']) = map_peptides(prot_prec_pep_info['Sequence'].tolist(), 
#                                                                prot_prec_pep_info['Peptides'].tolist())
    
#     return prot_prec_pep_info


# def map_peptides(protein_seqs, peptide_seqs):
#     results_positions = []
#     results_coverage = []

#     for protein, peptides_str in zip(protein_seqs, peptide_seqs):
#         if type(protein) == str:
#             peptides = peptides_str.split(', ')
#             positions = []
#             covered = set()
    
#             for peptide in peptides:
#                 # Use while loop to find all occurrences
#                 start = 0
#                 while True:
#                     idx = protein.find(peptide, start)
#                     if idx == -1:
#                         break
#                     end = idx + len(peptide)
#                     positions.append((idx + 1, end))  # 1-based position
#                     covered.update(range(idx, end))
#                     start = idx + 1
    
#             coverage_rate = len(covered) / len(protein) if protein else 0
#             results_positions.append(positions)
#             results_coverage.append(np.round(coverage_rate*100, 2))
#         else:
#             results_positions.append([])
#             results_coverage.append(0.0)

#     return results_positions, results_coverage


# def extract_from_single_fasta(fasta_file, protein_ids, extract_sequence=True):
#     logging.info(f"Extracting protein information from FASTA file: {fasta_file}")

#     prot_info = {}
#     with open(fasta_file) as fasta:
#         records = list(SeqIO.parse(fasta, 'fasta'))

#         for record in records:
#             header = record.description
#             temp = header.split(" ", 1)
#             prot_id = temp[0].split("|")[1]
#             if prot_id in protein_ids:
#                 prot_data = {
#                     'Accession.Number': temp[0],
#                     'Protein.Description': temp[1] if len(temp) > 1 else ''
#                 }
#                 if extract_sequence:
#                     prot_data['Sequence'] = str(record.seq)

#                 prot_info[prot_id] = prot_data
#     return prot_info


# def extract_prot_info_from_FASTAs(fasta_files, protein_ids, extract_sequence=True, out_file=0):
#     collected_info = {}
#     for protein_id in protein_ids:
#         collected_info[protein_id] = {
#             'Accession.Number': '',
#             'Protein.Description': '',
#             'Sequence': '' if extract_sequence else ''
#         }
#     if len(fasta_files) == 1:
#         prot_info = extract_from_single_fasta(fasta_files[0], protein_ids, extract_sequence)
#     else:
#         prot_info = {}
#         for fasta_file in fasta_files:
#             temp_info = extract_from_single_fasta(fasta_file, protein_ids, extract_sequence)
#             prot_info.update(temp_info)

#     for prot_id, info in prot_info.items():
#         collected_info[prot_id] = info

#     prot_info_df = pd.DataFrame.from_dict(collected_info, orient='index').reset_index()
#     prot_info_df.rename(columns={'index': 'Protein.Id'}, inplace=True)

#     if out_file != 0:
#         prot_info_df.to_csv(out_file, index=False)
#     return prot_info_df


# def add_all_protein_names(df, fasta_files, id_col='Protein.Ids', extract_sequence=False, out_file=0):
#     """
#     Splits semicolon-separated ids in `id_col`, looks up every individual id
#     in the given FASTA file(s), and adds an 'Protein.Names.All' column with
#     all corresponding protein names joined by ';' (in the same order as
#     the ids in id_col).
#     """
#     # 1. Collect every individual protein id from the (possibly multi-id) column
#     split_ids = df[id_col].astype(str).str.split(';')
#     all_ids = set(pid for sublist in split_ids for pid in sublist)

#     # 2. Extract info for all of them at once
#     prot_info_df = extract_prot_info_from_FASTAs(
#         fasta_files, all_ids, extract_sequence=extract_sequence
#     )

#     # 3. Build id -> name lookup.
#     #    'Accession.Number' looks like 'sp|P14550|AK1A1_HUMAN' -> take last field
#     #    Fall back to '' if not found in the FASTA.
#     def name_from_accession(acc):
#         if not acc:
#             return ''
#         parts = acc.split('|')
#         return parts[-1] if len(parts) > 1 else acc

#     id_to_name = dict(
#         zip(prot_info_df['Protein.Id'], prot_info_df['Accession.Number'].apply(name_from_accession))
#     )

#     # 4. Map back onto each row, preserving id order, joined by ';'
#     def lookup_names(id_string):
#         ids = str(id_string).split(';')
#         names = [id_to_name.get(pid, '') for pid in ids]
#         return ';'.join(names)

#     df = df.copy()
#     df['Protein.Names.All'] = df[id_col].apply(lookup_names)

#     if out_file != 0:
#         df.to_csv(out_file, index=False)

#     return df