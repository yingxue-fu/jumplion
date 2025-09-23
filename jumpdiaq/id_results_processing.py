# builtin
import re
import json
import random
import logging
from pathlib import Path
# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
# local
from jumpdiaq.arguments import Arguments
from jumpdiaq.read_ms_files import get_cycle_time
from jumpdiaq.utils import remove_extension, get_extension, get_file_name
from jumpdiaq.calibration import (refine_spline, calc_spline_vectorized, 
                                  convert_to_arrays, check_model_accuracy, 
                                  spline_fitting_plot)

args = Arguments()
SaveSupplements = args.save_supplements
MinRunsQuant = args.min_runs_quant
MinRunsModel = args.min_runs_model
MinRunsCal = args.min_runs_cal
MaxQvalueCal = args.max_qvalue_cal
MaxMassCal = args.max_mass_cal
MinXICPoints = 21

def process_id_results(id_file, 
                       lib_file, 
                       ms_files,
                       out_dir,
                       save_supplements=SaveSupplements,
                       min_runs_quant=MinRunsQuant,
                       min_runs_model=MinRunsModel,
                       min_runs_cal=MinRunsCal,
                       max_qvalue_cal=MaxQvalueCal,
                       max_mass_cal=MaxMassCal,):
    out_int_dir = Path(out_dir) / 'intermediate'
    
    ################################################
    # 0. read id report and spectral library files #
    ################################################
    # read id report file
    logging.info(f"Reading identification report file: {id_file}")
    required_columns = ['Run', 'Precursor.Id', 'RT', 'Q.Value']
    is_tims = False
    if get_extension(ms_files[0]) == 'd':
        is_tims = True
        required_columns = required_columns + ['IM']
        
    id_df = pd.read_parquet(id_file, columns=required_columns)
    run_names = [remove_extension(get_file_name(ms_file)) for ms_file in ms_files]
    id_df = id_df[id_df['Run'].isin(run_names)]
    
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
        
    ###########################################
    # 1. Choose precursors for quantification #
    ###########################################
    logging.info(f"Filtering precursors detected in at least (>=) {min_runs_quant} runs for quantification")
    # get rt and im matrix
    rt_mat = id_df.pivot(index='Precursor.Id', columns='Run', values='RT')
    rt_mat = rt_mat[(rt_mat.shape[1] - rt_mat.isna().sum(axis=1)) >= min_runs_quant]
    if save_supplements:
        out_rt_dir = out_int_dir / 'rt_prediction'
        out_rt_dir.mkdir(parents=True, exist_ok=True)
        rt_mat.to_csv(out_rt_dir / "rt_mat_before_prediction.csv")
    if is_tims:
        im_mat = id_df.pivot(index='Precursor.Id', columns='Run', values='IM')
        im_mat = im_mat.loc[rt_mat.index, :]
        if save_supplements:
            out_im_dir = out_int_dir / 'im_prediction'
            out_im_dir.mkdir(parents=True, exist_ok=True)
            im_mat.to_csv(out_im_dir / "im_mat_before_prediction.csv")
    else:
        im_mat = None
    logging.info(f"{len(rt_mat)} precursors are included for quantification")
    
    # filter and save new library for re-quantification
    lib_df = lib_df[lib_df['Precursor.Id'].isin(rt_mat.index.values)]
    if is_tims:
        lib_df_uniq_prec = lib_df[['Precursor.Id', 'RT', 'IM', 'Precursor.Mz']].copy()
    else:
        lib_df_uniq_prec = lib_df[['Precursor.Id', 'RT', 'Precursor.Mz']].copy()
    lib_df_uniq_prec.drop_duplicates(subset=['Precursor.Id'], inplace=True, ignore_index=True)
    
    lib_file_filtered = out_int_dir / 'speclib_filtered.parquet'
    lib_df.to_parquet(lib_file_filtered, index=False)
    logging.info(f"Filtered spectral library saved to {lib_file_filtered}\n")
    
    #######################
    # 2. RT/IM prediction #
    #######################
    logging.info("Predicting precursor RTs (and IMs)") 
    rt_dict, rt_segments = collect_model_data(rt_mat,  
                                              min_runs_model,
                                              lib_df_uniq_prec,
                                              data_type="rt")
    rt_mat_pred, model_accs = modeling_and_prediction(rt_dict, rt_segments, rt_mat, 
                                                      lib_df_uniq_prec, 
                                                      data_type="rt", 
                                                      out_dir=out_int_dir, 
                                                      save_model=True,
                                                      save_plot=save_supplements)
    rt_mat_pred = adjust_prediction_outliers(rt_mat_pred, rt_mat, n_iter=5, save_plot=False)
    if save_supplements:
        rt_mat_pred.to_csv(out_rt_dir / "rt_mat_after_prediction.csv")
    
    if is_tims:
        im_dict, im_segments = collect_model_data(im_mat,  
                                                  min_runs_model,
                                                  lib_df_uniq_prec,
                                                  data_type="im")
        im_mat_pred, _ = modeling_and_prediction(im_dict, im_segments, im_mat, 
                                                 lib_df_uniq_prec, 
                                                 data_type="im",
                                                 out_dir=out_int_dir, 
                                                 save_model=True,
                                                 save_plot=save_supplements)
        if save_supplements:
            im_mat_pred.to_csv(Path(out_im_dir) / "im_mat_after_prediction.csv")
    
    #############################################
    # 3. Choose precursors for mass calibration #
    #############################################
    id_df_cal = id_df[id_df['Q.Value'] < max_qvalue_cal]
    rt_mat_cal = rt_mat[(rt_mat.shape[1] - rt_mat.isna().sum(axis=1)) >= min_runs_cal]
    mass_cal_precs = list(set(rt_mat_cal.index.values).intersection(set(id_df_cal['Precursor.Id'].drop_duplicates())))
    logging.info(f"{len(mass_cal_precs)} precursors with Q-values < {max_qvalue_cal} are detected in at least (>=) {min_runs_cal} runs")
    random.seed(42)
    mass_cal_precs = random.sample(mass_cal_precs, min(len(mass_cal_precs), max_mass_cal))
    logging.info(f"{len(mass_cal_precs)} of them are randomly selected for mass calibration\n")
    
    # mark precursors used for calibration
    lib_df_uniq_prec['mass_cal'] = 0
    lib_df_uniq_prec.loc[lib_df_uniq_prec['Precursor.Id'].isin(mass_cal_precs), 'mass_cal'] = 1
    lib_df_uniq_prec = lib_df_uniq_prec.set_index('Precursor.Id').reindex(rt_mat_pred.index)
    
    rt_mat_pred = rt_mat_pred.add_suffix('_RT')
    prec_rt_info = pd.concat([rt_mat_pred, lib_df_uniq_prec['mass_cal']], axis=1)    
    if is_tims:
        im_mat_pred = im_mat_pred.add_suffix('_IM')
        prec_rt_info = pd.concat([prec_rt_info, im_mat_pred], axis=1) 
    prec_rt_info.to_csv(out_int_dir / "prec_rt_info.csv")
    
    #########################################
    # 4. determine the number of XIC points #
    #########################################
    if args.n_xic_points == 0:
        logging.info("Determining the number of data points to extract for an XIC")
        # 
        cycle_time = get_cycle_time(ms_files[0])
        logging.info(f"Cycle time: {cycle_time} ({(cycle_time*60):3g} seconds)")
        # 
        average_model_acc = np.mean(model_accs)
        average_model_acc_n_points = int(np.round(average_model_acc / cycle_time))
        logging.info(f"Average RT prediction accuracy in terms of the number of data points in XICs: {average_model_acc_n_points}")
        # 
        args.n_xic_points = max(MinXICPoints, average_model_acc_n_points * 4 + 1)
    elif args.n_xic_points < MinXICPoints and args.n_xic_points > 0:
        logging.warning(f"WARNING: the provided number ({args.n_xic_points}) of data points to extract for an XIC may be too small")
        args.n_xic_points = MinXICPoints
    logging.info(f"Number of data points to extract for a XIC set to: {args.n_xic_points}\n")
    
    return prec_rt_info


def collect_model_data(mat, 
                       min_runs_model,
                       lib_df_uniq_prec, 
                       data_type="rt", # or "im"
                       trim_pct=5, 
                       max_segments=30, 
                       min_points_per_bin=20,
                       ):
            
    ######################################################
    # Choose precursors for building RT prediction model #
    ######################################################
    # only keep precursors without missing values
    #mat = mat[mat.isna().sum(axis=1) == 0]
    logging.info(f"Precursors detected in at least (>=) {min_runs_model} runs are used for building {data_type.upper()} prediction model.")
    mat = mat[(mat.shape[1] - mat.isna().sum(axis=1)) >= min_runs_model]
    mat_stat = calculate_mat_stats(mat)
    
    # filtering based on CV of RTs
    logging.info(f"Filtering out precursors with the highest CVs of {data_type.upper()}s across runs ({trim_pct}%).")
    high = mat_stat['CV'].quantile(1-trim_pct / 100)
    mat = mat.loc[mat_stat['CV'] < high, :]
    
    # get rt in library
    lib_df_uniq_prec_sub = lib_df_uniq_prec[lib_df_uniq_prec['Precursor.Id'].isin(mat.index)]
    if data_type == "rt":
        lib_df_uniq_prec_sub = lib_df_uniq_prec_sub.sort_values(by='RT', ascending = True)
    elif data_type == "im":
        lib_df_uniq_prec_sub = lib_df_uniq_prec_sub.sort_values(by='IM', ascending = True)
    lib_df_uniq_prec_sub.set_index('Precursor.Id', inplace=True, drop=False)
    
    # sort to the same order
    mat = mat.loc[lib_df_uniq_prec_sub['Precursor.Id'].values]
    n_segments = min(max_segments, max(1, int(2.0 * np.sqrt(len(mat) / min_points_per_bin))))
    logging.info(f"Using {len(mat)} precursors")
    
    # get RT data for each run
    data_dict = {}
    for run in mat.columns:
        data_dict[run] = list(zip(lib_df_uniq_prec_sub[f'{data_type.upper()}'], mat[run]))
        data_dict[run] = [
            (iRT, expRT)
            for iRT, expRT in zip(lib_df_uniq_prec_sub[f'{data_type.upper()}'], mat[run])
            if pd.notna(expRT)
        ]
        
    return data_dict, n_segments


def calculate_mat_stats(mat, axis=1):
    mat_stat = pd.DataFrame({'mean': mat.mean(axis=axis),
                             'median': mat.median(axis=axis),
                             'std': mat.std(axis=axis)})
    # calculate CVs
    mat_stat = mat_stat.assign(CV = mat_stat['std'] / mat_stat['mean'])
    
    return mat_stat
    

def modeling_and_prediction(data_dict, 
                            n_segments, 
                            mat, 
                            lib_df_uniq_prec, 
                            data_type, 
                            out_dir,
                            save_plot=True,
                            save_model=True):

    models = {}
    model_accs = []
    mat_pred = mat.copy()
    for run_name in data_dict.keys():
        # build the model for predicting RT from iRT
        logging.info(f"Building {data_type.upper()} prediction model for run {run_name}")
        model_data = data_dict[run_name]
        x, y = convert_to_arrays(model_data)
        model_coeff, model_points = refine_spline(x, y, n_segments)
        models[run_name] = {"coeff": model_coeff, "points": model_points}
        delta_percentile=0.8
        model_acc = check_model_accuracy(model_data, model_coeff, model_points, delta_percentile)
        logging.info(f"{data_type.upper()} prediction accuracy for {delta_percentile * 100}% precursors: {model_acc:.4g}")
        model_accs.append(model_acc)
        if save_plot:
            plot_out_dir = out_dir / f'{data_type}_prediction'
            spline_fitting_plot(model_data, model_coeff, 
                                model_points, model_acc, 
                                show_window=False, model=data_type.upper(), 
                                out_dir=plot_out_dir, run_name=run_name,
                                save_plot=save_plot)
            
        # predict RT for precursors with no RT        
        # Identify missing RTs
        missing_mask = mat[run_name].isna()
        missing_prec_ids = mat.index[missing_mask]
        
        # Get matching library values (dropping ones not found)
        lib_series = lib_df_uniq_prec.set_index('Precursor.Id')[data_type.upper()]
        lib_subset = lib_series.reindex(missing_prec_ids).dropna()
        
        # Vectorized prediction
        x_vals = lib_subset.values.astype(np.float32)
        predicted_rts = calc_spline_vectorized(model_coeff, model_points, x_vals)
        
        # Batch update
        mat_pred.loc[lib_subset.index, run_name] = predicted_rts
    
    if save_model:
        json_file = out_dir / f'{data_type}_models.json'
        with open(json_file, "w") as f:
            json.dump(models, f, indent=4, default=convert_ndarray)
        logging.info(f"{data_type.upper()} prediction models saved to {json_file}\n")
    
    return mat_pred, model_accs


def adjust_prediction_outliers(mat_pred, mat, n_iter, save_plot=True, out_dir=""):
    # calculate reference RT deviation range from actual matrix
    ## medians of each row
    mat_stat = calculate_mat_stats(mat)
    ## diffs with row medians
    mat_median_diff = mat.sub(mat_stat['median'].values, axis=0)
    ## min and max of diffs
    median_diff = mat_median_diff.values.flatten()
    diff_max = np.nanmax(median_diff)
    diff_min = np.nanmin(median_diff)
    logging.info(f"Min and max RT differences across runs before RT prediction: {diff_min:.4g}, {diff_max:.4g}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1.hist(mat_median_diff.stack(), bins=50, color='steelblue', edgecolor='black')
    ax1.set_ylabel("Frequency")
    ax1.set_title("Before prediction")
    
    logging.info("Adjusting RT prediction outliers\n")
    initial_n_iter = n_iter
    while n_iter > 0:
        # compute prediction deviation from row median
        mat_pred_stat = calculate_mat_stats(mat_pred)
        mat_pred_median_diff = mat_pred.sub(mat_pred_stat['median'].values, axis=0)
        pred_median_diff = mat_pred_median_diff.values.flatten()
        
        if n_iter == initial_n_iter:
            ax2.hist(mat_pred_median_diff.stack(), bins=50, color='steelblue', edgecolor='black')
            ax2.set_title("After prediction")
            
        if np.all(pred_median_diff <= diff_max) and np.all(pred_median_diff >= diff_min):
            break

        # clip values to allowed range
        mat_pred_median_diff_masked = mat_pred_median_diff.mask((mat_pred_median_diff > diff_max) | (mat_pred_median_diff < diff_min), 0.0)
        mat_pred = mat_pred_median_diff_masked.add(mat_pred_stat['median'].values, axis=0)
        
        n_iter -= 1
    
    mat_pred_stat = calculate_mat_stats(mat_pred)
    mat_pred_median_diff = mat_pred.sub(mat_pred_stat['median'].values, axis=0)
    ax3.hist(mat_pred_median_diff.stack(), bins=50, color='steelblue', edgecolor='black')
    ax3.set_title("After prediction and outlier adjustment")
    fig.supxlabel("RT differences across runs compared with median RT")
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(Path(out_dir) / "rt_diff_histogram.pdf")
        plt.close()
    
    return mat_pred


# Convert NumPy arrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
# 
def get_prec_prot_relation(lib_file, out_dir):
    prec_prot_df = pd.read_parquet(lib_file, columns=['Precursor.Id', 
                                                        'Protein.Ids', 
                                                        'Protein.Group',
                                                        'Protein.Names', 
                                                        'Genes'])
    prec_prot_df = prec_prot_df.drop_duplicates(ignore_index=True)
    prec_prot_df.to_csv(Path(out_dir) / "prec_prot_relation.csv", index=False)
    
    return prec_prot_df

# 
def get_prot_prec_pep_info(report_file):
    
    # report_file = r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\jumpdia_gui\jump_dia_report.parquet"
    df = pd.read_parquet(report_file, columns=['Protein.Group', 'Precursor.Id', 'Stripped.Sequence'])
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
    
    
def calculate_protein_coverage(prot_prec_pep_info, fasta_files): 
    prot_prec_pep_info.insert(1, 'Protein.Group.Rep', prot_prec_pep_info['Protein.Group'].str.split(pat=';', n=1, expand=True)[0])
    #
    prot_fas_info = extract_prot_info_from_FASTAs(fasta_files, prot_prec_pep_info['Protein.Group.Rep'].values.tolist())
    #prot_fas_info.drop(columns=['Protein.Name','Database'], inplace = True)
    prot_fas_info.rename(columns={"Protein.Id": "Protein.Group.Rep"}, inplace = True)
    prot_fas_info = prot_fas_info.drop_duplicates(subset=['Protein.Group.Rep'], ignore_index=True)
    #
    prot_prec_pep_info = prot_fas_info.merge(prot_prec_pep_info, how = 'right')
    prot_prec_pep_info.drop(columns=['Protein.Group.Rep'], inplace = True)
    #
    (prot_prec_pep_info['Peptide.Positions'], 
     prot_prec_pep_info['Coverage.Percentage']) = map_peptides(prot_prec_pep_info['Sequence'].tolist(), 
                                                               prot_prec_pep_info['Peptides'].tolist())
    
    return prot_prec_pep_info


def map_peptides(protein_seqs, peptide_seqs):
    results_positions = []
    results_coverage = []

    for protein, peptides_str in zip(protein_seqs, peptide_seqs):
        if type(protein) == str:
            peptides = peptides_str.split(', ')
            positions = []
            covered = set()
    
            for peptide in peptides:
                # Use while loop to find all occurrences
                start = 0
                while True:
                    idx = protein.find(peptide, start)
                    if idx == -1:
                        break
                    end = idx + len(peptide)
                    positions.append((idx + 1, end))  # 1-based position
                    covered.update(range(idx, end))
                    start = idx + 1
    
            coverage_rate = len(covered) / len(protein) if protein else 0
            results_positions.append(positions)
            results_coverage.append(np.round(coverage_rate*100, 2))
        else:
            results_positions.append([])
            results_coverage.append(0.0)

    return results_positions, results_coverage


def extract_from_single_fasta(fasta_file, protein_ids, extract_sequence=True):
    logging.info(f"Extracting protein information from FASTA file: {fasta_file}")
    
    prot_info = {}

    with open(fasta_file) as fasta:
        records = list(SeqIO.parse(fasta, 'fasta'))
        
        for record in records:
            header = record.description
            temp = header.split(" ", 1)
            prot_id = temp[0].split("|")[1]

            if prot_id in protein_ids:
                prot_data = {
                    'Accession.Number': temp[0],
                    'Protein.Description': temp[1] if len(temp) > 1 else ''
                }
                if extract_sequence:
                    prot_data['Sequence'] = str(record.seq)
                
                prot_info[prot_id] = prot_data

    return prot_info


def extract_prot_info_from_FASTAs(fasta_files, protein_ids, extract_sequence=True, out_file=0):
    # Prepare dictionary to collect all protein info
    collected_info = {}

    for protein_id in protein_ids:
        collected_info[protein_id] = {
            'Accession.Number': '',
            'Protein.Description': '',
            'Sequence': '' if extract_sequence else ''
        }

    if len(fasta_files) == 1:
        prot_info = extract_from_single_fasta(fasta_files[0], protein_ids, extract_sequence)
    else:
        prot_info = {}
        for fasta_file in fasta_files:
            temp_info = extract_from_single_fasta(fasta_file, protein_ids, extract_sequence)
            prot_info.update(temp_info)

    # Update collected_info with found proteins
    for prot_id, info in prot_info.items():
        collected_info[prot_id] = info

    # Build DataFrame
    prot_info_df = pd.DataFrame.from_dict(collected_info, orient='index').reset_index()
    prot_info_df.rename(columns={'index': 'Protein.Id'}, inplace=True)

    # Final DataFrame
    prot_info_final = prot_info_df

    if out_file != 0:
        prot_info_final.to_csv(out_file, index=False)

    return prot_info_final