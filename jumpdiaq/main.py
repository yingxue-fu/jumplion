# builtin
import sys
import platform  # os info
import psutil    # cpu and memory info
import gc
from pathlib import Path
import logging
# external
import pandas as pd
import matplotlib
matplotlib.use('Agg') # This line must come before importing pyplot
# local
from jumpdiaq.arguments import parse_arguments, setup_logger, close_logger
from jumpdiaq.utils import remove_extension, get_file_name
from jumpdiaq.summarization import (quantify_precursors, 
                                    quantify_proteins, 
                                    quality_hist_plot, 
                                    norm_dataframe, 
                                    filter_based_on_quality)


def main():
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    if len(sys.argv) < 2:
        print("Type jumpdiaq -h to show help message!")
        sys.exit(1)
    
    # read arguments
    args = parse_arguments(sys.argv[1:])  
    from jumpdiaq.library import Library
    from jumpdiaq.run import Run
    from jumpdiaq.id_results_processing import (process_id_results, 
                                             get_precursor_locations, 
                                             get_protein_info, 
                                             calculate_protein_coverage)

    #
    MSFiles = args.ms_files
    IdFile = args.id_report
    LibFile = args.spec_lib
    FastaFiles = args.fastas
    # 
    OutDir = args.out_dir
    LogLevel = args.log_level
    # 
    UseSavedRTs = args.use_saved_rts
    UseSavedXIC = args.use_saved_xic
    UseXICReport = args.use_xic_report
    #
    NormSpecies = args.norm_species
    NormMethod = "median" if args.norm_method == 0 else "mean"
    TrimPercentage = args.trim_percentage
    
    # System information
    system = platform.system()
    release = platform.release()
    architecture = platform.architecture()[0]
    cores = psutil.cpu_count(logical=False)   # Physical cores
    threads = psutil.cpu_count(logical=True)  # Logical threads
    mem_info = psutil.virtual_memory()
    mem_available = mem_info.available / (1024 ** 3)
    SaveMemory = True if mem_available < 64 else False
    
    logger = setup_logger(LogLevel, False, OutDir)
    logger.info("=====================================")
    logger.info("+         JUMPdiaq (v0.1.0)         +")
    logger.info("+         Author: Yingxue Fu        +")
    logger.info("=====================================\n")
    logger.info(f"System: {system} {release}, {architecture}")
    logger.info(f"CPU: {cores} Cores, {threads} Threads")
    logger.info(f"Memory: {mem_available:.2f} GB/{mem_info.total / (1024 ** 3):.2f} GB  (Available/Total)\n")
    
    report_file = Path(OutDir) / 'xic_quan_report.parquet'
    if UseXICReport:
        logging.info(f"Using existing XIC quantification report: {report_file}") 
    else:
        logging.info(f"{len(MSFiles)} runs will be analyzed")
        (is_tims, id_df, 
         rt_mat, im_mat, 
         lib_df_uniq_prec, 
         cycle_time) = process_id_results(IdFile, LibFile, MSFiles, OutDir)
        
        lib = Library()
        if not lib.load(Path(OutDir) / 'splib_quan.parquet'):
            raise Exception("Cannot load spectral library")
        lib.initialize()
        
        if not UseSavedRTs:
            logging.info("Predicting precursor RTs (and IMs)") 
            prec_locations = get_precursor_locations(is_tims, id_df,
                                                     rt_mat, im_mat,
                                                     lib_df_uniq_prec,
                                                     cycle_time, OutDir)
        else:
            prec_locations_file = Path(OutDir) / "precursor_locations.csv"
            logging.info(f"Using existing precursor RTs (and IMs): {prec_locations_file}\n") 
            prec_locations = pd.read_csv(prec_locations_file)
            prec_locations.set_index('Precursor.Id', inplace=True)
            
        for i, file in enumerate(MSFiles):
            # i = 0
            # file = MSFiles[i]
            logging.info(f"Analyzing run {i+1} of {len(MSFiles)}")
            if UseSavedXIC:
                run_name = remove_extension(get_file_name(file))
                xic_file = Path(OutDir) / "xic_files" / f"{run_name}.xic.parquet"
                if xic_file.exists():
                    logging.info(f"Using existing xic file {xic_file}")
                    logging.info("Skipping analysis\n")
                    continue
            
            run = Run(i)
            if not run.load(file):
                logging.info(f"ERROR: cannot load the file {file}, skipping")
            run.load_library(lib)
            run.load_prec_locations(prec_locations)
            run.process()
            run.free()
            del run
            if SaveMemory:
                gc.collect()
                
        # 
        prec_prot_info = get_protein_info(IdFile, OutDir)
        lib.info.load(lib, MSFiles, OutDir)
        lib.info.quantify(MSFiles, OutDir)
        lib.report(report_file, MSFiles, prec_prot_info)
    
    # OutDir = r"C:\Users\yfu\Downloads\miss_dia_test_data\chengrp_test\chengrp_20250701"
    # Precursor quantification
    prec_quant_table, prec_quality_table = quantify_precursors(
                                                report_file, 
                                                xic_quality_cutoff=0.0, 
                                                xic_quant_cutoff=0.0, 
                                                quant_source='ms1_ms2',
                                                out_dir=OutDir)

    # Normalization
    # NormSpecies='HUMAN'
    # NormMethod='median'
    # TrimPercentage=5
    prec_quant_table_norm, norm_factors = norm_dataframe(
                                                  prec_quant_table, 
                                                  data_type='Precursors',
                                                  tag_column='Protein.Names', 
                                                  value_columns=prec_quant_table.columns[4:-3], 
                                                  norm_species=NormSpecies, 
                                                  norm_method=NormMethod, 
                                                  trim_percentage=TrimPercentage)
    
    prec_quant_table.write_csv(Path(OutDir) / "Precursor_Quan_table.csv")
    prec_quant_table_norm.write_csv(Path(OutDir) / "Precursor_Quan_table_Norm.csv")
    
    prot_quant_table, prot_quality_table = quantify_proteins(
                                                    prec_quant_table_norm, 
                                                    prec_quality_table, 
                                                    max_n_precs=15)
                           
    # filter based on quality
    quality_hist_plot(prot_quality_table.to_pandas(), 
                      prot_quality_table.columns[3:-1], 
                      Path(OutDir))
    QualityCutoffs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    prot_quant_table_filtered = filter_based_on_quality(prot_quant_table.to_pandas(), 
                                                        prot_quality_table.to_pandas(),
                                                        tag_column='Protein.Group',
                                                        value_columns=prot_quant_table.columns[3:-1],
                                                        quality_cutoffs=QualityCutoffs)
    if len(FastaFiles):
        prot_prec_pep_info = calculate_protein_coverage(Path(OutDir) / 'xic_quan_report.parquet', 
                                                        FastaFiles, 
                                                        str(Path(OutDir)))
    else:
        logging.warning("WARNING: no Fasta files are provided, protein coverage will not be calculated.")
    prot_quant_table = prot_prec_pep_info.merge(prot_quant_table.to_pandas(), how='right')
    prot_quant_table.to_csv(Path(OutDir) / "Protein.Group_Quan_QT0.0.csv", index=False)
    
    for i, cutoff in enumerate(QualityCutoffs):
        prot_quant_table_f = prot_quant_table_filtered[i]
        prot_quant_table_f = prot_prec_pep_info.merge(prot_quant_table_f, how='right')
        prot_quant_table_f.to_csv(Path(OutDir) / f"Protein.Group_Quan_QT{cutoff}.csv", index=False)
        
    prot_quality_table = prot_prec_pep_info.merge(prot_quality_table.to_pandas(), how='right')
    prot_quality_table.to_csv(Path(OutDir) / "Protein.Group_Quan_Quality.csv", index=False)
    
    close_logger(logger)
