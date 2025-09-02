# builtin
import os
import sys
import shutil
import platform  # os info
import psutil    # cpu and memory info
import gc
from pathlib import Path
import logging
# external
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # This line must come before importing pyplot
# local
import jumpdiaq
from jumpdiaq.arguments import parse_arguments, setup_logger, close_logger
from jumpdiaq.utils import remove_extension, get_file_name


def main():
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    if len(sys.argv) < 2:
        print("Use jumpdiaq -h to show the help message")
        sys.exit(1)
    
    # read arguments
    args, cmd = parse_arguments(sys.argv[1:])
    if args is None:
        sys.exit(1)
    if args.gen_params:
        args.generate_file(args.gen_params)
        sys.exit(1)
    
    # System information
    system = platform.system()
    release = platform.release()
    architecture = platform.architecture()[0]
    cores = psutil.cpu_count(logical=False)   # Physical cores
    threads = psutil.cpu_count(logical=True)  # Logical threads
    mem_info = psutil.virtual_memory()
    mem_available = mem_info.available / (1024 ** 3)
    SaveMemory = True if mem_available < 64 else False
    
    OutDir = args.out_dir
    logger = setup_logger(False, OutDir)
    logger.info("====================================================")
    logger.info(f" JUMPdiaq (v{jumpdiaq.__version__}) ")
    logger.info(" ")
    logger.info(f" System: {system} {release}, {architecture} ")
    logger.info(f" CPU: {cores} Cores, {threads} Threads ")
    logger.info(f" Memory: {mem_available:.2f} GB/{mem_info.total / (1024 ** 3):.2f} GB  (Available/Total) ")
    logger.info("====================================================\n")
    
    logger.info(f"Command: {cmd}\n")
    if os.path.isfile(sys.argv[1]):
        # copy parameter file
        shutil.copy(sys.argv[1], Path(OutDir) /'jumpdiaq.params')
    
    from jumpdiaq.library import Library
    from jumpdiaq.run import Run
    from jumpdiaq.id_results_processing import (process_id_results, 
                                                get_prec_prot_relation, 
                                                calculate_protein_coverage)
    from jumpdiaq.summarization import (quantify_precursors, 
                                        quantify_proteins, 
                                        quality_hist_plot, 
                                        norm_dataframe, 
                                        filter_based_on_quality)

    # Input files
    MSFiles = args.ms_files
    IdFile = args.id_report
    LibFile = args.spec_lib
    FastaFiles = args.fastas
    
    if not LibFile:
        logging.warning("WARNING: no spectral library provided: nothing to do\n")
    # 
    SkipIDprocess = args.skip_id_process
    UseSavedXIC = args.use_saved_xic
    UseXICReport = args.use_xic_report
    #
    XicQualityCutoff = args.xic_quality_cutoff
    FillMissingXic = args.fill_missing_xic
    NormSpecies = args.norm_species
    NormMethod = args.norm_method
    TrimPercentage = args.trim_percentage
    #
    PrecQuantLevel = args.prec_quant_level
    MinPrecQuality = args.min_prec_quality
    MS2PriorityQuality = args.ms2_priority_quality
    ProtQualityCutoffs = args.prot_quality_cutoffs
    #MaxNPrecs = args.max_n_precs
    #
    out_int_dir = Path(OutDir) / 'intermediate'
    out_pub_dir = Path(OutDir) / 'publication'
    report_file = out_pub_dir / 'xic_quantification.parquet'
    if UseXICReport:
        logging.info(f"Using existing XIC quantification report: {report_file}") 
    else:
        logging.info(f"{len(MSFiles)} runs will be analyzed")
        if SkipIDprocess:
            prec_locations_file = out_int_dir / "prec_locations.csv"
            logging.info(f"ID results processing skipped. Using existing precursor RTs (and IMs): {prec_locations_file}\n") 
            prec_locations = pd.read_csv(prec_locations_file)
            prec_locations.set_index('Precursor.Id', inplace=True)
        else:
            prec_locations = process_id_results(IdFile, LibFile, MSFiles, OutDir)
        
        # load library
        lib = Library()
        if not lib.load(out_int_dir / 'speclib_filtered.parquet'):
            raise Exception("Cannot load spectral library")
        lib.initialize()
        
        # Analyze each run
        for i, file in enumerate(MSFiles):
            logging.info(f"Analyzing run {i+1} of {len(MSFiles)}")
            if UseSavedXIC:
                run_name = remove_extension(get_file_name(file))
                xic_file = out_int_dir / "xic_files" / f"{run_name}.xic.parquet"
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

        # Assambling xic data and genereate xic quant report
        prec_prot_relation = get_prec_prot_relation(LibFile, str(out_int_dir))
        lib.info.load(lib, MSFiles, str(out_int_dir))
        lib.info.quantify(MSFiles, str(out_int_dir))
        lib.report(report_file, MSFiles, prec_prot_relation)
    
    # Precursor quantification
    out_prec_dir = out_int_dir / 'prec_quant'
    prec_quant_table, prec_quality_table = quantify_precursors(
                                                report_file, 
                                                xic_quality_cutoff=XicQualityCutoff, 
                                                xic_quant_cutoff=0.0, 
                                                fill_missing_xic=FillMissingXic,
                                                quant_source=PrecQuantLevel,
                                                ms2_priority_quality=MS2PriorityQuality,
                                                out_dir=str(out_prec_dir))
    prec_quant_table.write_parquet(out_pub_dir / "precursor_quantity_summ_table.parquet")
    prec_quality_table.write_parquet(out_pub_dir / "precursor_quality_summ_table.parquet")
    
    # Normalization
    # NormSpecies='HUMAN'
    # NormMethod='median'
    # TrimPercentage=5
    prec_quant_table_norm, norm_factors = norm_dataframe(
                                                  prec_quant_table, 
                                                  data_type='precursor',
                                                  tag_column='Protein.Names', 
                                                  value_columns=prec_quant_table.columns[4:-3], 
                                                  norm_species=NormSpecies, 
                                                  norm_method=NormMethod, 
                                                  trim_percentage=TrimPercentage)
    prec_quant_table_norm.write_parquet(out_pub_dir / "precursor_quantity_summ_table_norm.parquet")
    
    # 
    prot_quant_table, prot_quality_table = quantify_proteins(
                                               prec_quant_table_norm, 
                                               prec_quality_table, 
                                               min_prec_quality=MinPrecQuality)
    prot_quant_table.write_csv(out_pub_dir / "Protein.Group_Quant_noFiltering.csv")
    prot_quality_table.write_csv(out_pub_dir / "Protein.Group_Quant_Quality.csv")
                           
    # filter based on quality
    value_columns = prot_quality_table.columns[3:-1]
    if ProtQualityCutoffs < 1:
        quantiles = np.array([np.nan])
        QualityCutoffs = [ProtQualityCutoffs]
    else:
        quantiles = np.round(np.arange(0.0, 1.0, 1 / ProtQualityCutoffs), 2)
        QualityCutoffs = np.nanquantile(prot_quality_table.select(value_columns).to_numpy().flatten(), quantiles)
        QualityCutoffs = np.round(QualityCutoffs, 2)
    
    if args.save_supplements:
        quality_hist_plot(prot_quality_table.to_pandas(), 
                          value_columns, QualityCutoffs, out_pub_dir,
                          "Protein.Group_Quant_Quality_distribution.pdf")
    logging.info(f"Filtering based on protein quality cutoffs: {QualityCutoffs}")
    prot_quant_table_filtered = filter_based_on_quality(prot_quant_table.to_pandas(), 
                                                        prot_quality_table.to_pandas(),
                                                        tag_column='Protein.Group',
                                                        value_columns=prot_quant_table.columns[3:-1],
                                                        quality_cutoffs=QualityCutoffs)
    for i, (qt, cutoff) in enumerate(zip(quantiles, QualityCutoffs)):
        prot_quant_table_f = prot_quant_table_filtered[i]
        prot_quant_table_f.to_csv(out_pub_dir / f"Protein.Group_Quant_quantile{qt}_quality{cutoff}.csv", index=False)
    
    if len(FastaFiles):
        prot_prec_pep_info = calculate_protein_coverage(out_pub_dir / 'xic_quantification.parquet', 
                                                        FastaFiles, 
                                                        str(out_int_dir))
        prot_prec_pep_info.to_csv(out_int_dir / "prot_prec_pep_info.csv", index=False)
    else:
        logging.warning("WARNING: no Fasta files are provided, protein coverage will not be calculated.")
    
    logging.info("Finished!")
    close_logger(logger)
