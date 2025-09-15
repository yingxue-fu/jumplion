# builtin
import os
import gc
import sys
import shutil
import psutil    # cpu and memory info
import platform  # os info
import logging
from pathlib import Path
# external
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
# local
import jumpdiaq
from jumpdiaq.arguments import parse_arguments, setup_logger, close_logger
from jumpdiaq.utils import remove_extension, get_file_name


def main():
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    if len(sys.argv) < 2:
        print("Type 'jumpdiaq -h' to show the help message.")
        sys.exit(1)
    
    # os info
    system = platform.system()
    release = platform.release()
    architecture = platform.architecture()[0]
    # cpu and memory info
    cores = psutil.cpu_count(logical=False)   # Physical cores
    threads = psutil.cpu_count(logical=True)  # Logical threads
    mem_info = psutil.virtual_memory()
    mem_available = mem_info.available / (1024 ** 3)
    # save memory mode
    SaveMemory = True if mem_available < 64 else False
    
    # Read arguments
    args, cmd = parse_arguments(sys.argv[1:])
    if args is None:
        sys.exit(1)
    if args.gen_params:
        args.generate_file(args.gen_params)
        sys.exit(1)
    
    OutDir = args.out_dir
    out_int_dir = Path(OutDir) / 'intermediate'
    out_pub_dir = Path(OutDir) / 'publication'
    
    logger = setup_logger(False, OutDir)
    logger.info("====================================================")
    logger.info(f" JUMPdiaq (v{jumpdiaq.__version__}) ")
    logger.info(" ")
    logger.info(f" System: {system} {release}, {architecture} ")
    logger.info(f" CPU: {cores} Cores, {threads} Threads")
    logger.info(f" Memory: {mem_available:.2f} GB/{mem_info.total / (1024 ** 3):.2f} GB  (Available/Total) ")
    logger.info("====================================================\n")
    logger.info(f"Command: {cmd}\n")
    
    # Copy the parameter file
    if os.path.isfile(sys.argv[1]):
        shutil.copy(sys.argv[1], Path(OutDir) /'jumpdiaq.params')
    
    from jumpdiaq.library import Library
    from jumpdiaq.run import Run
    from jumpdiaq.id_results_processing import (process_id_results, 
                                                get_prec_prot_relation, 
                                                get_prot_prec_pep_info,
                                                calculate_protein_coverage)
    from jumpdiaq.summarization import (quantify_precursors, 
                                        quantify_proteins, 
                                        norm_dataframe)

    # Input files
    MSFiles = args.ms_files
    IdFile = args.id_report
    LibFile = args.spec_lib
    
    # Parameters
    ## Use existing results
    SkipIDprocess = args.skip_id_process
    UseSavedXIC = args.use_saved_xic
    UseXICReport = args.use_xic_report
    ## Precursor quantification
    UseRawMS1XIC = args.use_raw_ms1xic
    UseRawMS2XIC = args.use_raw_ms2xic
    XicQualityCutoff = args.xic_quality_cutoff
    FillMissingXic = args.fill_missing_xic
    NormSpecies = args.norm_species
    NormMethod = args.norm_method
    TrimPercentage = args.trim_percentage
    ## Protein quantification
    PrecQuantLevel = args.prec_quant_level
    MinPrecQuality = args.min_prec_quality
    MinPepNum = args.min_pep_num
    
    # XIC quantification
    report_file = out_pub_dir / 'xic_quantification.parquet'
    if UseXICReport:
        logging.info(f"Using existing XIC quantification report: {report_file}") 
    else:
        logging.info(f"{len(MSFiles)} runs will be analyzed")
        
        ## Skip ID results processing
        if SkipIDprocess:
            prec_rt_file = out_int_dir / "prec_rt_info.csv"
            logging.info(f"ID results processing skipped. Using existing precursor RTs (and IMs): {prec_rt_file}\n") 
            prec_rts = pd.read_csv(prec_rt_file)
            prec_rts.set_index('Precursor.Id', inplace=True)
        else:
            prec_rts = process_id_results(IdFile, LibFile, MSFiles, OutDir)
        
        ## Load library
        lib = Library()
        if not lib.load(out_int_dir / 'speclib_filtered.parquet'):
            raise Exception("Cannot load spectral library")
        lib.initialize()
        
        ## XIC extraction in each run
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
            run.load_prec_rts(prec_rts)
            run.process()
            run.free()
            del run
            if SaveMemory:
                gc.collect()

        ## XIC quantification across runs
        prec_prot_relation = get_prec_prot_relation(LibFile, str(out_int_dir))
        lib.info.load(lib, MSFiles, str(out_int_dir))
        lib.info.quantify(MSFiles, str(out_int_dir))
        lib.report(report_file, MSFiles, prec_prot_relation)
    
    # Precursor quantification
    out_prec_dir = out_int_dir / 'prec_quant'
    prec_quantity_table, prec_quality_table = quantify_precursors(report_file, 
                                                  use_raw_ms1xic=UseRawMS1XIC,
                                                  use_raw_ms2xic=UseRawMS2XIC,
                                                  xic_quality_cutoff=XicQualityCutoff,
                                                  fill_missing_xic=FillMissingXic,
                                                  prec_quant_level=PrecQuantLevel,
                                                  out_dir=str(out_prec_dir))
    prec_quantity_table.write_parquet(out_pub_dir / "precursor_quantity_summ_table.parquet")
    prec_quality_table.write_parquet(out_pub_dir / "precursor_quality_summ_table.parquet")
    
    # Precursor normalization
    # NormSpecies='HUMAN'
    # NormMethod='median'
    # TrimPercentage=5
    prec_quantity_table_norm, norm_factors = norm_dataframe(
                                                 prec_quantity_table, 
                                                 data_type='Precursor',
                                                 tag_column='Protein.Names', 
                                                 value_columns=prec_quantity_table.columns[4:-2], 
                                                 norm_species=NormSpecies, 
                                                 norm_method=NormMethod, 
                                                 trim_percentage=TrimPercentage)
    prec_quantity_table_norm.write_parquet(out_pub_dir / "precursor_quantity_summ_table_norm.parquet")
    
    # Protein quantification
    # prec_quantity_table_norm = pl.read_parquet(r"C:\Users\yfu\Downloads\miss_dia_test_data\timstof_pro2\test_20250908_xicHybrid_precMS2raw_cmd\publication\precursor_quantity_summ_table_norm.parquet")
    # prec_quality_table = pl.read_parquet(r"C:\Users\yfu\Downloads\miss_dia_test_data\timstof_pro2\test_20250908_xicHybrid_precMS2raw_cmd\publication\precursor_quality_summ_table.parquet")
    prot_quantity_table, prot_quality_table = quantify_proteins(prec_quantity_table_norm, 
                                                                prec_quality_table,
                                                                min_prec_quality=MinPrecQuality)
    # out_pub_dir = Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\timstof_pro2\test_20250908_xicHybrid_precMS2raw_cmd\publication")
    prot_quantity_table.write_csv(out_pub_dir / "protein.group_quantity_matrix_all.csv")
    prot_quality_table.write_csv(out_pub_dir / "protein.group_quality_matrix.csv")
    
    # Calculate the number of peptides of a protein
    prot_prec_pep_info = get_prot_prec_pep_info(out_pub_dir / 'xic_quantification.parquet')
    # Calculate protein coverage
    FastaFiles = args.fastas
    if len(FastaFiles):
        logging.info("Calculating protein coverage based on provided fasta files")
        prot_prec_pep_info = calculate_protein_coverage(prot_prec_pep_info, FastaFiles)
    else:
        logging.warning("WARNING: no Fasta files are provided, protein coverage will not be calculated.")
    prot_prec_pep_info.to_csv(out_int_dir / "prot_prec_pep_info.csv", index=False)
    
    # Filter based on the number of peptides
    accepted_pgs = prot_prec_pep_info['Protein.Group'][prot_prec_pep_info['n.Peptides'] >= MinPepNum]
    prot_quantity_table_f = prot_quantity_table.filter(pl.col('Protein.Group').is_in(accepted_pgs))
    prot_quantity_table_f.write_csv(out_pub_dir / "protein.group_quantity_matrix.csv")
    
    logging.info("Finished!")
    close_logger(logger)