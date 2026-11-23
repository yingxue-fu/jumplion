# builtin
import os
import gc
import sys
import shutil
import psutil    # cpu and memory info
import platform  # os info
import logging
import warnings
from pathlib import Path
from importlib.metadata import version
warnings.filterwarnings(
    "ignore",
    message="You may be importing Biopython from inside the source tree.*",
)
# external
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
# local
from jumplion.arguments import parse_arguments, setup_logger, close_logger
from jumplion.utils import get_extension, remove_extension, get_file_name


def main():
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    if len(sys.argv) < 2:
        print("Type 'jumplion -h' to show the help message.")
        sys.exit(1)
    
    # os info
    system = platform.system()
    release = platform.release().split(sep='-')[0]
    architecture = platform.architecture()[0]
    # cpu and memory info
    cores = psutil.cpu_count(logical=False)   # Physical cores
    threads = psutil.cpu_count(logical=True)  # Logical threads
    mem_info = psutil.virtual_memory()
    mem_avail = mem_info.available / (1024 ** 3)
    mem_total = mem_info.total / (1024 ** 3)

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
    logger.info("=====================================")
    logger.info(f"     JUMPlion (v{version('jumplion')})")
    logger.info(" ")
    logger.info(f"   OS: {system} {release}, {architecture} ")
    logger.info(f"  CPU: {cores} Cores, {threads} Threads")
    logger.info(f"  RAM: {mem_avail:.0f} GB/{mem_total:.0f} GB (Avail/Total) ")
    logger.info("=====================================\n")
    logger.info(f"Command: {cmd}\n")
    
    # Copy the parameter file
    if os.path.isfile(sys.argv[1]):
        shutil.copy(sys.argv[1], Path(OutDir) /'jumplion.params')
    
    from jumplion.library import Library
    from jumplion.run import Run
    from jumplion.id_results_processing import process_id_results
    from jumplion.summarization import (quantify_precursors, 
                                        quantify_proteins, 
                                        norm_dataframe, 
                                        filter_one_hit_wonder)
    from jumplion.intensity_mapping import run_intensity_mapping, define_status

    # Input files
    MSFiles = args.ms_files
    IdFile = args.prec_ids
    LibFile = args.spec_lib
    
    # Parameters
    ## Use existing results
    SkipIDProcessing = args.skip_id_processing
    UseSavedXIC = args.use_saved_xic
    UseXICQuant = args.use_xic_quant
    ##  
    UseRawMS1 = args.use_raw_ms1
    UseRawMS2 = args.use_raw_ms2
    ## Protein quantification
    PrecQuantLevel = args.prec_quant_level
    MinPrecCount = args.min_prec_count
    MinPepCount = args.min_pep_count
    ## Protein normalization
    Normalization = args.normalization
    NormSpecies = args.norm_species
    NormMethod = args.norm_method
    TrimPercentage = args.trim_percentage
    
    ## ID results processing
    logging.info(f"{len(MSFiles)} runs will be analyzed")
    MSFileType = get_extension(MSFiles[0])
    if MSFileType == 'd':
        vendor = 'Bruker'
    elif MSFileType == 'raw':
        vendor = 'Thermo Fisher'
    elif MSFileType == 'mzML':
        vendor = 'open format'
    else:
        logging.info(f"MS file format .{MSFileType} not supported!")
        sys.exit(1)
    logging.info(f"MS file format recognized as {vendor} .{MSFileType}")
    if SkipIDProcessing:
        prec_rt_file = out_int_dir / "prec_rt_info.tsv"
        missing_status_file = out_int_dir / "missing_status.tsv"
        
        logging.info(f"ID results processing skipped. Using existing precursor RTs (and IMs): {prec_rt_file}\n") 
        prec_rts = pd.read_csv(prec_rt_file, sep='\t', index_col=0)
        missing_status = pd.read_csv(missing_status_file, sep='\t', index_col=0)
    else:
        prec_rts, missing_status = process_id_results(IdFile, LibFile, MSFiles, MSFileType, OutDir)
    
    # XIC quantification
    xic_quant_file = out_pub_dir / 'xic_quantification.parquet'
    if UseXICQuant:
        logging.info(f"Using existing XIC quantification result: {xic_quant_file}") 
    else:
        ## Load library
        lib = Library()
        if not lib.load(out_int_dir / 'speclib_filtered.parquet'):
            raise Exception("Cannot load spectral library")
        lib.initialize()
        
        ## Initialize lib.info for in-memory accumulation
        lib.info.n_runs = len(MSFiles)
        lib.info.lib = lib
        lib.info.clear()
        saved_xic_runs = []  # (run_index, xic_file) for skipped runs

        ## XIC extraction in each run
        for i, file in enumerate(MSFiles):
            logging.info(f"Analyzing run {i+1} of {len(MSFiles)}")
            if UseSavedXIC:
                run_name = remove_extension(get_file_name(file))
                xic_file = out_int_dir / "xic_files" / f"{run_name}.xic.parquet"
                if xic_file.exists():
                    logging.info(f"Using existing xic file {xic_file}")
                    logging.info("Skipping analysis\n")
                    saved_xic_runs.append((i, xic_file))
                    continue

            run = Run(i)
            if not run.load(file):
                logging.info(f"ERROR: cannot load the file {file}, skipping")
            run.load_library(lib)
            run.load_prec_rts(prec_rts, missing_status)
            run.process(info=lib.info)
            run.free()
            del run
            # determine if performing garbage collection
            if psutil.virtual_memory().available / (1024 ** 3) < 32:
                logging.info("Available RAM less than 32 GB, performing garbage collection")
                gc.collect()

        ## Load parquet only for runs that were skipped via --use-saved-xic
        if saved_xic_runs:
            lib_n = len(lib.entries)
            for run_index, xic_file in saved_xic_runs:
                logging.info(f"Loading XIC data from {xic_file}")
                xic_df = pl.read_parquet(xic_file)
                lib.info._load_xic_dataframe(xic_df, run_index, lib_n)

        lib.info.n_entries = len(lib.info.map)
        lib.info.index(MSFiles)
        lib.info.quantify(MSFiles, str(out_int_dir))
        lib.report(xic_quant_file, MSFiles)
    
    # Precursor quantification
    xic_quant_file = Path(xic_quant_file)
    if not xic_quant_file.exists():
        raise FileNotFoundError(f"XIC quantification file not found: {xic_quant_file}")
    xic_quant_df = pd.read_parquet(xic_quant_file)
    out_prec_dir = out_int_dir / 'prec_quant'
    jumplion_prec_matrix = quantify_precursors(pl.from_pandas(xic_quant_df), 
                                               use_raw_ms1 = UseRawMS1,
                                               use_raw_ms2 = UseRawMS2,
                                               prec_quant_level = PrecQuantLevel, 
                                               out_dir = str(out_prec_dir))
    jumplion_prec_matrix = jumplion_prec_matrix.to_pandas()
    run_cols = jumplion_prec_matrix.columns[1:]
    jumplion_long = jumplion_prec_matrix.melt(
        id_vars=["Precursor.Id"],        # Columns to keep as identifiers
        value_vars=run_cols,             # Columns to unpivot (optional, defaults to all others)
        var_name='Run',                  # Rename the unpivoted column headers
        value_name="JUMPlion.Raw.Quantity"
    )
    
    prec_prot_df = pd.read_csv(out_int_dir / "prec_prot_relation.tsv", sep='\t')
    jumplion_prec_matrix = prec_prot_df.merge(jumplion_prec_matrix, on='Precursor.Id', how='inner')
    jumplion_prec_matrix.to_csv(out_pub_dir / "prec_matrix_jumplion.tsv", sep='\t', index=False)
    

    # 
    result = run_intensity_mapping(
        quantity_diann=out_pub_dir / "prec_matrix_diann.tsv",
        quantity_jumplion=out_pub_dir / "prec_matrix_jumplion.tsv",
        id_col="Precursor.Id",
        min_overlap=5,
        min_overlap_fraction=0.2,
        min_r2=0.00,
        max_mae=0.50,
        min_spearman=-1,
        min_slope=0.50,
        max_slope=1.50,
        max_extrapolation_log2=2,
        min_missing_values_for_recover=1,
        min_missing_fraction_for_recover=0.0,
        output_dir=out_prec_dir,
    )
    
    final_prec_matrix = result["quantity_recoverd"]
    #recover_summary = result["recover_summary"]
    
    final_long = final_prec_matrix.melt(
        id_vars=["Precursor.Id"],        # Columns to keep as identifiers
        value_vars=run_cols,             # Columns to unpivot (optional, defaults to all others)
        var_name='Run',                  # Rename the unpivoted column headers
        value_name='Final.Quantity'      # Rename the unpivoted values
    )
    final_prec_matrix = prec_prot_df.merge(final_prec_matrix, on='Precursor.Id', how='inner')
    final_prec_matrix.to_csv(out_pub_dir / "prec_matrix_diann_jumplion.tsv", sep='\t', index=False)
    
    # Precursor normalization
    # Normalization=True
    # NormSpecies='HUMAN'
    # NormMethod='median'
    # TrimPercentage=5
    diann_prec_matrix = pd.read_csv(out_pub_dir / "prec_matrix_diann.tsv", sep='\t')
    diann_long = diann_prec_matrix.melt(
        id_vars=["Precursor.Id"],        # Columns to keep as identifiers
        value_vars=run_cols,             # Columns to unpivot (optional, defaults to all others)
        var_name='Run',                  # Rename the unpivoted column headers
        value_name='DIANN.Quantity'      # Rename the unpivoted values
    )
    
    # diann_prec_matrix_norm = norm_dataframe(df=pl.from_pandas(diann_prec_matrix), 
    #                                           data_type='Precursor',
    #                                           tag_column='Protein.Names', 
    #                                           value_columns=run_cols.values.tolist(), 
    #                                           normalization=Normalization,
    #                                           norm_species=NormSpecies, 
    #                                           norm_method=NormMethod, 
    #                                           trim_percentage=TrimPercentage)
    # diann_prec_matrix_norm = diann_prec_matrix_norm.to_pandas()
    diann_prec_matrix_norm = pd.read_csv(out_pub_dir / "prec_matrix_diann_norm.tsv", sep='\t')
    diann_norm_long = diann_prec_matrix_norm.melt(
        id_vars=["Precursor.Id"],             # Columns to keep as identifiers
        value_vars=run_cols,                  # Columns to unpivot (optional, defaults to all others)
        var_name='Run',                       # Rename the unpivoted column headers
        value_name='DIANN.Quantity.Norm'      # Rename the unpivoted values
    )
    #diann_prec_matrix_norm.to_csv(out_pub_dir / "prec_matrix_diann_norm.tsv", sep='\t', index=False)
    
    
    final_prec_matrix_norm = norm_dataframe(df=pl.from_pandas(final_prec_matrix), 
                                            data_type='Precursor',
                                            tag_column='Protein.Names', 
                                            value_columns=run_cols.values.tolist(), 
                                            normalization=Normalization,
                                            norm_species=NormSpecies, 
                                            norm_method=NormMethod, 
                                            trim_percentage=TrimPercentage)
    final_prec_matrix_norm = final_prec_matrix_norm.to_pandas()
    final_norm_long = final_prec_matrix_norm.melt(
        id_vars=["Precursor.Id"],        # Columns to keep as identifiers
        value_vars=run_cols,             # Columns to unpivot (optional, defaults to all others)
        var_name='Run',                  # Rename the unpivoted column headers
        value_name='Final.Quantity.Norm'      # Rename the unpivoted values
    )
    final_prec_matrix_norm.to_csv(out_pub_dir / "prec_matrix_diann_jumplion_norm.tsv", sep='\t', index=False)
    
    #
    adding_cols = ['JUMPlion.Raw.Quantity', 'DIANN.Quantity', 'DIANN.Quantity.Norm', 'Final.Quantity', 'Final.Quantity.Norm']
    xic_quant_df = xic_quant_df.drop(columns=[col for col in adding_cols if col in xic_quant_df.columns])
    xic_quant_df = xic_quant_df.merge(jumplion_long, on=['Precursor.Id', 'Run'], how='left')
    xic_quant_df = xic_quant_df.merge(diann_long, on=['Precursor.Id', 'Run'], how='left')
    xic_quant_df = xic_quant_df.merge(diann_norm_long, on=['Precursor.Id', 'Run'], how='left')
    xic_quant_df = xic_quant_df.merge(final_long, on=['Precursor.Id', 'Run'], how='left')    
    xic_quant_df = xic_quant_df.merge(final_norm_long, on=['Precursor.Id', 'Run'], how='left')    
    xic_quant_df = define_status(xic_quant_df)
    xic_quant_df.to_parquet(xic_quant_file)
    
    # Protein quantification
    # diann_prec_quant = pl.read_csv(out_pub_dir / 'prec_matrix_diann.tsv', separator='\t')
    # diann_prot_quant = quantify_proteins(diann_prec_quant)
    # diann_prot_quant.write_csv(out_pub_dir / "prot_matrix_diann_requant.tsv", separator='\t')
    
    prot_quantity_table = quantify_proteins(pl.from_pandas(final_prec_matrix_norm), top_n=None)
    prot_quantity_table.write_csv(out_pub_dir / "prot_matrix_jumplion.tsv", separator='\t')
    
    out_prot_dir = out_int_dir / 'prot_quant'
    result = run_intensity_mapping(
        quantity_diann=out_pub_dir / "prot_matrix_diann.tsv",
        quantity_jumplion=out_pub_dir / "prot_matrix_jumplion.tsv",
        id_col="Protein.Group",
        min_overlap=5,
        min_overlap_fraction=0.2,
        min_r2=0.00,
        max_mae=0.50,
        min_spearman=-1,
        min_slope=0.50,
        max_slope=1.50,
        max_extrapolation_log2=2,
        min_missing_values_for_recover=1,
        min_missing_fraction_for_recover=0.0,
        output_dir=out_prot_dir,
    )
    
    final_prot_matrix = result["quantity_recoverd"]
    prot_df = prot_quantity_table.to_pandas().iloc[:, 0:7]
    final_prot_matrix = prot_df.merge(final_prot_matrix, on='Protein.Group', how='inner')
    final_prot_matrix.to_csv(out_pub_dir / "prot_matrix_diann_jumplion.tsv", sep='\t', index=False)
    
    # Filter one-hit wonder (protein with one precursor)
    prot_quantity_table_f = filter_one_hit_wonder(pl.from_pandas(final_prot_matrix), 
                                                  min_prec_count=MinPrecCount,
                                                  min_pep_count=MinPepCount)
    prot_quantity_table_f.write_csv(out_pub_dir / "prot_matrix_diann_jumplion_rm1hw.tsv", separator='\t') 
    
    
    logging.info("Finished!")
    close_logger(logger)