# builtin
import os
import gc
import sys
import shutil
import psutil    # cpu and memory info
import platform  # os info
import logging
from pathlib import Path
from importlib.metadata import version
# external
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
# local
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
    logger.info(f"     JUMPdiaq (v{version('jumpdiaq')})")
    logger.info(" ")
    logger.info(f"   OS: {system} {release}, {architecture} ")
    logger.info(f"  CPU: {cores} Cores, {threads} Threads")
    logger.info(f"  RAM: {mem_avail:.0f} GB/{mem_total:.0f} GB (Avail/Total) ")
    logger.info("=====================================\n")
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
                                        norm_dataframe, 
                                        filter_one_hit_wonder)

    # Input files
    MSFiles = args.ms_files
    IdFile = args.prec_ids
    LibFile = args.spec_lib
    
    # Parameters
    ## Use existing results
    SkipIDProcessing = args.skip_id_processing
    UseSavedXIC = args.use_saved_xic
    UseXICQuant = args.use_xic_quant
    ## Precursor quantification
    UseRawMS1 = args.use_raw_ms1
    UseRawMS2 = args.use_raw_ms2
    XicQualityCutoff = args.xic_quality_cutoff
    AllowedXicMissing = args.allowed_xic_missing
    MinNumXic = args.min_n_xic
    FillMissingXic = args.fill_missing_xic
    FillMethod = args.fill_method
    LocalMinScale = args.local_min_scale
    ## Precursor normalization
    Normalization = args.normalization
    NormSpecies = args.norm_species
    NormMethod = args.norm_method
    TrimPercentage = args.trim_percentage
    ## Protein quantification
    PrecQuantLevel = args.prec_quant_level
    MinPrecQuality = args.min_prec_quality
    Min1HitWQuality = args.min_1hitw_quality
    MinPrecCount = args.min_prec_count
    MinPepCount = args.min_pep_count
    
    # XIC quantification
    xic_quant_file = out_pub_dir / 'xic_quantification.parquet'
    if UseXICQuant:
        logging.info(f"Using existing XIC quantification result: {xic_quant_file}") 
    else:
        logging.info(f"{len(MSFiles)} runs will be analyzed")
        
        ## Skip ID results processing
        if SkipIDProcessing:
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
            # determine if performing garbage collection
            if psutil.virtual_memory().available / (1024 ** 3) < 32:
                logging.info("Available RAM less than 32 GB, performing garbage collection")
                gc.collect()

        ## XIC quantification across runs
        lib.info.load(lib, MSFiles, str(out_int_dir))
        lib.info.quantify(MSFiles, str(out_int_dir))

        ## Write report
        prec_prot_relation = get_prec_prot_relation(LibFile, str(out_int_dir))
        lib.report(xic_quant_file, MSFiles, prec_prot_relation)
    
    # Precursor quantification
    sample_meta_file = args.sample_meta
    out_prec_dir = out_int_dir / 'prec_quant'
    prec_quantity_table, prec_quality_table = quantify_precursors(
                                                  xic_quant_file, 
                                                  sample_meta_file,
                                                  use_raw_ms1=UseRawMS1,
                                                  use_raw_ms2=UseRawMS2,
                                                  xic_quality_cutoff=XicQualityCutoff,
                                                  allowed_xic_missing=AllowedXicMissing,
                                                  min_n_xic=MinNumXic,
                                                  fill_missing_xic=FillMissingXic,
                                                  fill_method=FillMethod,
                                                  fill_factor=LocalMinScale,
                                                  prec_quant_level=PrecQuantLevel,
                                                  out_dir=str(out_prec_dir))
    prec_quantity_table.write_parquet(out_pub_dir / "precursor_quantity_summ_table.parquet")
    prec_quality_table.write_parquet(out_pub_dir / "precursor_quality_summ_table.parquet")
    
    # Precursor normalization
    # NormSpecies='HUMAN'
    # NormMethod='median'
    # TrimPercentage=5
    prec_quantity_table_norm = norm_dataframe(prec_quantity_table, 
                                              data_type='Precursor',
                                              tag_column='Protein.Names', 
                                              value_columns=prec_quantity_table.columns[4:-2], 
                                              normalization=Normalization,
                                              norm_species=NormSpecies, 
                                              norm_method=NormMethod, 
                                              trim_percentage=TrimPercentage)
    prec_quantity_table_norm.write_parquet(out_pub_dir / "precursor_quantity_summ_table_norm.parquet")
    
    # Protein quantification
    prot_quantity_table, prot_quality_table = quantify_proteins(prec_quantity_table_norm, 
                                                                prec_quality_table,
                                                                min_prec_quality=MinPrecQuality)
    # Calculate the number of peptides of a protein
    prot_prec_pep_info = get_prot_prec_pep_info(xic_quant_file)
    prot_quantity_table = pl.from_pandas(prot_prec_pep_info).join(prot_quantity_table, 
                                                                  on='Protein.Group', 
                                                                  how='inner')
    prot_name_cols = ['Protein.Group', 'Protein.Names', 'Genes']
    remaining_cols = [col for col in prot_quantity_table.columns if col not in prot_name_cols]
    prot_quantity_table = prot_quantity_table.select(prot_name_cols + remaining_cols)

    prot_quantity_table.write_csv(out_pub_dir / "protein.group_quantity_matrix_all.csv")
    prot_quality_table.write_csv(out_pub_dir / "protein.group_quality_matrix_all.csv")
    
    # Filter one-hit wonder (protein with one precursor)
    (prot_quantity_table_1st_f, 
     prot_quantity_table_2nd_f) = filter_one_hit_wonder(prot_prec_pep_info, 
                                                        prot_quantity_table, 
                                                        prot_quality_table, 
                                                        sample_meta_file, 
                                                        min_1hitw_quality=Min1HitWQuality, 
                                                        min_prec_count=MinPrecCount,
                                                        min_pep_count=MinPepCount)
    prot_quantity_table_1st_f.write_csv(out_pub_dir / "protein.group_quantity_matrix.csv")
    prot_quantity_table_2nd_f.write_csv(out_pub_dir / "protein.group_quantity_matrix_rm1hitw.csv")
    
    # Calculate protein coverage (mainly used for GUI)
    FastaFiles = args.fastas
    if len(FastaFiles):
        try:
            logging.info("Calculating protein coverage based on provided FASTA file(s)")
            prot_prec_pep_info = calculate_protein_coverage(prot_prec_pep_info, FastaFiles)
        except FileNotFoundError as e:
            logging.warning(f"WARNING: cannot read the FASTA file. {e}")
    else:
        logging.warning("WARNING: no FASTA files are provided, protein coverage will not be calculated.")
    prot_prec_pep_info.to_parquet(out_int_dir / "prot_prec_pep_info.parquet", index=False)
    
    logging.info("Finished!")
    close_logger(logger)