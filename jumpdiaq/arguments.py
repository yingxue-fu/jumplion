# builtin
import sys
import os
import time
import argparse
import configparser
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging
# local
import jumpdiaq
from jumpdiaq.utils import get_extension, MSFormatsExt

@dataclass
class Arguments:
    # Singleton pattern: ensure it has only one instance throughout the program
    _instance = None
    
    # Input/Output
    input_runs: List[str] = field(default_factory=list)
    id_report: str = ""
    spec_lib: str = ""
    fastas: List[str] = field(default_factory=list)
    out_dir: str = ""
    
    # General settings
    log_level: int = 4
    use_saved_rts: int = 0
    use_saved_xic: int = 0
    use_xic_report: int = 0
    
    # ID results processing
    min_runs_quant: int = 1
    min_runs_model: int = 2
    min_runs_cal: int = 3
    min_qvalue_cal: float = 0.0001
    max_mass_cal: int = 3000
    
    # Mass calibration
    mass_calibration: int = 1    
    mass_cal_acc: float = 50.0   # calibration mass accuracy
    mass_cal_bins: int = 1
    mass_cal_bins_max: int = 1
    fix_mass_acc: int = 1
    mass_acc_ms2: float = 15.0
    mass_acc_ms1: float = 15.0
    individual_mass_acc: int = 0
    
    # IM window
    im_window_cal: float = 0.03
    fix_im_window: int = 0
    im_window: float = 0.01
    individual_im_window: int = 0
    
    # XIC extraction and scoring
    min_quan_fr: int = 3
    max_quan_fr: int = 6
    n_isotops: int = 2
    min_xic_points: int = 20
    corr_radius: int = 5
    corr_method: int = 0
    weight_by_shape: int = 1
    weight_by_intensity: int = 1
    min_shape_score: float = 0.3
    min_xic_intensity: float = 10.0
    min_apex_intensity: float = 30.0
    min_apex_evidence: float = 0.99
    optimal_xic_source: int = 0
    
    # Interference removal
    int_removal: int = 1
    int_removal_corr: float = 0.0
    int_filter_factor: float = 1.5
    
    # Normalization
    norm_species: str = "HUMAN"
    norm_method: int = 0
    trim_percentage: int = 5
    
    # only load files in specific formats
    ms_files: List[str] = field(default_factory=list)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Arguments, cls).__new__(cls)
            # Initialize dataclass fields here
            from dataclasses import fields
            for field in fields(cls):
                if not hasattr(cls._instance, field.name):
                    setattr(cls._instance, field.name, field.default_factory() if field.default_factory is not field.default_factory else field.default)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"JUMPdiaq (v{jumpdiaq.__version__})",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=120)
    )
    
    # Add positional argument for params file
    parser.add_argument("params_file", type=str, nargs='?', help="Use a parameter file")
    
    # Create argument groups   
    required_group = parser.add_argument_group('required arguments')
    optional_group = parser.add_argument_group('optional arguments')
    
    # Input/Output 
    required_group.add_argument("--input-runs", help="Paths of MS files", nargs='*', default=[])
    required_group.add_argument("--id-report", type=str, help="Path to identification report")
    required_group.add_argument("--spec-lib", type=str, help="Path to spectral library")
    required_group.add_argument("--fastas", help="Paths of Fasta files", nargs='*', default=[])
    required_group.add_argument("--out-dir", type=str, help="Output directory name")
    
    # General settings
    optional_group.add_argument("--log-level", type=int, help="Level of log messages")
    optional_group.add_argument("--use-saved-rts", type=int, help="Use existing precursor RTs (and IMs)")
    optional_group.add_argument("--use-saved-xic", type=int, help="Use existing XIC files")
    optional_group.add_argument("--use-xic-report", type=int, help="Use existing XIC quant report file")
    
    # ID results processing
    optional_group.add_argument("--min-runs-quant", type=int, help="")
    optional_group.add_argument("--min-runs-model", type=int, help="")
    optional_group.add_argument("--min-runs-cal", type=int, help="")
    optional_group.add_argument("--min-qvalue-cal", type=float, help="")
    optional_group.add_argument("--max-mass-cal", type=int, help="")
    
    # Mass calibration
    optional_group.add_argument("--mass-calibration", type=int, help="") 
    optional_group.add_argument("--mass-cal-acc", type=float, help="")  # calibration mass accuracy
    optional_group.add_argument("--mass-cal-bins", type=int, help="")
    optional_group.add_argument("--mass-cal-bins-max", type=int, help="")
    optional_group.add_argument("--fix-mass-acc", type=int, help="")
    optional_group.add_argument("--mass-acc-ms2", type=float, help="")
    optional_group.add_argument("--mass-acc-ms1", type=float, help="")
    optional_group.add_argument("--individual-mass-acc", type=int, help="")
    
    # IM window 
    optional_group.add_argument("--im-window-cal", type=float, help="")
    optional_group.add_argument("--fix-im-window", type=int, help="")
    optional_group.add_argument("--im-window", type=float, help="")
    optional_group.add_argument("--individual-im-window", type=int, help="")
    
    # XIC extraction and scoring
    optional_group.add_argument("--min-quan-fr", type=int, help="")
    optional_group.add_argument("--max-quan-fr", type=int, help="")
    optional_group.add_argument("--n-isotops", type=int, help="")
    optional_group.add_argument("--min-xic-points", type=int, help="")
    optional_group.add_argument("--corr-radius", type=int, help="")
    optional_group.add_argument("--corr-method", type=int, help="")
    optional_group.add_argument("--weight-by-shape", type=int, help="")
    optional_group.add_argument("--weight-by-intensity", type=int, help="")
    optional_group.add_argument("--min-shape-score", type=float, help="")
    optional_group.add_argument("--min-xic-intensity", type=float, help="")
    optional_group.add_argument("--min-apex-intensity", type=float, help="")
    optional_group.add_argument("--min-apex-evidence", type=float, help="")   
    optional_group.add_argument("--optimal-xic-source", type=int, help="")
    
    # Interference removal 
    optional_group.add_argument("--int-removal", type=int, help="")
    optional_group.add_argument("--int-removal-corr", type=float, help="")
    optional_group.add_argument("--int-filter-factor", type=float, help="")
    
    # Normalization
    optional_group.add_argument("--norm-species", type=str, help="")
    optional_group.add_argument("--norm-method", type=int, help="")
    optional_group.add_argument("--trim-percentage", type=int, help="")
    
    return parser


def parse_arguments(args: Optional[List[str]] = None) -> Arguments:
    """Parse arguments from command line."""
    if args is None:
        args = sys.argv[1:]
    
    parser = create_parser()
    
    # Check if first argument might be a parameter file
    if len(args) > 0 and not args[0].startswith('-'):
        potential_param_file = args[0]
        if os.path.isfile(potential_param_file):
            # It's a parameter file, convert to arguments
            param_args = params_to_args(potential_param_file)
            # Combine with any other command line arguments (which will override params)
            args = param_args + args[1:]
            print_cmds = ' '.join(args)
            print(f"\nCommand: {print_cmds}\n")
            
    parsed_args = parser.parse_args(args)
    
    # Update arguments from parsed arguments
    arguments = Arguments()
    for key, value in vars(parsed_args).items():
        if value is not None:
            if hasattr(arguments, key.replace('-', '_')):
                setattr(arguments, key.replace('-', '_'), value)
    
    # Process MS files
    arguments.ms_files = []
    if arguments.input_runs:
        for file_path in arguments.input_runs:
            extension = get_extension(file_path)
            if extension not in MSFormatsExt:
                print(f"WARNING: Skipping {file_path} - invalid raw MS data format")
                continue
            arguments.ms_files.append(file_path)   

    if not arguments.spec_lib:
        print("WARNING: no spectral library provided: nothing to do\n")
        
    if arguments.fix_mass_acc:
        arguments.individual_mass_acc = False
        print(f"Using fixed mass accuracy for all runs: {arguments.mass_acc_ms1}, {arguments.mass_acc_ms2} ppm (MS1, MS2)")
    
    if arguments.individual_mass_acc:
        arguments.fix_mass_acc = False
        print("Mass accuracy will be determined for each run individually")
    
    if len(arguments.out_dir):
        out_dir = Path(arguments.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        out_xic_dir = out_dir / 'xic_files'
        out_xic_dir.mkdir(parents=True, exist_ok=True)
    
    if arguments.use_xic_report:
        report_file = Path(out_dir) / 'xic_quan_report.parquet'
        if not report_file.exists():
            arguments.use_xic_report = 0
    
    return arguments


def preprocess_params(param_file):
    # param_file = r"C:\Users\yfu\Downloads\miss_dia_test_data\trixi_data\jumpdiaq.params"
    
    # Read the file
    with open(param_file, 'r') as f:
        content = f.read()
    
    # Split content into sections and process each section
    sections = []
    current_section = []
    lines = content.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines and full-line comments
        if not line.strip() or line.strip().startswith('#'):
            i += 1
            continue
        
        # If this is a section header
        if line.strip().startswith('['):
            # Add the previous section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line.split('#')[0].strip())
        else:
            # This is a parameter line
            cleaned_line = line.split('#')[0].strip()
            if '=' in cleaned_line:
                # Parameter definition starts
                param_name = cleaned_line.split('=')[0].strip()
                param_value = cleaned_line.split('=')[1].strip()
                
                # Check if this is a multi-line parameter
                if not param_value:
                    # Multi-line parameter
                    multi_line_values = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('#') and '=' not in lines[j]:
                        multi_line_values.append(lines[j].split('#')[0].strip())
                        j += 1
                    
                    # Add the parameter with all its values
                    current_section.append(f"{param_name} = \n    " + "\n    ".join(multi_line_values))
                    i = j - 1  # Adjust the index to continue from the right place
                else:
                    # Single-line parameter
                    current_section.append(f"{param_name} = {param_value}")
        
        i += 1
    
    # Add the last section if it exists
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Join all sections with double newlines
    cleaned_params = '\n\n'.join(sections)
    return cleaned_params


def params_to_args(param_file):
    
    parser = create_parser()
    default_args = parser.parse_args([])
    
    # param_file = "jumpdiaq.params"
    cleaned_params = preprocess_params(param_file)
    config = configparser.ConfigParser()
    config.read_string(cleaned_params)
    
    # Prepare list to collect commands
    commands = []
    
    for section in config.sections():
        for key, value in config.items(section):
            # Get the default value from the default_args object
            default_value = getattr(default_args, key, None)
    
            # Handle multi-line values
            value = value.strip()
            if '\n' in value:
                # Convert multi-line into list of strings
                value_list = [line.strip() for line in value.splitlines() if line.strip()]
                # Compare with default (assuming default_value is also a list)
                if value_list != default_value:
                    # Append as space-separated values
                    commands.append(f"--{key.replace('_', '-')}")
                    commands = commands + value_list
            else:
                # Compare simple values (as strings for safety)
                if str(value) != str(default_value):
                    if key == 'out_dir':
                        value = str(Path(param_file).resolve().parent / value)
                    commands.append(f"--{key.replace('_', '-')}")
                    commands.append(value)
    
    return commands 
    

def setup_logger(log_level, show_log_level, out_dir):
    """Sets up the main logger with console, file, and queue handlers."""
    
    level = logging.WARNING  # Default level
    if log_level == 5:
        level = logging.DEBUG
    elif log_level == 4:
        level = logging.INFO

    # Create a named logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    while logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()

    # Create handlers
    console_handler = logging.StreamHandler()
    log_file_name = "jumpdia_log_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
    file_handler = logging.FileHandler(Path(out_dir) / log_file_name, mode="w", encoding="utf-8")  # Overwrite mode
    
    # Set levels and formats
    if show_log_level:
        log_format = logging.Formatter(fmt="%(levelname)9s: %(asctime)s %(message)s", datefmt="[%H:%M:%S]")
    else:
        log_format = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="[%H:%M:%S]")
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    console_handler.setLevel(level)
    file_handler.setLevel(level)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def close_logger(logger):
    """Closes and removes all handlers from the logger."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)