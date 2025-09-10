# builtin
import os
import re
import sys
import time
import logging
import argparse
import configparser
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field, fields
# local
import jumpdiaq
from jumpdiaq.utils import get_extension, MSFormatsExt

@dataclass
class Arguments:
    # Singleton pattern: ensure it has only one instance throughout the program
    _instance = None
    
    # Inputs
    input_runs: List[str] = field(default_factory=list)
    id_report: str = ""
    spec_lib: str = ""
    out_dir: str = ""
    
    # Use existing results
    skip_id_process: int = 0
    use_saved_xic: int = 0
    use_xic_report: int = 0
    
    # ID results processing
    min_runs_quant: int = 3
    min_runs_model: int = 3
    min_runs_cal: int = 6
    min_qvalue_cal: float = 0.0001
    max_mass_cal: int = 5000
    
    # XIC extraction
    n_xic_points: int = 21
    fix_mass_acc: int = 0
    mass_acc_ms2: float = 15.0
    mass_acc_ms1: float = 15.0
    individual_mass_acc: int = 0
    ## Mass calibration
    mass_calibration: int = 1    
    mass_cal_acc: float = 30.0   # calibration mass accuracy
    mass_cal_bins: int = 1
    mass_cal_bins_max: int = 1
    mass_cal_score: float = 0.9
    ## IM window settings
    im_window_cal: float = 0.03
    fix_im_window: int = 0
    im_window: float = 0.01
    individual_im_window: int = 0
    
    # XIC quantification
    min_fr_quant: int = 3
    max_fr_quant: int = 6
    max_pr_isotopes: int = 5
    optimal_xic_source: str = "ms1_ms2" # Other options: 'ms1', 'ms2'
    intf_removal: int = 1
    intf_factor_ms1: float = 1.2
    intf_factor_ms2: float = 1.5
    
    # Precursor summarization
    xic_quality_cutoff: float = 0.0
    fill_missing_xic: int = 1
    use_raw_ms1xic: int = 0
    use_raw_ms2xic: int = 1
    
    # Precursor normalization
    norm_species: str = "HUMAN"
    norm_method: str = "median"       # Other options: 'mean'
    trim_percentage: int = 5
    
    # Protein summarization
    prec_quant_level: str = "ms2" # Other options: 'ms1', 'ms2'
    fastas: List[str] = field(default_factory=list)
    
    # timsTOF data preprocessing (hidden from users)
    tims_centroid: int = 0
    weighted_centroid: int = 1
    mass_acc_centroid: int = 5
    im_window_centroid: float = 0.005
    
    # Other hidden parameters
    gen_params: str = ""
    config_file: str = ""
    show_progress: int = 1
    inter_progress: int = 0     
    save_supplements: int = 1
    
    # only load files in specific formats
    ms_files: List[str] = field(default_factory=list)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Arguments, cls).__new__(cls)
            # Initialize dataclass fields here
            for _field in fields(cls):
                if not hasattr(cls._instance, _field.name):
                    setattr(cls._instance, _field.name, _field.default_factory() if _field.default_factory is not _field.default_factory else _field.default)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        
    def generate_file(self, output_path="jumpdiaq.params"):
        with open(output_path, 'w') as f:
            f.write("[required]\n")
            f.write("# Input runs: .raw/.mzML files for Thermo instruments or diaPASEF .d files for timsTOF instruments\n")
            f.write("# Leave the 'input_runs = ' row empty when there are multiple files. The same applies to the 'fastas = ' parameter.\n")
            for fdef in fields(self):
                if fdef.name == "tims_centroid":
                    break
                if fdef.name == "id_report":
                    f.write("\n# Precursor-level idetification report\n")
                if fdef.name == "spec_lib":
                    f.write("\n# Spectral library (support .tsv/.csv/.parquet format)\n") 
                if fdef.name == "out_dir":
                    f.write("\n# Output directory name\n")
                if fdef.name == "skip_id_process":
                    f.write("\n\n[optional]\n")
                    f.write("# Use existing results\n")
                if fdef.name == "min_runs_quant":
                    f.write("\n# ID results processing\n")
                if fdef.name == "n_xic_points":
                    f.write("\n# XIC extraction\n")
                if fdef.name == "min_fr_quant":
                    f.write("\n# XIC quantification\n")
                if fdef.name == "xic_quality_cutoff":
                    f.write("\n# Precursor summarization\n")
                if fdef.name == "norm_species":
                    f.write("\n# Precursor normalization\n")
                if fdef.name == "prec_quant_level":
                    f.write("\n# Protein summarization\n")
                value = getattr(self, fdef.name)
                if isinstance(value, list):
                    f.write(f"{fdef.name} = \n")
                    for item in value:
                        f.write(f"{item}\n")
                else:
                    f.write(f"{fdef.name} = {value}\n")
        print(f"Parameter file generated to: {Path(output_path).resolve()}")
        
def create_parser() -> argparse.ArgumentParser:
    VERSION = jumpdiaq.__version__
    parser = argparse.ArgumentParser(
        description=f"JUMPdiaq (v{VERSION})",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=120)
    )
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {VERSION}')
    
    # Add positional argument for params file
    parser.add_argument("param_file", type=str, nargs='?', help="use a parameter file (.params)")
    parser.add_argument("--gen-params", type=str, nargs='?', const="jumpdiaq.params", metavar="PARAM_FILE",
                        help="generate a parameter file with default values (default: %(const)s)")
    
    # Create argument groups   
    required_group = parser.add_argument_group('required arguments')
    optional_group = parser.add_argument_group('optional arguments')
    
    # Inputs 
    required_group.add_argument("--input-runs", help="paths of raw files", nargs='*', default=[])
    required_group.add_argument("--id-report", type=str, help="path to identification report")
    required_group.add_argument("--spec-lib", type=str, help="path to spectral library")
    required_group.add_argument("--out-dir", type=str, help="output directory name")
    
    # Use existing results
    optional_group.add_argument("--skip-id-process", type=int, help="")
    optional_group.add_argument("--use-saved-xic", type=int, help="")
    optional_group.add_argument("--use-xic-report", type=int, help="")
    
    # ID results processing
    optional_group.add_argument("--min-runs-quant", type=int, help="")
    optional_group.add_argument("--min-runs-model", type=int, help="")
    optional_group.add_argument("--min-runs-cal", type=int, help="")
    optional_group.add_argument("--min-qvalue-cal", type=float, help="")
    optional_group.add_argument("--max-mass-cal", type=int, help="")
    
    # XIC extraction
    optional_group.add_argument("--n-xic-points", type=int, help="")
    optional_group.add_argument("--fix-mass-acc", type=int, help="")
    optional_group.add_argument("--mass-acc-ms2", type=float, help="")
    optional_group.add_argument("--mass-acc-ms1", type=float, help="")
    optional_group.add_argument("--individual-mass-acc", type=int, help="")
    ## Mass calibration
    optional_group.add_argument("--mass-calibration", type=int, help="") 
    optional_group.add_argument("--mass-cal-acc", type=float, help="")  # calibration mass accuracy
    optional_group.add_argument("--mass-cal-bins", type=int, help="")
    optional_group.add_argument("--mass-cal-bins-max", type=int, help="")
    optional_group.add_argument("--mass-cal-score", type=float, help="")
    ## IM window settings
    optional_group.add_argument("--im-window-cal", type=float, help="")
    optional_group.add_argument("--fix-im-window", type=int, help="")
    optional_group.add_argument("--im-window", type=float, help="")
    optional_group.add_argument("--individual-im-window", type=int, help="")
    
    # XIC quantification
    optional_group.add_argument("--min-fr-quant", type=int, help="")
    optional_group.add_argument("--max-fr-quant", type=int, help="")
    optional_group.add_argument("--max-pr-isotopes", type=int, help="")  
    optional_group.add_argument("--optimal-xic-source", type=str, help="")
    optional_group.add_argument("--intf-removal", type=int, help="")
    optional_group.add_argument("--intf-factor-ms1", type=float, help="")
    optional_group.add_argument("--intf-factor-ms2", type=float, help="")
    
    # Precursor summarization
    optional_group.add_argument("--use-raw-ms1xic", type=float, help="")
    optional_group.add_argument("--use-raw-ms2xic", type=int, help="")
    optional_group.add_argument("--xic-quality-cutoff", type=float, help="")
    optional_group.add_argument("--fill-missing-xic", type=int, help="")
    
    # Precursor normalization
    optional_group.add_argument("--norm-species", type=str, help="")
    optional_group.add_argument("--norm-method", type=str, help="")
    optional_group.add_argument("--trim-percentage", type=int, help="")
    
    # Protein summarization
    optional_group.add_argument("--prec-quant-level", type=str, help="")
    optional_group.add_argument("--fastas", help="", nargs='*', default=[])
    
    # timsTOF data preprocessing
    parser.add_argument("--tims-centroid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--weighted-centroid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mass-acc-centroid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--im-window-centroid", type=float, help=argparse.SUPPRESS)
    
    # Other hidden parameters
    # Config file
    parser.add_argument("--config-file", type=str, help=argparse.SUPPRESS)
    # Whether to save additional supplementary files
    parser.add_argument("--save-supplements", type=int, help=argparse.SUPPRESS)
    # Progress bar dispaly
    parser.add_argument("--show-progress", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--inter-progress", type=int, help=argparse.SUPPRESS)
    
    return parser


def parse_config_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        content = file.read()
    # Replace newlines with spaces and split into arguments
    return re.sub(r'\n|\r', ' ', content).split()


def parse_arguments(args: Optional[List[str]] = None) -> Arguments:
    """Parse arguments from command line."""
    if args is None:
        args = sys.argv[1:]
    
    parser = create_parser()
    
    # Use a parameter file
    if len(args) > 0 and not args[0].startswith('-'):
        potential_param_file = args[0]
        if os.path.isfile(potential_param_file):
            # It's a parameter file, convert to arguments
            param_args = params_to_args(potential_param_file)
            # Combine with any other command line arguments (which will override params)
            args = param_args + args[1:]
        else:
            print(f"Error: Parameter file {potential_param_file} not found.")
            return None, None
    
    # Use a config file
    first_pass_args, _ = parser.parse_known_args(args)
    if first_pass_args.config_file:
        if not Path(first_pass_args.config_file).exists():
            print(f"Error: Config file {first_pass_args.config_file} not found.")
        else:
            config_args = parse_config_file(first_pass_args.config_file)
            args = config_args
    
    command = 'jumpdiaq ' + ' '.join(args)
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
        
    if arguments.fix_mass_acc:
        arguments.individual_mass_acc = False
        print(f"Using fixed mass accuracy for all runs: {arguments.mass_acc_ms1}, {arguments.mass_acc_ms2} ppm (MS1, MS2)")
    
    if arguments.individual_mass_acc:
        arguments.fix_mass_acc = False
        print("Mass accuracy will be determined for each run individually")
    
    if len(arguments.out_dir):
        out_dir = Path(arguments.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        out_pub_dir = out_dir / 'publication'
        out_pub_dir.mkdir(parents=True, exist_ok=True)
        out_int_dir = out_dir / 'intermediate'
        out_int_dir.mkdir(parents=True, exist_ok=True)
        out_xic_dir = out_int_dir / 'xic_files'
        out_xic_dir.mkdir(parents=True, exist_ok=True)
        out_prec_dir = out_int_dir / 'prec_quant'
        out_prec_dir.mkdir(parents=True, exist_ok=True)
    
    if arguments.use_xic_report:
        report_file = Path(out_pub_dir) / 'xic_quantification.parquet'
        if not report_file.exists():
            print(f"Cannot find the XIC quantification file {report_file}. Parameter 'use_xic_report' reset to 0.")
            arguments.use_xic_report = 0
    
    return arguments, command


def preprocess_params(param_file):
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
    

def setup_logger(show_log_level, out_dir):
    """Sets up the main logger with console, file, and queue handlers."""
    
    level = logging.INFO  # Default level

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
    log_file_name = "jumpdiaq_log_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
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