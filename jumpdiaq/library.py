# builtin
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
# external
import numpy as np
import polars as pl
# local
from jumpdiaq.arguments import Arguments, get_extension
from jumpdiaq.precursor import (Precursor, to_eg, to_canonical_with_charge, 
                                pep_name, get_aas, get_sequence_masses, 
                                peptide_length, Product, annotate_products_charges)
from jumpdiaq.utils import (remove_extension, 
                            location_to_file_name, 
                            get_file_name, progress_bar,
                            MaxLibFrNum, fTypeB, fTypeY, fExclude, 
                            char_of_fType, name_of_loss,
                            loss_none, loss_H2O, loss_NH3, loss_CO, loss_other)
from jumpdiaq.elution_peaks import identify_optimal_xic, calculate_quantity

args = Arguments()
nIsotops = args.n_isotops
MaxFr = args.max_quan_fr
CorrRadius = args.corr_radius
CorrMethod = args.corr_method
WeightByShape = args.weight_by_shape
WeightByIntensity = args.weight_by_intensity
MinShapeScore = args.min_shape_score
MinXicIntensity = args.min_xic_intensity
MinApexIntensity = args.min_apex_intensity
MinApexEvidence = args.min_apex_evidence
IntRemoval = args.int_removal
IntRemovalCorr = args.int_removal_corr
IntFilterFactor = args.int_filter_factor
ShowProgress = args.show_progress
InterProgress = args.inter_progress

# Mapping of column indices 
libPr = 0
libCharge = 1
libPrMz = 2
libiRT = 3
libIM = 4
libFrMz = 5
libFrI = 6
libFrCharge = 7
libFrType = 8
libFrNumber = 9
libFrLoss = 10
libFrExc = 11

# List of possible library headers for each column
library_headers = [
    ("PrecursorId", "Precursor"),
    ("PrecursorCharge", "Charge", "prec_z"),
    ("PrecursorMz", "prec_mz"),
    ("iRT", "RT", "RetentionTime", "NormalizedRetentionTime", "Tr_recalibrated", "RT_detected"),
    ("IM", "iIM", "mobility", "IonMobility"),
    ("FragmentMz", "ProductMz"),
    ("RelativeIntensity", "RelativeFragmentIntensity", "LibraryIntensity", "relative_intensity"),
    ("FragmentCharge", "FragmentIonCharge", "ProductCharge", "frg_z"),
    ("FragmentType", "FragmentIonType", "ProductType", "frg_type"),
    ("FragmentNumber", "FragmentSeriesNumber"),
    ("FragmentLossType", "ProductLossType"),
    ("ExcludeFromAssay", 'ExcludeFromQuant', "ExcludeFromQuantification"),
]


class LibEntry:
    def __init__(self, library="", name="", eg_id=0):
        self.lib = library
        self.name = name              # Precursor ID
        self.eg_id = eg_id            # elution group
        self.prec = Precursor()
        
    def init(self):
        # Sort the product ions by descending height (relative intensity) and keep the top n product ions
        self.prec.products.sort(key=lambda x: x.height, reverse=True)
        self.prec.products = self.prec.products[:MaxLibFrNum]
        # Normalize product relative intensity
        max_height = max((prod.height for prod in self.prec.products), default=1)
        for prod in self.prec.products:
            prod.height /= max_height
        # Set peptide length
        if self.prec.nAAs == 0:
            self.prec.nAAs = peptide_length(self.name)
    
    def free(self):
        self.prec.free()
        

class Library:
    def __init__(self):      
        self.name = ""
        self.iRT_min = 0.0
        self.iRT_max = 0.0
        self.gen_charges = True
               
        self.precursors = []      # precursor IDs in canonical format; library-based entries only
        self.entries = []
        self.eg = []        # elution groups
        self.eg_index = []  # same length as entries, indices of the elution groups
        self.co_elution = []
        self.co_elution_index = []
        
        self.info = Info()
    
    def load(self, file_path):
        
        self.name = str(file_path)
        logging.info(f"Loading spectral library {self.name}")
        
        lib_format = get_extension(self.name)
        
        if lib_format in ["parquet", "csv", "tsv"]:
            self.load_table_format_library(file_path)
        else:
            logging.error("ERROR: Spectral library format not supported.")
                
        self.elution_group_index()  
        logging.info(f"Spectral library loaded: {len(self.entries)} precursors in {len(self.co_elution_index)} elution groups.\n")
        
        return True
        
    def load_table_format_library(self, file_path):
        # file_path=Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\jumpdia_test_withIM\splib_quan.parquet")
        try:
            if file_path.suffix == '.parquet':
                df = pl.read_parquet(file_path)
            else:
                delim = ',' if file_path.suffix == '.csv' else '\t'
                df = pl.read_csv(file_path, sep=delim)
        except FileNotFoundError:
            logging.error("Cannot read the file")
            return False
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return False
        
        # Initialize flags and data structures
        ptw = pzw = ptmw = False
        library_data = {}
        eg = set()
        
        # Identify column indices based on column headers
        col_indices = {}
        for col_index, header in enumerate(df.columns):
            for header_options in library_headers:
                if re.sub(r"[\s\._-]", "", header.lower()) in [h.lower() for h in header_options]:
                    col_indices[library_headers.index(header_options)] = col_index
                    break
        
        if libFrCharge not in col_indices:
            logging.warning("WARNING: no product ion charge information found in the library - assuming fragments with charge 1\n")
        if libFrLoss not in col_indices:
            logging.warning("WARNING: no neutral loss information found in the library - assuming fragments without losses\n")
        if libFrNumber not in col_indices:
            logging.warning("WARNING: no product ion number information found in the library\n")
        if libFrType not in col_indices:
            logging.warning("WARNING: no product ion type information found in the library\n")
            
        for row in df.iter_rows():
            # convert the precursor to a unified format
            precursor_lib = row[col_indices[libPr]]
            charge = row[col_indices[libCharge]]
            precursor_id = to_canonical_with_charge(precursor_lib, charge)
            
            # create entry object
            if precursor_id not in library_data:
                entry = LibEntry()
                entry.lib = self
                library_data[precursor_id] = entry
                entry.name = precursor_id
                entry.prec.nAAs = peptide_length(precursor_id)
                eg_name = to_eg(precursor_id)
                eg.add(eg_name)
                entry.eg_id = len(eg) - 1
        
            entry = library_data[precursor_id]
            product = Product(mz=row[col_indices[libFrMz]],
                              height=row[col_indices[libFrI]])
            
            if libFrCharge in col_indices:
                self.gen_charges = False
                product.charge = row[col_indices[libFrCharge]]
                if product.charge <= 0:
                    product.charge = 1
                    if not pzw: # product ion charge warning
                        pzw = True 
                        logging.warning("WARNING: product ion charge 0 specified - assuming 1")
                if libFrType in col_indices:
                    fragment_type = True
                    ft = row[col_indices[libFrType]]
                    if fragment_type:
                        if ft == 'y':
                            product.ion_type = fTypeY
                        elif ft == 'b':
                            product.ion_type = fTypeB
                        elif not ptw: # product ion type warning
                            ptw = True
                            product.ion_type = 0
                            logging.warning(f"WARNING: unknown product ion type {ft}")
                    elif not ptmw:  # product ion type missing warning
                        ptmw = True
                        product.ion_type = 0
                        logging.warning(f"WARNING: product ion type missing for row {row}")
            
                    if product.ion_type and libFrNumber in col_indices:
                        product.index = int(row[col_indices[libFrNumber]])
                        if product.ion_type != fTypeB: # for y ions
                            product.index = entry.prec.nAAs - product.index
            
                        if libFrLoss in col_indices:
                            wl = row[col_indices[libFrLoss]]
                            if wl == "noloss":
                                product.loss = loss_none
                            elif wl == "H2O":
                                product.loss = loss_H2O
                            elif wl == "NH3":
                                product.loss = loss_NH3
                            elif wl == "CO":
                                product.loss = loss_CO
                            else:
                                product.loss = loss_other
                        else:
                            product.loss = loss_none
            else:
                product.charge = 1
                    
            if libFrExc in col_indices and str(row[col_indices[libFrExc]]).lower() in ['t', '1', 'true']:
                product.ion_type |= fExclude
            
            prec = entry.prec
            prec.mz = row[col_indices[libPrMz]]
            prec.iRT = row[col_indices[libiRT]]
            prec.iIM = row[col_indices[libIM]]
            prec.charge = charge
            prec.products.append(product)
        
        self.eg = list(eg)
        self.precursors = list(library_data.keys())
        self.entries = list(library_data.values())
        
        for i, entry in enumerate(self.entries):
            entry.prec.index = i

            self.iRT_min = min(self.iRT_min, entry.prec.iRT)
            self.iRT_max = max(self.iRT_max, entry.prec.iRT)
            self.eg_index.append(entry.eg_id)  
    
    @staticmethod
    def default_co_elution():
        return [0, 1]
    
    def elution_group_index(self):
        self.co_elution_index = defaultdict(self.default_co_elution)
        ce = defaultdict(list)
    
        for i, group in enumerate(self.eg_index):
            ce[group].append(i)
    
        self.co_elution = []
        peg = None
    
        for group, indices in ce.items():
            if peg is not None:
                self.co_elution_index[peg][1] = len(indices)
    
            self.co_elution_index[group][0] = len(self.co_elution)
            self.co_elution.extend(indices)
            peg = group
    
        if self.co_elution:
            self.co_elution_index[peg][1] = len(ce[peg])
    
    def initialize(self):
        corrected_charge = False
    
        for i, entry in enumerate(self.entries):
            if entry.prec.charge <= 0:
                entry.prec.charge = 2
                corrected_charge = True
            if self.gen_charges: 
                seq_mass = get_sequence_masses(entry.name)
                entry.prec.products = annotate_products_charges(seq_mass, 
                                                                entry.prec.products, 
                                                                entry.prec.charge)
            entry.init()
    
        if corrected_charge:
            logging.info("WARNING: at least one library precursor had non-positive charge; corrected to charge = 2\n")
            
            
    def report(self, file_name: str, ms_files, prec_prot_info):
        try:
            # Create a list to store the rows of the report
            report_data = []
    
            runs = [remove_extension(get_file_name(ms_file)) for ms_file in ms_files]
    
            for index, v in self.info.map.items():
                entry = self.entries[index]

                for jt in v:
                    
                    # Prepare the row data
                    row_data = {
                        "File.Name": ms_files[jt[0]],
                        "Run": runs[jt[0]],
                        "Modified.Sequence": pep_name(entry.name),
                        "Stripped.Sequence": get_aas(entry.name),
                        "Precursor.Id": entry.name,
                        "Precursor.Charge": entry.prec.charge,
                        "Precursor.Mz": entry.prec.mz,
                        "FWHM": jt[1].fwhm,
                        "RT.Apex": jt[1].RT,
                        "RT.Left": jt[1].RT_left,
                        "RT.Right": jt[1].RT_right,
                    }

                    row_data.update({
                        "MS1.XIC.Quant.Raw": ";".join([str(pr.quantity[0]) for pr in jt[1].ms1_quant]),
                        "MS1.XIC.Quant.Adjusted": ";".join([str(pr.quantity[1]) for pr in jt[1].ms1_quant]),
                        "MS1.XIC.Corr.Raw": ";".join([str(pr.corr[0]) for pr in jt[1].ms1_quant]),
                        "MS1.XIC.Corr.Adjusted": ";".join([str(pr.corr[1]) for pr in jt[1].ms1_quant]),
                        "MS2.XIC.Quant.Raw": ";".join([str(fr.quantity[0]) for fr in jt[1].ms2_quant]),
                        "MS2.XIC.Quant.Adjusted": ";".join([str(fr.quantity[1]) for fr in jt[1].ms2_quant]),
                        "MS2.XIC.Corr.Raw": ";".join([str(fr.corr[0]) for fr in jt[1].ms2_quant]),
                        "MS2.XIC.Corr.Adjusted": ";".join([str(fr.corr[1]) for fr in jt[1].ms2_quant])
                    })
                    
                    ms1_xic_names = ['M+0', 'M+1', 'M+2']
                    row_data.update({
                        "Optimal.XIC": jt[1].optimal_xic,
                        "Optimal.XIC.Type": f"ms{jt[1].optimal_xic_type}",
                        "MS1.XIC.Info": ";".join([f"{ms1_xic_names[i]}/{np.round(pr.mz, 4)}" 
                                                  for i, pr in enumerate(jt[1].ms1_quant)]),
                        "MS2.XIC.Info": ";".join([
                                             f"{char_of_fType[(cfr.ion_type & (fTypeB | fTypeY)) >> 0]}"  # Ion type (B or Y)
                                             f"{cfr.index if cfr.ion_type & fTypeB else len(get_aas(entry.name)) - cfr.index}"  # Index
                                             f"{'-' + name_of_loss[cfr.loss] if cfr.loss != loss_none else ''}"  # Loss info (if any)
                                             f"^{cfr.charge}/{np.round(cfr.mz, 4)}"  # Charge and m/z
                                             for fr in jt[1].ms2_quant 
                                             for cfr in [entry.prec.products[fr.index]]
                                             ])
                    })
    
                    # Append the row data to the report_data list
                    report_data.append(row_data)
    
            # Convert the report_data list to a DataFrame
            report_df = pl.DataFrame(report_data, orient="row")
            
            # add protein info
            report_df = report_df.join(pl.from_pandas(prec_prot_info), on='Precursor.Id', how='inner')
    
            # Save the DataFrame to a Parquet file
            report_df.write_parquet(file_name)
            logging.info(f"XIC quantification report saved to {file_name}")
    
        except IOError as e:
            logging.error(f"ERROR: cannot write to {file_name}. Check if the destination folder is write-protected or the file is in use: {e}")
            return
    
    
@dataclass
class XICQuant:
    index: int = 0  # Index of the product ion in the library
    mz: float = 0.0
    quantity: List[float] = field(default_factory=lambda: [0.0] * 2)  
    corr: List[float] = field(default_factory=lambda: [0.0] * 2)

@dataclass
class QuantEntry:
    index: int = 0
    run_index: int = 0

    apex: int = 0
    fwhm: int = 0
    
    xic_score = None
    optimal_xic: int = 0
    optimal_xic_type: int = 0

    RT: float = 0.0
    RT_left: float = 0.0
    RT_right: float = 0.0
    IM: float = 0.0

    ms1 = None
    ms2 = None
    ms1_i = None
    ms1_rt = None
    ms2_rt = None

    ms1_quant: List[XICQuant] = field(default_factory=lambda: [XICQuant() for _ in range(nIsotops + 1)])
    ms2_quant: List[XICQuant] = field(default_factory=lambda: [XICQuant() for _ in range(MaxFr)])

class Info:
    def __init__(self):
        self.lib: Optional[Library] = None
        self.n_runs: int = 0  # number of runs
        self.n_entries: int = 0
        self.map: Dict[int, List[Tuple[int, QuantEntry]]] = defaultdict(list)
        self.runs: List[List[QuantEntry]] = []

    def clear(self):
        self.map = defaultdict(list)
        self.runs.clear()

    def index(self, ms_files):
        self.runs.clear()
        self.runs = [[] for _ in range(max(self.n_runs, len(ms_files)))]

        for key, value in self.map.items():
            for run_index, qe in value:
                self.runs[run_index].append(qe)

    def load(self, parent: Library, ms_files: List[str], out_dir):
        self.n_runs = len(ms_files)
        self.lib = parent
        logging.info(f"Assembling XIC data of {self.n_runs} runs")
        
        lib_n = len(self.lib.entries)
        self.clear()
        
        for run_index, file in enumerate(ms_files):
            # run_index = 0
            # file = ms_files[run_index]
            run_name = remove_extension(get_file_name(file))
            xic_file = Path(out_dir) / "xic_files" / f"{run_name}.xic.parquet"
            logging.info(f"Loading XIC data from {xic_file}")
    
            try:
                df = pl.read_parquet(xic_file)
            except FileNotFoundError:
                alt_path = location_to_file_name(file) + ".xic.parquet"
                df = pl.read_parquet(alt_path)
    
            ms1_cols = [col for col in df.columns if col.startswith("ms1_intensity")]
            ms2_cols = [col for col in df.columns if col.startswith("ms2_intensity")]
            ms1_i_cols = [col for col in df.columns if col.startswith("ms1_interpol")]
    
            precursor_ids = df.select("Precursor.Id").unique().to_series()
            if len(precursor_ids) != lib_n:
                logging.warning("Number of precursors doesn't match between library and XIC parquet file.")
    
            precursor_groups = df.partition_by("Precursor.Id", as_dict=True)
    
            for precursor_id, subdf in progress_bar(precursor_groups.items(), ShowProgress, InterProgress):
                row_type_map = subdf.partition_by("row_type", as_dict=True)
                xic_rows = row_type_map.get(("xic_row",), pl.DataFrame())
                mz_row = row_type_map.get(("mz_row",), pl.DataFrame())
                index_row = row_type_map.get(("index_row",), pl.DataFrame())
                score_row = row_type_map.get(("score_row",), pl.DataFrame())
    
                if xic_rows.is_empty() or mz_row.is_empty() or index_row.is_empty() or score_row.is_empty():
                    logging.warning(f"Missing required rows for precursor {precursor_id}")
                    continue
    
                qe = QuantEntry()
                qe.index = int(index_row[ms1_cols].to_numpy()[0][0])
                qe.run_index = run_index
                
                qe.apex = int(score_row['ms1_rt'][0])
                qe.fwhm = int(score_row['ms2_rt'][0])
                
                qe.xic_score = score_row[ms1_cols+ms2_cols].to_numpy()[0]
                qe.optimal_xic = int(index_row['ms1_rt'][0])
                qe.optimal_xic_type = int(mz_row['ms1_rt'][0])
                
                qe.ms1 = xic_rows.select(ms1_cols).to_numpy()
                qe.ms2 = xic_rows.select(ms2_cols).to_numpy()
                qe.ms1_i = xic_rows.select(ms1_i_cols).to_numpy()
                qe.ms1_rt = xic_rows['ms1_rt'].to_numpy()
                qe.ms2_rt = xic_rows['ms2_rt'].to_numpy()
                
                qe.RT = qe.ms2_rt[qe.apex]
                qe.IM = 0.0 # todo 
                
                ms1_mzs = mz_row.select(ms1_cols).to_numpy()[0]
                for prec, mz in enumerate(ms1_mzs):
                    qe.ms1_quant[prec].index = prec
                    qe.ms1_quant[prec].mz = mz
                
                ms2_mzs = mz_row.select(ms2_cols).to_numpy()[0]
                ms2_indices = index_row.select(ms2_cols).to_numpy()[0]
                for fr, (idx, mz) in enumerate(zip(ms2_indices, ms2_mzs)):                   
                    qe.ms2_quant[fr].index = int(idx)
                    qe.ms2_quant[fr].mz = mz
                
                self.map[qe.index].append((run_index, qe))
        
        self.n_entries = len(self.map)
        self.index(ms_files)

    
    def quantify(self, ms_files, out_dir):
        
        logging.info("Quantifying XICs")
        
        ms1_intensity_cols = [f'ms1_intensity_M{i}' for i in range(nIsotops+1)]
        ms2_intensity_cols = [f'ms2_intensity_fr{i}' for i in range(MaxFr)]
        column_names = ['Run', 'Precursor.Id', 'ms1_rt', 'ms2_rt'] + ms1_intensity_cols + ms2_intensity_cols + ['elution']
        
        data_rows = []
        for pr_index, v in progress_bar(self.map.items(), ShowProgress, InterProgress):
            # pr_index = 1
            # v = self.map[pr_index]
            # for each precursor, check overall optimal xic type
            optimal_xic_types = [jt[1].optimal_xic_type for jt in v]
            n_ms1 = optimal_xic_types.count(1)
            n_ms2 = optimal_xic_types.count(2)
            if n_ms2 > n_ms1:
                global_optimal_xic_type = 2
            elif n_ms2 < n_ms1:
                global_optimal_xic_type = 1
            elif n_ms2 == n_ms1:
                cross_run_scores = [jt[1].xic_score.reshape(-1, 1) for jt in v]
                cross_run_scores = np.hstack(cross_run_scores)
                scores_sum = np.sum(cross_run_scores, axis=1)
                global_optimal_xic_type = 2 if np.argmax(scores_sum) >= len(ms1_intensity_cols) else 1
                
            precursor_id = self.lib.entries[pr_index].name
            fwhms = [jt[1].fwhm for jt in v]
            global_fwhm = int(np.ceil(np.mean(fwhms)))
            
            # quantify
            for j in range(self.n_runs):
                # j = 4
                run_name = remove_extension(get_file_name(ms_files[j]))
                jt = v[j]
                ms1 = jt[1].ms1_i
                ms2 = jt[1].ms2
                optimal_xic = int(jt[1].optimal_xic)
                optimal_xic_type = int(jt[1].optimal_xic_type)
                elution_apex = jt[1].apex

                # re-define optimal xic
                if optimal_xic_type != global_optimal_xic_type:
                    apex = ms1.shape[0] // 2
                    (_, _, _, 
                     scores, 
                     optimal_xic, 
                     optimal_xic_type, 
                     elution_apex, fwhm) = identify_optimal_xic(ms1, ms2, 
                                                    apex, 
                                                    radius=CorrRadius, 
                                                    corr_method=CorrMethod, 
                                                    shape_weighted=WeightByShape,
                                                    intensity_weighted=WeightByIntensity,
                                                    optimal_xic_source=f'ms{global_optimal_xic_type}',
                                                    min_shape_score=MinShapeScore,
                                                    min_apex_intensity=MinApexIntensity,
                                                    min_apex_evidence=MinApexEvidence)
                    jt[1].xic_score = scores
                    jt[1].optimal_xic_type = optimal_xic_type
                    jt[1].optimal_xic = optimal_xic
                    jt[1].apex = elution_apex
                    jt[1].fwhm = fwhm
                
                begin = max(0, int(elution_apex - global_fwhm))
                end = min(ms2.shape[0] - 1, int(elution_apex + global_fwhm))
                jt[1].begin = begin
                jt[1].end = end
                
                jt[1].RT = jt[1].ms2_rt[elution_apex]
                jt[1].RT_left = jt[1].ms2_rt[begin]
                jt[1].RT_right = jt[1].ms2_rt[end]
                #
                (ms1_ms2_filtered, 
                 elution, 
                 quant_corr, 
                 quant_quantity) = calculate_quantity(ms1, ms2, 
                                                      begin, end, 
                                                      elution_apex, 
                                                      optimal_xic, 
                                                      optimal_xic_type, 
                                                       corr_method=CorrMethod, 
                                                       int_removal=IntRemoval, 
                                                       int_removal_corr=IntRemovalCorr, 
                                                       filter_factor=IntFilterFactor)
                
                n_ms1 = len(jt[1].ms1_quant)
                for prec in range(n_ms1):
                    jt[1].ms1_quant[prec].corr = quant_corr[prec]
                    jt[1].ms1_quant[prec].quantity = quant_quantity[prec]
                                                      
                for fr in range(len(jt[1].ms2_quant)):                   
                    jt[1].ms2_quant[fr].corr = quant_corr[fr+n_ms1]
                    jt[1].ms2_quant[fr].quantity = quant_quantity[fr+n_ms1]
                
                xics_filtered = np.concat((jt[1].ms1_rt[begin:end+1].reshape(-1, 1), 
                                     jt[1].ms2_rt[begin:end+1].reshape(-1, 1), 
                                     ms1_ms2_filtered, 
                                     elution[begin:end+1].reshape(-1, 1)), axis=1)
          
                data_rows.extend([[run_name, precursor_id] + list(row) for row in xics_filtered])

        # Create Polars DataFrame
        df = pl.DataFrame(data_rows, schema=column_names, orient="row")
        out_xic = Path(out_dir) / "xic_files" / "xics.corrected.parquet"
        df.write_parquet(out_xic)
        logging.info(f"Corrected XIC data saved to {out_xic}\n")