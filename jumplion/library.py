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
from jumplion.arguments import Arguments, get_extension
from jumplion.precursor import (Precursor, to_eg, to_canonical_with_charge, 
                                pep_name, get_aas, get_sequence_masses, 
                                peptide_length, Product, annotate_products_charges)
from jumplion.utils import (CorrMethod, progress_bar, remove_extension, location_to_file_name, 
                            get_file_name, MaxLibFrNum, fTypeB, fTypeY, fExclude, 
                            char_of_fType, name_of_loss, loss_none, loss_H2O, 
                            loss_NH3, loss_CO, loss_other)
from jumplion.elution_peaks import calculate_quantity_with_intf_removal

import pyarrow.parquet as pq

CHUNK_ROWS = 200_000
MS1_XIC_NAMES = [f"M+{i}" for i in range(9)]   # hoisted out of the loop

args = Arguments()
MaxPrecNum = args.max_prec_num
MaxProdNum = args.max_prod_num
IntfRemoval = args.intf_removal
IntfFactorMS1 = args.intf_factor_ms1
IntfFactorMS2 = args.intf_factor_ms2
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
        logging.info(f"Spectral library loaded: {len(self.entries)} precursors in {len(self.co_elution_index)} elution groups\n")
        
        return True
        
    def load_table_format_library(self, file_path):
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
            
            
    # def report(self, file_name: str, ms_files):
    #     try:
    #         # Create a list to store the rows of the report
    #         report_data = []
            
    #         runs = [
    #             remove_extension(get_file_name(ms_file))
    #             for ms_file in ms_files
    #         ]
            
    #         for index, v in self.info.map.items():
    #             entry = self.entries[index]
            
    #             for run_index, qe in v:
            
    #                 row_data = {
    #                     "File.Name": ms_files[run_index],
    #                     "Run": runs[run_index],
    #                     "Precursor.Id": entry.name,
    #                     "Reported": qe.reported,
            
    #                     "Precursor.Charge": entry.prec.charge,
    #                     "Precursor.Mz": entry.prec.mz,
    #                     "Modified.Sequence": pep_name(entry.name),
    #                     "Stripped.Sequence": get_aas(entry.name),
            
    #                     "FWHM": qe.fwhm,
    #                     "RT.Query": qe.initial_RT,
    #                     "RT.Apex": qe.RT,
    #                     "RT.Left": qe.RT_left,
    #                     "RT.Right": qe.RT_right,
            
    #                     "Best.MS1.XIC.Score": qe.best_ms1_xic_score,
    #                     "Best.MS2.XIC.Score": qe.best_ms2_xic_score,
    #                     "Best.MS1.XIC.Index": qe.best_ms1_xic_index,
    #                     "Best.MS2.XIC.Index": qe.best_ms2_xic_index,
            
    #                     "MS1.Internal.Corr": qe.ms1_internal_corr,
    #                     "MS2.Internal.Corr": qe.ms2_internal_corr,
    #                     "MS1.MS2.Cross.Corr": qe.cross_ms1_ms2_corr,
            
    #                     # Reference XIC used for apex/FWHM
    #                     "Reference.XIC": qe.ref_xic,
    #                     "Reference.XIC.Type": (
    #                         f"ms{qe.ref_xic_type}"
    #                         if qe.ref_xic_type > 0
    #                         else "none"
    #                     ),
    #                     "Reference.XIC.Score": qe.ref_xic_score,
    #                     "Reference.XIC.Status": qe.ref_xic_status,
    #                 }
                    
    #                 row_data.update({
    #                     "Apex.Distance": qe.apex_distance,
    #                     "Evidence.Tier": qe.evidence_tier,

    #                     "MS1.Candidate.Apex": qe.ms1_candidate_apex,
    #                     "MS1.FWHM.Left": qe.ms1_fwhm_left,
    #                     "MS1.FWHM.Right": qe.ms1_fwhm_right,

    #                     "MS1.N.Usable.XICs.Region": qe.ms1_n_usable_xics_region,
    #                     "MS2.N.Usable.XICs.Region": qe.ms2_n_usable_xics_region,
    #                     "MS1.N.Usable.XICs.Local": qe.ms1_n_usable_xics_local,
    #                     "MS2.N.Usable.XICs.Local": qe.ms2_n_usable_xics_local,
    #                 })
                    
    #                 # ----------------------------------------------------------
    #                 # Relative isotope / fragment intensities
    #                 # ----------------------------------------------------------
    #                 ms1_rel = np.array(qe.isotope_dist)
            
    #                 ms2_rel = np.array([
    #                     entry.prec.products[fr.index].height
    #                     for fr in qe.ms2_quant
    #                 ])
            
    #                 row_data.update({
    #                     "MS1.Rel.Intensity": ";".join(
    #                         str(x) for x in ms1_rel
    #                     ),
            
    #                     "MS2.Rel.Intensity": ";".join(
    #                         str(x) for x in ms2_rel
    #                     ),
            
    #                     # ------------------------------------------------------
    #                     # Raw XIC quantities / correlations
    #                     # ------------------------------------------------------
    #                     "MS1.XIC.Quant": ";".join(
    #                         str(pr.quantity[0])
    #                         for pr in qe.ms1_quant
    #                     ),
    #                     "MS1.XIC.Quant.Adjusted": ";".join(
    #                         str(pr.quantity[1])
    #                         for pr in qe.ms1_quant
    #                     ),
    #                     "MS1.XIC.Corr": ";".join(
    #                         str(pr.corr[0])
    #                         for pr in qe.ms1_quant
    #                     ),
            
    #                     "MS2.XIC.Quant": ";".join(
    #                         str(fr.quantity[0])
    #                         for fr in qe.ms2_quant
    #                     ),
    #                     "MS2.XIC.Quant.Adjusted": ";".join(
    #                         str(fr.quantity[1])
    #                         for fr in qe.ms2_quant
    #                     ),
    #                     "MS2.XIC.Corr": ";".join(
    #                         str(fr.corr[0])
    #                         for fr in qe.ms2_quant
    #                     ),
            
    #                     # ------------------------------------------------------
    #                     # Adjacent-RT background
    #                     # ------------------------------------------------------
    #                     "MS1.XIC.Quant.Left": ";".join(
    #                         str(q)
    #                         for q in qe.ms1_quant_bg[:, 0].tolist()
    #                     ),
            
    #                     "MS1.XIC.Quant.Right": ";".join(
    #                         str(q)
    #                         for q in qe.ms1_quant_bg[:, 1].tolist()
    #                     ),
            
    #                     "MS2.XIC.Quant.Left": ";".join(
    #                         str(q)
    #                         for q in qe.ms2_quant_bg[:, 0].tolist()
    #                     ),
            
    #                     "MS2.XIC.Quant.Right": ";".join(
    #                         str(q)
    #                         for q in qe.ms2_quant_bg[:, 1].tolist()
    #                     ),
    #                 })
            
    #                 # ----------------------------------------------------------
    #                 # XIC information
    #                 # ----------------------------------------------------------
    #                 ms1_xic_names = [
    #                     f"M+{i}"
    #                     for i in range(9)
    #                 ]
            
    #                 row_data.update({
    #                     "MS1.XIC.Info": ";".join(
    #                         f"{ms1_xic_names[i]}/{np.round(pr.mz, 4)}"
    #                         for i, pr in enumerate(qe.ms1_quant)
    #                         if i < len(ms1_xic_names)
    #                     ),
            
    #                     "MS2.XIC.Info": ";".join([
    #                         f"{char_of_fType[
    #                             (cfr.ion_type & (fTypeB | fTypeY)) >> 0
    #                         ]}"
    #                         f"{cfr.index if cfr.ion_type & fTypeB else len(get_aas(entry.name)) - cfr.index}"
    #                         f"{'-' + name_of_loss[cfr.loss] if cfr.loss != loss_none else ''}"
    #                         f"^{cfr.charge}/{np.round(cfr.mz, 4)}"
    #                         for fr in qe.ms2_quant
    #                         for cfr in [entry.prec.products[fr.index]]
    #                     ]),
    #                 })
            
    #                 report_data.append(row_data)
                        
    #         report_df = pl.DataFrame(
    #             report_data,
    #             schema_overrides={
    #                 "MS1.Candidate.Apex": pl.Int64,
    #                 "MS1.FWHM.Left": pl.Int64,
    #                 "MS1.FWHM.Right": pl.Int64,
    #             },
    #             infer_schema_length=None,
    #         )
            
    #         # Save
    #         report_df.write_parquet(file_name)
            
    #         logging.info(
    #             f"XIC quantification report saved to {file_name}\n"
    #         )
            
    #     except IOError as e:
    #         logging.error(f"ERROR: cannot write to {file_name}. Check if the destination folder is write-protected or the file is in use: {e}")
    #         return
    
    
    def report(self, file_name: str, ms_files):
        writer = None
    
        def flush(rows):
            nonlocal writer
            if not rows:
                return
            table = pl.DataFrame(
                rows,
                schema_overrides={
                    "MS1.Candidate.Apex": pl.Int64,
                    "MS1.FWHM.Left": pl.Int64,
                    "MS1.FWHM.Right": pl.Int64,
                },
                infer_schema_length=None,
            ).to_arrow(compat_level=pl.CompatLevel.oldest())
    
            if writer is None:
                # first chunk fixes the file schema
                writer = pq.ParquetWriter(file_name, table.schema, compression="zstd")
            else:
                # later chunks must match it; raises loudly on a real type conflict
                table = table.cast(writer.schema)
            writer.write_table(table)
    
        try:
            report_data = []
            runs = [remove_extension(get_file_name(f)) for f in ms_files]
    
            for index, v in self.info.map.items():
                entry = self.entries[index]
    
                for run_index, qe in v:
                    row_data = {
                        "File.Name": ms_files[run_index],
                        "Run": runs[run_index],
                        "Precursor.Id": entry.name,
                        "Reported": qe.reported,
            
                        "Precursor.Charge": entry.prec.charge,
                        "Precursor.Mz": entry.prec.mz,
                        "Modified.Sequence": pep_name(entry.name),
                        "Stripped.Sequence": get_aas(entry.name),
            
                        "FWHM": qe.fwhm,
                        "RT.Query": qe.initial_RT,
                        "RT.Apex": qe.RT,
                        "RT.Left": qe.RT_left,
                        "RT.Right": qe.RT_right,
            
                        "Best.MS1.XIC.Score": qe.best_ms1_xic_score,
                        "Best.MS2.XIC.Score": qe.best_ms2_xic_score,
                        "Best.MS1.XIC.Index": qe.best_ms1_xic_index,
                        "Best.MS2.XIC.Index": qe.best_ms2_xic_index,
            
                        "MS1.Internal.Corr": qe.ms1_internal_corr,
                        "MS2.Internal.Corr": qe.ms2_internal_corr,
                        "MS1.MS2.Cross.Corr": qe.cross_ms1_ms2_corr,
            
                        # Reference XIC used for apex/FWHM
                        "Reference.XIC": qe.ref_xic,
                        "Reference.XIC.Type": (
                            f"ms{qe.ref_xic_type}"
                            if qe.ref_xic_type > 0
                            else "none"
                        ),
                        "Reference.XIC.Score": qe.ref_xic_score,
                        "Reference.XIC.Status": qe.ref_xic_status,
                    }
                    
                    row_data.update({
                        "Apex.Distance": qe.apex_distance,
                        "Evidence.Tier": qe.evidence_tier,

                        "MS1.Candidate.Apex": qe.ms1_candidate_apex,
                        "MS1.FWHM.Left": qe.ms1_fwhm_left,
                        "MS1.FWHM.Right": qe.ms1_fwhm_right,

                        "MS1.N.Usable.XICs.Region": qe.ms1_n_usable_xics_region,
                        "MS2.N.Usable.XICs.Region": qe.ms2_n_usable_xics_region,
                        "MS1.N.Usable.XICs.Local": qe.ms1_n_usable_xics_local,
                        "MS2.N.Usable.XICs.Local": qe.ms2_n_usable_xics_local,
                    })
                    
                    # ----------------------------------------------------------
                    # Relative isotope / fragment intensities
                    # ----------------------------------------------------------
                    ms1_rel = np.array(qe.isotope_dist)
            
                    ms2_rel = np.array([
                        entry.prec.products[fr.index].height
                        for fr in qe.ms2_quant
                    ])
            
                    row_data.update({
                        "MS1.Rel.Intensity": ";".join(
                            str(x) for x in ms1_rel
                        ),
            
                        "MS2.Rel.Intensity": ";".join(
                            str(x) for x in ms2_rel
                        ),
            
                        # ------------------------------------------------------
                        # Raw XIC quantities / correlations
                        # ------------------------------------------------------
                        "MS1.XIC.Quant": ";".join(
                            str(pr.quantity[0])
                            for pr in qe.ms1_quant
                        ),
                        "MS1.XIC.Quant.Adjusted": ";".join(
                            str(pr.quantity[1])
                            for pr in qe.ms1_quant
                        ),
                        "MS1.XIC.Corr": ";".join(
                            str(pr.corr[0])
                            for pr in qe.ms1_quant
                        ),
            
                        "MS2.XIC.Quant": ";".join(
                            str(fr.quantity[0])
                            for fr in qe.ms2_quant
                        ),
                        "MS2.XIC.Quant.Adjusted": ";".join(
                            str(fr.quantity[1])
                            for fr in qe.ms2_quant
                        ),
                        "MS2.XIC.Corr": ";".join(
                            str(fr.corr[0])
                            for fr in qe.ms2_quant
                        ),
            
                        # ------------------------------------------------------
                        # Adjacent-RT background
                        # ------------------------------------------------------
                        "MS1.XIC.Quant.Left": ";".join(
                            str(q)
                            for q in qe.ms1_quant_bg[:, 0].tolist()
                        ),
            
                        "MS1.XIC.Quant.Right": ";".join(
                            str(q)
                            for q in qe.ms1_quant_bg[:, 1].tolist()
                        ),
            
                        "MS2.XIC.Quant.Left": ";".join(
                            str(q)
                            for q in qe.ms2_quant_bg[:, 0].tolist()
                        ),
            
                        "MS2.XIC.Quant.Right": ";".join(
                            str(q)
                            for q in qe.ms2_quant_bg[:, 1].tolist()
                        ),
                    })
            
                    # ----------------------------------------------------------
                    # XIC information
                    # ----------------------------------------------------------            
                    row_data.update({
                        "MS1.XIC.Info": ";".join(
                            f"{MS1_XIC_NAMES[i]}/{np.round(pr.mz, 4)}"
                            for i, pr in enumerate(qe.ms1_quant)
                            if i < len(MS1_XIC_NAMES )
                        ),
            
                        "MS2.XIC.Info": ";".join([
                            f"{char_of_fType[
                                (cfr.ion_type & (fTypeB | fTypeY)) >> 0
                            ]}"
                            f"{cfr.index if cfr.ion_type & fTypeB else len(get_aas(entry.name)) - cfr.index}"
                            f"{'-' + name_of_loss[cfr.loss] if cfr.loss != loss_none else ''}"
                            f"^{cfr.charge}/{np.round(cfr.mz, 4)}"
                            for fr in qe.ms2_quant
                            for cfr in [entry.prec.products[fr.index]]
                        ]),
                    })
            
                    report_data.append(row_data)
                    if len(report_data) >= CHUNK_ROWS:
                        flush(report_data)
                        report_data = []          # drop the dicts we've written
    
            flush(report_data)
            logging.info(f"XIC quantification report saved to {file_name}\n")
    
        except IOError as e:
            logging.error(f"ERROR: cannot write to {file_name}. Check if the destination folder "
                          f"is write-protected or the file is in use: {e}")
        finally:
            if writer is not None:
                writer.close()

# @dataclass
# class QuantEntry:
#     reported: int = 0
#     index: int = 0
#     run_index: int = 0

#     n_isotopes: int = 0
#     n_prods: int = 0
    
#     isotope_dist = None
    
#     initial_RT: float = 0.0
#     initial_apex: int = 0
#     apex: int = 0
#     fwhm: int = 0

#     xic_score = None
#     corr_score = None
#     shape_score = None
#     intensity_score = None

#     ref_xic: int = 0
#     ref_xic_type: int = 0
#     RT: float = 0.0
#     RT_left: float = 0.0
#     RT_right: float = 0.0
#     IM: float = 0.0
#     ms1 = None
#     ms2 = None
#     ms1_i = None
#     ms1_rt = None
#     ms2_rt = None
#     ms1_quant: List[XICQuant] = field(default_factory=list)   # built to real length, not preallocated
#     ms2_quant: List[XICQuant] = field(default_factory=list)
#     ms1_quant_bg = None
#     ms2_quant_bg = None
        

# @dataclass
# class XICQuant:
#     index: int = 0
#     mz: float = 0.0
#     quantity: float = 0.0
#     corr: float = 0.0

@dataclass
class XICQuant:
    index: int = 0  # Index of the product ion in the library
    mz: float = 0.0
    quantity: List[float] = field(default_factory=lambda: [0.0] * 2)  
    corr: List[float] = field(default_factory=lambda: [0.0] * 2)


@dataclass
class QuantEntry:
    reported: int = 0
    index: int = 0
    run_index: int = 0

    n_isotopes: int = 0
    n_prods: int = 0

    isotope_dist: object = None

    initial_RT: float = 0.0
    initial_apex: int = 0
    apex: int = 0
    fwhm: int = 0

    # ------------------------------------------------------------
    # Per-XIC candidate scores
    # ------------------------------------------------------------
    xic_score = None
    corr_score = None
    shape_score = None
    intensity_score = None

    # ------------------------------------------------------------
    # Joint MS1/MS2 strategy scores
    # ------------------------------------------------------------
    best_ms1_xic_score: float = np.nan
    best_ms2_xic_score: float = np.nan
    best_ms1_xic_index: int = -1
    best_ms2_xic_index: int = -1
    cross_ms1_ms2_corr: float = np.nan
    ms1_internal_corr: float = np.nan
    ms2_internal_corr: float = np.nan

    # ------------------------------------------------------------
    # Final XIC used for apex/FWHM
    # ------------------------------------------------------------
    ref_xic: int = -1
    ref_xic_type: int = 0
    ref_xic_score: float = np.nan
    ref_xic_status: str = "not_found"
    
    apex_distance: int = -1
    apex_source: str = ""
    evidence_tier: str = ""

    ms1_candidate_apex: Optional[int] = None
    ms1_fwhm_left: Optional[int] = None
    ms1_fwhm_right: Optional[int] = None

    ms1_n_usable_xics_region: int = 0
    ms2_n_usable_xics_region: int = 0
    ms1_n_usable_xics_local: int = 0
    ms2_n_usable_xics_local: int = 0
    
    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------
    best_ms2_xic: int = -1
    best_ms2_mz: float = np.nan
    mass_delta_ms2: float = 0.0
    mass_delta_ms1: float = 0.0

    im_delta: float = 0.0

    # ------------------------------------------------------------
    # Quantification data
    # ------------------------------------------------------------
    RT: float = 0.0
    RT_left: float = 0.0
    RT_right: float = 0.0
    IM: float = 0.0

    ms1 = None
    ms2 = None
    ms1_i = None
    ms1_rt = None
    ms2_rt = None

    ms1_quant: List[XICQuant] = field(default_factory=list)
    ms2_quant: List[XICQuant] = field(default_factory=list)

    ms1_quant_bg = None
    ms2_quant_bg = None


def _make_quant_entry(
    reported,
    index,
    run_index,
    isotope_dist,
    initial_RT,
    initial_apex,
    apex,
    fwhm,

    # Final XIC
    ref_xic,
    ref_xic_type,
    ref_xic_score,
    ref_xic_status,
    
    apex_distance,
    apex_source,
    evidence_tier,
    
    ms1_candidate_apex,
    ms1_fwhm_left,
    ms1_fwhm_right,
    
    ms1_n_usable_xics_region,
    ms2_n_usable_xics_region,
    ms1_n_usable_xics_local,
    ms2_n_usable_xics_local,
    
    # Per-XIC scores
    combined_score,
    corr_score,
    shape_score,
    intensity_score,

    # Strategy scores
    best_ms1_xic_score,
    best_ms2_xic_score,
    best_ms1_xic_index,
    best_ms2_xic_index,
    cross_ms1_ms2_corr,
    ms1_internal_corr,
    ms2_internal_corr,

    # Ion/XIC data
    ms1_mz,
    ms2_mz,
    ms2_indices,
    ms1_rt,
    ms2_rt,
    ms1,
    ms2,
    ms1_i,
):
    qe = QuantEntry()

    qe.reported = int(reported)
    qe.index = int(index)
    qe.run_index = int(run_index)

    qe.n_isotopes = len(ms1_mz)
    qe.n_prods = len(ms2_mz)

    qe.isotope_dist = np.asarray(isotope_dist)

    qe.initial_RT = float(initial_RT)
    qe.initial_apex = int(initial_apex)
    qe.apex = int(apex) if np.isfinite(apex) else -1
    qe.fwhm = int(fwhm) if np.isfinite(fwhm) else -1

    # ------------------------------------------------------------
    # Per-XIC arrays
    # ------------------------------------------------------------
    qe.xic_score = np.asarray(combined_score)
    qe.corr_score = np.asarray(corr_score)
    qe.shape_score = np.asarray(shape_score)
    qe.intensity_score = np.asarray(intensity_score)

    # ------------------------------------------------------------
    # Strategy-level scalar metadata
    # ------------------------------------------------------------
    qe.best_ms1_xic_score = float(best_ms1_xic_score)
    qe.best_ms2_xic_score = float(best_ms2_xic_score)
    qe.best_ms1_xic_index = int(best_ms1_xic_index)
    qe.best_ms2_xic_index = int(best_ms2_xic_index)
    qe.cross_ms1_ms2_corr = float(cross_ms1_ms2_corr)
    qe.ms1_internal_corr = float(ms1_internal_corr)
    qe.ms2_internal_corr = float(ms2_internal_corr)

    # Best candidate before final shape validation
    qe.ref_xic_score = float(ref_xic_score)
    qe.ref_xic_status = str(ref_xic_status)
    qe.ref_xic = int(ref_xic)
    qe.ref_xic_type = int(ref_xic_type)
    
    qe.apex_distance = int(apex_distance)
    qe.apex_source = str(apex_source)
    qe.evidence_tier = str(evidence_tier)
    
    qe.ms1_candidate_apex = (
        int(ms1_candidate_apex) if ms1_candidate_apex is not None else None
    )
    qe.ms1_fwhm_left = int(ms1_fwhm_left) if ms1_fwhm_left is not None else None
    qe.ms1_fwhm_right = int(ms1_fwhm_right) if ms1_fwhm_right is not None else None

    qe.ms1_n_usable_xics_region = int(ms1_n_usable_xics_region)
    qe.ms2_n_usable_xics_region = int(ms2_n_usable_xics_region)
    qe.ms1_n_usable_xics_local = int(ms1_n_usable_xics_local)
    qe.ms2_n_usable_xics_local = int(ms2_n_usable_xics_local)
    
    # ------------------------------------------------------------
    # XIC traces
    # ------------------------------------------------------------
    qe.ms1_rt = np.asarray(ms1_rt)
    qe.ms2_rt = np.asarray(ms2_rt)
    qe.ms1 = np.asarray(ms1)
    qe.ms2 = np.asarray(ms2)
    qe.ms1_i = np.asarray(ms1_i)

    if (
        qe.apex >= 0
        and qe.apex < len(qe.ms2_rt)
    ):
        qe.RT = float(qe.ms2_rt[qe.apex])
    else:
        qe.RT = float(initial_RT)

    qe.IM = 0.0

    qe.ms1_quant = [
        XICQuant(index=i, mz=float(mz))
        for i, mz in enumerate(ms1_mz)
    ]

    qe.ms2_quant = [
        XICQuant(index=int(idx), mz=float(mz))
        for idx, mz in zip(ms2_indices, ms2_mz)
    ]

    qe.ms1_quant_bg = np.ones((len(ms1_mz), 2))
    qe.ms2_quant_bg = np.ones((len(ms2_mz), 2))

    return qe



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

    def accumulate_from_xic_list(self, xic_list, run_index: int):
        for xg in xic_list:
            info_ = xg.info
            qe = _make_quant_entry(
                reported=xg.reported,
                index=xg.pr_index,
                run_index=run_index,
            
                isotope_dist=info_.isotope_dist,
            
                initial_RT=info_.initial_RT,
                initial_apex=info_.initial_apex,
                apex=info_.apex,
                fwhm=info_.fwhm,
            
                ref_xic=info_.ref_xic,
                ref_xic_type=info_.ref_xic_type,
                ref_xic_score=info_.ref_xic_score,
                ref_xic_status=info_.ref_xic_status,
                
                apex_distance=info_.apex_distance,
                apex_source=info_.apex_source,
                evidence_tier=info_.evidence_tier,

                ms1_candidate_apex=info_.ms1_candidate_apex,
                ms1_fwhm_left=info_.ms1_fwhm_left,
                ms1_fwhm_right=info_.ms1_fwhm_right,

                ms1_n_usable_xics_region=info_.ms1_n_usable_xics_region,
                ms2_n_usable_xics_region=info_.ms2_n_usable_xics_region,
                ms1_n_usable_xics_local=info_.ms1_n_usable_xics_local,
                ms2_n_usable_xics_local=info_.ms2_n_usable_xics_local,
                
                # Per-XIC arrays
                combined_score=info_.scores["combined_score"],
                corr_score=info_.scores["corr_score"],
                shape_score=info_.scores["shape_score"],
                intensity_score=info_.scores["intensity_score"],
            
                # Strategy
                best_ms1_xic_score=info_.best_ms1_xic_score,
                best_ms2_xic_score=info_.best_ms2_xic_score,
                best_ms1_xic_index=info_.best_ms1_xic_index,
                best_ms2_xic_index=info_.best_ms2_xic_index,
                cross_ms1_ms2_corr=info_.cross_ms1_ms2_corr,
                ms1_internal_corr=info_.ms1_internal_corr,
                ms2_internal_corr=info_.ms2_internal_corr,

                # XIC arrays
                ms1_mz=info_.ms1_mzs,
                ms2_mz=[fr.mz for fr in info_.products],
                ms2_indices=info_.prodi,
            
                ms1_rt=xg.xics[:, 0],
                ms2_rt=xg.xics[:, 1],
                ms1=xg.xics[
                    :,
                    2:2 + info_.n_isotopes
                ],
                ms2=xg.xics[
                    :,
                    2 + info_.n_isotopes:
                    2 + info_.n_isotopes + info_.n_prods
                ],
                ms1_i=xg.xics[
                    :,
                    2 + info_.n_isotopes + info_.n_prods:
                ],
            )
            self.map[qe.index].append((run_index, qe))


    def load(self, parent, ms_files, out_dir):
        self.n_runs = len(ms_files)
        self.lib = parent
        logging.info(f"Assembling XIC data of {self.n_runs} runs")
        lib_n = len(self.lib.entries)
        self.clear()
    
        for run_index, file in enumerate(ms_files):
            run_name = remove_extension(get_file_name(file))
            xic_file = Path(out_dir) / "xic_files" / f"{run_name}.xic.parquet"
            logging.info(f"Loading XIC data from {xic_file}")
            try:
                df = pl.read_parquet(xic_file)
            except FileNotFoundError:
                alt_path = location_to_file_name(file) + ".xic.parquet"
                df = pl.read_parquet(alt_path)
            self._load_xic_dataframe(df, run_index, lib_n)
    
        self.n_entries = len(self.map)
        self.index(ms_files)
    
    
    def _load_xic_dataframe(self, df: pl.DataFrame, run_index: int, lib_n: int):
        # precursor_count = df["Precursor.Id"].n_unique()
        # if precursor_count != lib_n:
        #     logging.warning("Number of precursors doesn't match between library and XIC parquet file.")
    
        loaded = 0
        for row in df.iter_rows(named=True):
            qe = _make_quant_entry(
                reported=row["reported"],
                index=row["prec_index"],
                run_index=run_index,
            
                isotope_dist=row["isotope_dist"],
            
                initial_RT=row["initial_RT"],
                initial_apex=row["initial_apex"],
                apex=row["apex"],
                fwhm=row["fwhm"],
            
                ref_xic=row["ref_xic"],
                ref_xic_type=row["ref_xic_type"],
                ref_xic_score=row["ref_xic_score"],
                ref_xic_status=row["ref_xic_status"],
                
                apex_distance=row["apex_distance"],
                apex_source=row["apex_source"],
                evidence_tier=row["evidence_tier"],

                ms1_candidate_apex=row["ms1_candidate_apex"],
                ms1_fwhm_left=row["ms1_fwhm_left"],
                ms1_fwhm_right=row["ms1_fwhm_right"],

                ms1_n_usable_xics_region=row["ms1_n_usable_xics_region"],
                ms2_n_usable_xics_region=row["ms2_n_usable_xics_region"],
                ms1_n_usable_xics_local=row["ms1_n_usable_xics_local"],
                ms2_n_usable_xics_local=row["ms2_n_usable_xics_local"],
                
                combined_score=row["combined_score"],
                corr_score=row["corr_score"],
                shape_score=row["shape_score"],
                intensity_score=row["intensity_score"],
            
                best_ms1_xic_score=row["best_ms1_xic_score"],
                best_ms2_xic_score=row["best_ms2_xic_score"],
                best_ms1_xic_index=row["best_ms1_xic_index"],
                best_ms2_xic_index=row["best_ms2_xic_index"],
                cross_ms1_ms2_corr=row["cross_ms1_ms2_corr"],
                ms1_internal_corr=row["ms1_internal_corr"],
                ms2_internal_corr=row["ms2_internal_corr"],
            
                ms1_mz=row["ms1_mz"],
                ms2_mz=row["ms2_mz"],
                ms2_indices=row["ms2_indices"],
            
                ms1_rt=row["ms1_rt"],
                ms2_rt=row["ms2_rt"],
                ms1=row["ms1"],
                ms2=row["ms2"],
                ms1_i=row["ms1_i"],
            )
            self.map[qe.index].append((run_index, qe))
            loaded += 1
    
        if ShowProgress and loaded:
            logging.info(f"Assembled {loaded} precursor XIC groups")

    
    def quantify(self, ms_files, out_dir):
        logging.info("Quantifying XICs")
        run_names = [remove_extension(get_file_name(file)) for file in ms_files]
        
        out_runs, out_precursors = [], []
        out_n_isotopes, out_n_prods = [], []
        out_begin, out_end = [], []
        out_ms1_rt, out_ms2_rt = [], []
        out_ms1_ms2_filtered = [] 
        out_ms1_elution, out_ms2_elution = [], []
    
        for pr_index, v in progress_bar(self.map.items(), ShowProgress, InterProgress):
            entries = [qt[1] for qt in v]
    
            fwhms = [entry.fwhm for entry in entries if entry.fwhm != 0]
            global_fwhm = int(np.ceil(np.median(fwhms))) if fwhms else 0
            
            # apex_dists = []
            # for entry in entries:
            #     if not entry.reported:
            #         apex_dists.append(entry.apex_distance)

            # q1, q3 = np.percentile(apex_dists, [25, 75])
            # iqr = q3 - q1
            # #lower_bound = q1 - (1.5 * iqr)
            # upper_bound = q3 + (1.5 * iqr)
            # for entry in entries:
            #     if not entry.reported:
            #         entry.rt_outlier = (entry.apex_distance > upper_bound) #| (entry.apex_distance < lower_bound)
            #     else:
            #         entry.rt_outlier = 0
            
            precursor_id = self.lib.entries[pr_index].name
            for j in range(self.n_runs):
                run_name = run_names[j]
                entry = entries[j]
                # reported = entry.reported
                ms1 = entry.ms1_i
                ms2 = entry.ms2
                best_ms1_xic = int(entry.best_ms1_xic_index)
                best_ms2_xic = int(entry.best_ms2_xic_index)
                # initial_apex = entry.initial_apex
                apex = entry.apex
                # apex_tolerance = int(apex_tolerance_runs[run_name])
    
                begin = max(0, int(apex - global_fwhm))
                end = min(ms2.shape[0] - 1, int(apex + global_fwhm))
                entry.begin, entry.end = begin, end
                entry.RT = entry.ms2_rt[apex]
                entry.RT_left = entry.ms2_rt[begin]
                entry.RT_right = entry.ms2_rt[end]
                
                # target region quantity
                # (elution, 
                #  quant_corr, 
                #  quant_quantity) = calculate_quantity(
                #     ms1, ms2, 
                #     begin, end, apex, 
                #     ref_xic, 
                #     ref_xic_type,
                #     corr_method=CorrMethod, 
                # )
                
                (ms1_ms2_filtered, 
                 ms1_elution, 
                 ms2_elution,
                 quant_corr, 
                 quant_quantity) = calculate_quantity_with_intf_removal(
                     ms1, ms2, 
                     begin, end, apex, 
                     best_ms1_xic,
                     best_ms2_xic,
                     corr_method=CorrMethod, 
                     intf_removal=IntfRemoval, 
                     intf_factor_ms1=IntfFactorMS1,
                     intf_factor_ms2=IntfFactorMS2)
                
                
                n_isotopes = entry.n_isotopes
                n_prods = entry.n_prods
                for prec in range(n_isotopes):
                    entry.ms1_quant[prec].corr = quant_corr[prec]
                    entry.ms1_quant[prec].quantity = quant_quantity[prec]
                for fr in range(n_prods):
                    entry.ms2_quant[fr].corr = quant_corr[fr + n_isotopes]
                    entry.ms2_quant[fr].quantity = quant_quantity[fr + n_isotopes]
                
                len_ = end - begin + 1
                left_begin = begin - len_
                left_end = begin - 1
                right_begin = end + 1
                right_end = end + len_
                entry.ms1_quant_bg[:, 0] = np.sum(ms1[left_begin:left_end+1, :], axis=0) # left
                entry.ms1_quant_bg[:, 1] = np.sum(ms1[right_begin:right_end+1, :], axis=0) # right
                entry.ms2_quant_bg[:, 0] = np.sum(ms2[left_begin:left_end+1, :], axis=0) # left
                entry.ms2_quant_bg[:, 1] = np.sum(ms2[right_begin:right_end+1, :], axis=0) # right
                
                
                
                out_runs.append(run_name)
                out_precursors.append(precursor_id)
                out_n_isotopes.append(n_isotopes)
                out_n_prods.append(n_prods)
                out_begin.append(begin)
                out_end.append(end)
                out_ms1_rt.append(entry.ms1_rt[begin:end+1].tolist())
                out_ms2_rt.append(entry.ms2_rt[begin:end+1].tolist())
                out_ms1_ms2_filtered.append(ms1_ms2_filtered.tolist())   # shape (T, n_precs+n_prods)
                out_ms1_elution.append(ms1_elution[begin:end+1].tolist())
                out_ms2_elution.append(ms2_elution[begin:end+1].tolist())
    
        df = pl.DataFrame({
            "Run": out_runs,
            "Precursor.Id": out_precursors,
            "n_isotopes": out_n_isotopes,
            "n_prods": out_n_prods,
            "begin": out_begin,
            "end": out_end,
            "ms1_rt": out_ms1_rt,
            "ms2_rt": out_ms2_rt,
            "ms1_ms2_filtered": out_ms1_ms2_filtered,
            "ms1_elution": out_ms1_elution,
            "ms2_elution": out_ms2_elution,
        })
    
        out_xic = Path(out_dir) / "xic_files" / "xics.quant.parquet"
        out_xic.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_xic)
        logging.info(f"Quantitative XIC data saved to {out_xic}")
