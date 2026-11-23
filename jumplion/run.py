# builtin
import bisect
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
# external
import numpy as np
import pandas as pd
import polars as pl
# local
from jumplion.arguments import Arguments, get_extension
from jumplion.utils import (InitialRadius, CorrMethod, MinMs1Ms2Corr,
                            MinXicIntensity, MinApexIntensity, 
                            MassCalBins, MassCalBinsMax, MassCalScore, C13delta, 
                            fExclude, E, INF, estimate_isotopes, consolidate_by_ms1, #weighted_average_mz,
                            remove_extension, get_file_name, progress_bar)
from jumplion.read_ms_files import load_thermo_raw, load_mzml, load_timsTOF_d, get_cycle_time
from jumplion.calibration import (split_into_rt_bins, 
                                  predicted_mz, 
                                  predicted_mz_parallel, 
                                  build_mass_cali_model, 
                                  mass_calibration_plot, 
                                  im_error_plot)
from jumplion.precursor import Product
from jumplion.elution_peaks import interpolate_ms1, evaluate_xics

args = Arguments()
OutDir = args.out_dir
out_int_dir = Path(OutDir) / 'intermediate'
SaveSupplements = args.save_supplements
# XIC extraction
MinXICPointsHalf = args.min_xic_points_half
Calibrate = args.mass_calibration
MassCalAcc = args.mass_cal_acc
IndividualMassAcc = args.individual_mass_acc
IMWindowCal = args.im_window_cal
IndividualIMWindow = args.individual_im_window
MaxPrecNum = args.max_prec_num
MinProdNum = args.min_prod_num
MaxProdNum = args.max_prod_num
# XIC quantification
RefXicSource = args.optimal_xic_source
CorrWeight = args.corr_weight 
ShapeWeight = args.shape_weight
IntensityWeight = args.intensity_weight


@dataclass
class XICGroupInfo:
    # MS1 XIC m/zs
    ms1_mzs: List[float] = field(default_factory=list)
    isotope_dist: List[float] = field(default_factory=list)
    n_isotopes: int = 0

    # MS2 ion info
    products: List[Product] = field(default_factory=list)
    prodi: List[int] = field(default_factory=list)
    n_prods: int = 0

    # XIC apex and peak width
    initial_RT: float = 0.0
    initial_apex: int = 0
    apex: int = 0
    fwhm: int = 0

    # Per-XIC scores
    scores: Dict[str, List[float]] = field(default_factory=lambda: {
        "corr_score": [],
        "shape_score": [],
        "intensity_score": [],
        "combined_score": [],
    })

    # ------------------------------------------------------------------
    # XIC selection / strategy state
    # ------------------------------------------------------------------
    # Final XIC used for apex/FWHM
    ref_xic: int = -1
    ref_xic_type: int = 0  # 1 = MS1, 2 = MS2, 0 = none
    ref_xic_score: float = np.nan
    ref_xic_status: str = "not_found"
    
    apex_distance: int = -1
    apex_source: str = ""
    evidence_tier: str = ''

    ms1_candidate_apex: Optional[int] = None
    ms1_fwhm_left: Optional[int] = None
    ms1_fwhm_right: Optional[int] = None

    ms1_n_usable_xics_region: int = 0
    ms2_n_usable_xics_region: int = 0
    ms1_n_usable_xics_local: int = 0
    ms2_n_usable_xics_local: int = 0
    
    mono_ms1_xic_score: float = np.nan
    best_ms1_xic_score: float = np.nan
    best_ms2_xic_score: float = np.nan
    best_ms1_xic_index: int = -1
    best_ms2_xic_index: int = -1

    # Joint MS1/MS2 coherence
    cross_ms1_ms2_corr: float = np.nan
    ms1_internal_corr: float = np.nan
    ms2_internal_corr: float = np.nan

    # For mass calibration
    #best_ms2_xic: int = -1
    best_ms2_xic_mz: float = np.nan
    mass_delta_ms2: float = 0.0
    mass_delta_ms1: float = 0.0

    im_delta: float = 0.0

    def clear(self):
        self.initial_apex = 0
        self.apex = 0
        self.fwhm = 0

        self.scores = {
            "corr_score": [],
            "shape_score": [],
            "intensity_score": [],
            "combined_score": [],
        }

        # XIC selection / strategy state
        self.ref_xic_status = "not_found"
        self.ref_xic = -1
        self.ref_xic_type = 0
        self.ref_xic_score = np.nan
        
        self.apex_distance = -1
        self.apex_source = ""
        self.evidence_tier = ''

        self.ms1_candidate_apex = None
        self.ms1_fwhm_left = None
        self.ms1_fwhm_right = None

        self.ms1_n_usable_xics_region = 0
        self.ms2_n_usable_xics_region = 0
        self.ms1_n_usable_xics_local = 0
        self.ms2_n_usable_xics_local = 0
        
        self.mono_ms1_xic_score = np.nan
        self.best_ms1_xic_score = np.nan
        self.best_ms2_xic_score = np.nan
        self.best_ms1_xic_index = -1
        self.best_ms2_xic_index = -1

        self.cross_ms1_ms2_corr = np.nan
        self.ms1_internal_corr = np.nan
        self.ms2_internal_corr = np.nan

        # Calibration
        #self.best_ms2_xic = -1
        self.best_ms2_xic_mz = np.nan
        self.mass_delta_ms2 = 0.0
        self.mass_delta_ms1 = 0.0
        self.im_delta = 0.0
        

@dataclass
class XICGroup:
    run: int = -1          # run index
    pr_index: int = -1     # precursor index in the library
    pr_id: str = ""        # precursor id
    reported: int = 0
    
    info: XICGroupInfo = field(default_factory=XICGroupInfo)  # XIC info
    xics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # RTs and intensities

    
class RunEntry:
    def __init__(self):
        self.run = None      # Run class
        self.name = ""       # precursor id
        self.prec = None     # 'Precursor' object
        self.mass_cal = 0    # whether it will be used for mass calibration
        self.re_quan = 0     # whether it will be subjected to re-quantification
        self.reported = 0    # whether it is reported by DIA-NN (run-specific)
        
        # RT and IM (ion mobility)
        self.RT = 0.0 
        self.IM = 0.0
        
        self.fInit = 0       # binary 000
        self.ms2_indices_window = np.zeros(1)  # ms2 scan indices corresponding to the precursor
        self.info = XICGroupInfo()
        
        self.ms1_rts = None
        self.ms2_rts = None
        
        self.ms1 = None
        self.ms1_i = None
        self.ms2 = None
        

    def init(self, run, prec):
        self.run = run           # Run class
        self.prec = prec         # Precursor class
    
    def prepare_products(self, min_prod, max_prod):
        # get products info
        for prodi, prod in enumerate(self.prec.products):
            if not (prod.ion_type & fExclude):
                # check product ion m/z
                if self.run.min_ms2_mz < prod.mz < self.run.max_ms2_mz:
                    self.info.products.append(prod)
                    self.info.prodi.append(prodi)
            # exit if there is enough product ion
            if len(self.info.products) == max_prod:
                break
        
        self.info.n_prods = len(self.info.products)
        if self.info.n_prods < min_prod:  # default: at least 3 products
            self.info.products.clear()
            self.info.prodi.clear()
            self.info.n_prods = 0
    
    def build_index(self):
        # the number of MS2 acquisition window corresponding to the precursor
        window_index = self.run.get_window_index(self.prec.mz)
        # all MS2 scan indices corresponding to the window
        ms2_indices_window = (np.arange(self.run.n_cycle) * self.run.n_ms2_per_cycle) + window_index
        # make sure the generated index do not exceed total scan number
        self.ms2_indices_window = ms2_indices_window[ms2_indices_window < len(self.run.ms2sp)]
    
    def extract_xic(self, from_idx: int, to_idx: int, report_mz_im: int):
        l = to_idx - from_idx

        indices_i = self.ms2_indices_window[from_idx:to_idx]
        ms2_rts   = self.run.ms2_RT[indices_i]

        # Fallback: single MS1 scan immediately left of each MS2 (used when no
        # MS1 scans fall inside the cycle interval).
        ms1_lefts = np.clip(
            np.searchsorted(self.run.ms1_RT, ms2_rts) - 1,
            0, len(self.run.ms1_RT) - 1
        )

        # Cycle boundaries: all MS1 scans in (prev_ms2_rt, this_ms2_rt] belong
        # to the same DIA cycle as this MS2 scan.  For Astral data this captures
        # all 3+ MS1 scans per cycle; for regular DIA it reduces to the single
        # scan to the left (identical to the previous behaviour).
        gap = (ms2_rts[1] - ms2_rts[0]) if l > 1 else ms2_rts[0]
        prev_ms2_rts = np.empty(l, dtype=ms2_rts.dtype)
        prev_ms2_rts[0]  = ms2_rts[0] - gap
        prev_ms2_rts[1:] = ms2_rts[:-1]

        ms1_cycle_starts = np.searchsorted(self.run.ms1_RT, prev_ms2_rts, side='right')
        ms1_cycle_ends   = np.searchsorted(self.run.ms1_RT, ms2_rts,      side='right')

        # Representative MS1 RT per cycle: mean of all MS1 RTs in the cycle.
        ms1_rts = np.empty(l, dtype=np.float32)
        for ind in range(l):
            s, e = ms1_cycle_starts[ind], ms1_cycle_ends[ind]
            ms1_rts[ind] = (self.run.ms1_RT[s:e].mean() if s < e
                            else self.run.ms1_RT[ms1_lefts[ind]])
        
        n_isotopes = min(MaxPrecNum, self.info.n_isotopes)
        n_prods = self.info.n_prods
        
        self.ms1    = np.zeros((l, n_isotopes))
        self.ms2    = np.zeros((l, n_prods))
        exp_ms1_mzs = np.zeros((l, n_isotopes))
        exp_ms2_mzs = np.zeros((l, n_prods))
        exp_ims     = np.zeros((l, n_isotopes))

        for ind in range(l):
            s, e = ms1_cycle_starts[ind], ms1_cycle_ends[ind]
            # Use all MS1 scans in the cycle; fall back to the single left scan.
            ms1_scan_range = range(s, e) if s < e else range(ms1_lefts[ind], ms1_lefts[ind] + 1)

            for j in range(n_isotopes):
                total_intensity = 0.0
                last_mz         = 0.0
                total_im        = 0.0
                n_scans         = 0
                for ms1_idx in ms1_scan_range:
                    scan_rt  = self.run.ms1_RT[ms1_idx]
                    query_mz = predicted_mz(self.run.MassCorrectionMS1, self.info.ms1_mzs[j],
                                            scan_rt, self.run.MassCalCenter, MassCalBins)
                    if self.run.tims:
                        inten, mz_obs, im_obs = self.run.ms1sp[ms1_idx].level(
                            report_mz_im, query_mz, self.run.MassAccuracyMS1,
                            self.IM, self.run.IM_window)
                        total_im += im_obs
                    else:
                        inten, mz_obs = self.run.ms1sp[ms1_idx].level(
                            query_mz, self.run.MassAccuracyMS1)
                    total_intensity += inten
                    if inten > 0.0:
                        last_mz = mz_obs  # keep the last observed m/z with signal
                    n_scans += 1
                self.ms1[ind, j]    = total_intensity
                exp_ms1_mzs[ind, j] = last_mz
                if self.run.tims and n_scans > 0:
                    exp_ims[ind, j] = total_im / n_scans

            i = indices_i[ind]
            for fr in range(n_prods):
                query_mz = predicted_mz(self.run.MassCorrectionMS2, self.info.products[fr].mz,
                                        ms2_rts[ind], self.run.MassCalCenter, MassCalBins)
                if self.run.tims:
                    self.ms2[ind, fr], exp_ms2_mzs[ind, fr], _ = self.run.ms2sp[i].level(
                        report_mz_im, query_mz, self.run.MassAccuracyMS2, self.IM, self.run.IM_window)
                else:
                    self.ms2[ind, fr], exp_ms2_mzs[ind, fr] = self.run.ms2sp[i].level(
                        query_mz, self.run.MassAccuracyMS2)
                if self.ms2[ind, fr] < MinXicIntensity:
                    self.ms2[ind, fr] = 0.0

        self.ms1_rts, self.ms2_rts = consolidate_by_ms1(ms1_rts, ms2_rts, method='first')
        _, self.ms1     = consolidate_by_ms1(ms1_rts, self.ms1,     method='first')
        _, self.ms2     = consolidate_by_ms1(ms1_rts, self.ms2,     method='sum')
        _, exp_ms1_mzs  = consolidate_by_ms1(ms1_rts, exp_ms1_mzs,  method='first')
        _, exp_ms2_mzs  = consolidate_by_ms1(ms1_rts, exp_ms2_mzs,  method='max')
        _, exp_ims      = consolidate_by_ms1(ms1_rts, exp_ims,       method='first')

        return exp_ms1_mzs, exp_ms2_mzs, exp_ims
    

    def collect_calibration_data(self, 
                                 apex_index, 
                                 exp_ims,
                                 exp_ms1_mzs, 
                                 exp_ms2_mzs, 
                                 best_ms2_score, 
                                 mono_ms1_score,
                                 ms1_ms2_corr):
        
        # collect data for mass calibration on the center point of best fragment
        ms1_mz = exp_ms1_mzs[apex_index, 0]
        ms2_mz = exp_ms2_mzs[apex_index, self.info.best_ms2_xic_index]
        if self.run.tims:
            im = exp_ims[apex_index, 0]
        
        if (best_ms2_score >= MassCalScore and 
            mono_ms1_score >= MassCalScore and 
            ms1_ms2_corr >= MinMs1Ms2Corr and 
            ms1_mz > E and ms2_mz > E and
            self.ms2[apex_index, self.info.best_ms2_xic_index] >= MinApexIntensity):
            self.info.mass_delta_ms1 = ms1_mz - self.prec.mz
            self.info.mass_delta_ms2 = ms2_mz - self.info.best_ms2_xic_mz
            if self.run.tims:
                self.info.im_delta = im - self.IM
        else:
            self.info.mass_delta_ms1 = 0.0
            self.info.mass_delta_ms2 = 0.0
            if self.run.tims:
                self.info.im_delta = 0.0
            
    
    def collect_xic_info(self, stage, n_xic_points):
        # for stage = 0, extract info for mass calibration
        # for stage = 1, perform quantification and export XIC
        if not self.fInit:
            # theoritical MS1 m/zs
            n_isotopes, isotope_dist = estimate_isotopes(self.prec.mz * self.prec.charge)
            self.info.n_isotopes = n_isotopes
            self.info.isotope_dist = isotope_dist
            self.info.ms1_mzs = self.prec.mz + (C13delta * np.arange(0, self.info.n_isotopes) / self.prec.charge)
            # get products info and build ms2 scan index
            self.prepare_products(MinProdNum, MaxProdNum)
            if not self.info.n_prods:
                return
            self.build_index()
            self.fInit = 1
            
        self.info.clear()
        # determine initial peak based on RT
        ms2_RTs_window = self.run.ms2_RT[self.ms2_indices_window]
        ms2_index_RT = min(np.argmin(np.abs(ms2_RTs_window - self.RT)), 
                           len(self.ms2_indices_window) - 3)
           
        # set XIC extraction boundaries 
        R = int(n_xic_points // 2)
        start = max(0, ms2_index_RT - R)
        stop = min(len(self.ms2_indices_window) - 2, ms2_index_RT + R)
        
        # extract XICs
        report_mz_im = 1 if stage == 0 else 0
        exp_ms1_mzs, exp_ms2_mzs, exp_ims = self.extract_xic(start, stop + 1, report_mz_im)
        self.ms1_i = interpolate_ms1(self.ms1_rts, self.ms1, self.ms2_rts)
        
        # evaluate XICs
        initial_apex = ms2_index_RT - start
        self.info.initial_apex = initial_apex
        self.info.apex = initial_apex
        xics_outputs = evaluate_xics(
            self.reported,
            self.ms1_i,
            self.ms2,
            center=initial_apex,
            radius=InitialRadius,
            corr_method=CorrMethod,
            corr_weight=CorrWeight,
            shape_weight=ShapeWeight,
            intensity_weight=IntensityWeight,
            ref_xic_source=RefXicSource,
            within_modality_weight=0.5,
            cross_modality_weight=0.5,
            rt_prediction_error=self.run.rt_model_error, 
            prior_apex_floor=0.5,
            min_ms2_points=1,
            min_ms1_points=2,
            ms1_max_apex_distance=self.run.rt_model_error, 
        )
        
        # ------------------------------------------------------------------
        # Candidate-level XIC scores
        # ------------------------------------------------------------------
        self.info.scores["corr_score"] = xics_outputs["corr_scores"]
        self.info.scores["shape_score"] = xics_outputs["shape_scores"]
        self.info.scores["intensity_score"] = xics_outputs["intensity_scores"]
        self.info.scores["combined_score"] = xics_outputs["combined_scores"]
        
        # ------------------------------------------------------------------
        # New joint MS1/MS2 strategy metrics
        # ------------------------------------------------------------------
        self.info.mono_ms1_xic_score = xics_outputs["mono_ms1_xic_score"]
        self.info.best_ms1_xic_score = xics_outputs["best_ms1_xic_score"]
        self.info.best_ms2_xic_score = xics_outputs["best_ms2_xic_score"]
        self.info.best_ms1_xic_index = xics_outputs["best_ms1_xic_index"]
        self.info.best_ms2_xic_index = xics_outputs["best_ms2_xic_index"]
        self.info.cross_ms1_ms2_corr = xics_outputs["cross_ms1_ms2_corr"]
        self.info.ms1_internal_corr = xics_outputs["ms1_internal_corr"]
        self.info.ms2_internal_corr = xics_outputs["ms2_internal_corr"]
        
        # ------------------------------------------------------------------
        # Best-ranked candidate
        # ------------------------------------------------------------------
        self.info.ref_xic = xics_outputs["ref_xic"]
        self.info.ref_xic_type = xics_outputs["ref_xic_type"]
        self.info.ref_xic_score = xics_outputs["ref_xic_score"]
        self.info.ref_xic_status = xics_outputs["ref_xic_status"]
        
        self.info.apex = xics_outputs["apex"]
        self.info.fwhm = xics_outputs["fwhm"]
        
        self.info.ms1_candidate_apex = xics_outputs["ms1_candidate_apex"]
        self.info.ms1_fwhm_left = xics_outputs["ms1_fwhm_left"]
        self.info.ms1_fwhm_right = xics_outputs["ms1_fwhm_right"]
        
        self.info.apex_distance = xics_outputs["apex_distance"]
        self.info.apex_source = xics_outputs["apex_source"]
        self.info.evidence_tier = xics_outputs["evidence_tier"]
        
        # Search-derived XIC usability
        self.info.ms1_n_usable_xics_region = xics_outputs["ms1_n_usable_xics_region"]
        self.info.ms2_n_usable_xics_region = xics_outputs["ms2_n_usable_xics_region"]
        self.info.ms1_n_usable_xics_local = xics_outputs["ms1_n_usable_xics_local"]
        self.info.ms2_n_usable_xics_local = xics_outputs["ms2_n_usable_xics_local"]
        
        #
        best_prod = xics_outputs["best_ms2_xic_index"]
        self.info.best_ms2_xic_mz = self.info.products[best_prod].mz
        
        # if stage = 0, exit the function
        if stage == 0:
            self.collect_calibration_data(xics_outputs["apex"], 
                                          exp_ims,
                                          exp_ms1_mzs, 
                                          exp_ms2_mzs, 
                                          self.info.best_ms2_xic_score, 
                                          self.info.mono_ms1_xic_score,
                                          self.info.cross_ms1_ms2_corr)
            return
        
        # save XICs
        self.save_xics()
        

    def save_xics(self):
   
        xg = XICGroup()
        
        xg.run = self.run.run_index
        xg.pr_index = self.prec.index
        xg.pr_id = self.run.lib.entries[self.prec.index].name
        xg.reported = self.reported
        
        xg.info.ms1_mzs = self.info.ms1_mzs
        xg.info.products = self.info.products
        xg.info.prodi = self.info.prodi
        
        xg.info.isotope_dist = self.info.isotope_dist
        
        xg.info.initial_RT = self.RT
        xg.info.initial_apex = self.info.initial_apex
        xg.info.apex = self.info.apex
        xg.info.fwhm = self.info.fwhm
        
        xg.info.n_isotopes = self.info.n_isotopes
        xg.info.n_prods = self.info.n_prods
        
        xg.info.scores = self.info.scores
        xg.info.best_ms2_xic_mz = self.info.best_ms2_xic_mz
        
        xg.info.mono_ms1_xic_score = self.info.mono_ms1_xic_score
        xg.info.best_ms1_xic_score = self.info.best_ms1_xic_score
        xg.info.best_ms2_xic_score = self.info.best_ms2_xic_score
        xg.info.best_ms1_xic_index = self.info.best_ms1_xic_index
        xg.info.best_ms2_xic_index = self.info.best_ms2_xic_index
        
        xg.info.cross_ms1_ms2_corr = self.info.cross_ms1_ms2_corr
        xg.info.ms1_internal_corr = self.info.ms1_internal_corr
        xg.info.ms2_internal_corr = self.info.ms2_internal_corr
        
        xg.info.ref_xic = self.info.ref_xic
        xg.info.ref_xic_type = self.info.ref_xic_type
        xg.info.ref_xic_score = self.info.ref_xic_score
        xg.info.ref_xic_status = self.info.ref_xic_status
        
        xg.info.apex_distance = self.info.apex_distance
        xg.info.apex_source = self.info.apex_source
        xg.info.evidence_tier = self.info.evidence_tier

        xg.info.ms1_candidate_apex = self.info.ms1_candidate_apex
        xg.info.ms1_fwhm_left = self.info.ms1_fwhm_left
        xg.info.ms1_fwhm_right = self.info.ms1_fwhm_right

        xg.info.ms1_n_usable_xics_region = self.info.ms1_n_usable_xics_region
        xg.info.ms2_n_usable_xics_region = self.info.ms2_n_usable_xics_region
        xg.info.ms1_n_usable_xics_local = self.info.ms1_n_usable_xics_local
        xg.info.ms2_n_usable_xics_local = self.info.ms2_n_usable_xics_local
        
        # store RTs and intensities
        xg.xics = np.concat((self.ms1_rts.reshape(-1, 1), 
                             self.ms2_rts.reshape(-1, 1), 
                             self.ms1, 
                             self.ms2,
                             self.ms1_i), axis=1)

        self.run.XIC_list.append(xg) 

class Run:
    def __init__(self, _run_index: int):
        
        self.run_index = _run_index
        self.file_path = ""
        self.run_name = ""
        self.tims = False

        # 
        self.ms1sp = None
        self.ms2sp = None
        self.ms1_flat_mz = None
        self.ms1_flat_intensity = None
        self.ms1_offsets = None
        self.ms2_flat_mz = None
        self.ms2_flat_intensity = None
        self.ms2_offsets = None
        
        self.ms1_RT = None
        self.ms2_RT = None
        
        self.RT_min = 0.0
        self.RT_max = 0.0
        
        self.ms2_cycle = []
        self.n_cycle = 0
        self.n_ms2_per_cycle = 0
        self.cycle_time = 0.0
        self.window_settings = {}
        
        self.rt_model_error = 0
        self.n_xic_points = 0
        
        self.min_ms2_mz = 0.0
        self.max_ms2_mz = INF
        self.min_ms1_isolation_mz = 0.0
        self.max_ms1_isolation_mz = INF
        
        #
        self.lib = None
        self.entries = []
        self.XIC_list = []
        
        #
        self.IM_window = 0.0
        self.MassAccuracyMS2 = args.mass_acc_ms2 / 1000000.0
        self.MassAccuracyMS1 = args.mass_acc_ms1 / 1000000.0
        
        self.mz_calibrated = False
        self.acc_ms2_calibrated = False
        self.acc_ms1_calibrated = False

        self.MassCorrectionMS2 = np.zeros(1 + 2 * MassCalBinsMax)
        self.MassCorrectionMS1 = np.zeros(1 + 2 * MassCalBinsMax)
        self.MassCalSplit = np.zeros(MassCalBinsMax + 1)
        self.MassCalCenter = np.zeros(MassCalBinsMax + 1)
    
    def free(self):
        # for arrays
        self.ms1sp[:] = None
        self.ms2sp[:] = None
        self.ms1_RT[:] = None
        self.ms2_RT[:] = None
        self.ms1_flat_mz = None
        self.ms1_flat_intensity = None
        self.ms1_offsets = None
        self.ms2_flat_mz = None
        self.ms2_flat_intensity = None
        self.ms2_offsets = None
        # for lib
        self.lib = None
        # for lists
        self.entries.clear()
        self.XIC_list.clear()
    
    def load(self, ms_file_path):
        
        self.file_path = ms_file_path
        self.run_name = remove_extension(get_file_name(self.file_path))
        
        logging.info(f"Loading data from file {self.file_path}")
        
        ext = get_extension(ms_file_path)
        if ext == 'raw':
            self.ms1sp, self.ms2sp = load_thermo_raw(ms_file_path)
        elif ext == 'mzml':
            self.ms1sp, self.ms2sp = load_mzml(ms_file_path)
        elif ext == 'd':
            self.tims = True
            self.ms1sp, self.ms2sp = load_timsTOF_d(ms_file_path)
        
        ms1_cnt = len(self.ms1sp)
        ms2_cnt = len(self.ms2sp)
        
        if not ms2_cnt:
            logging.warning("No MS2 spectra: aborting")
            return False

        # derive window settings and cycle information
        self.init()
        logging.info(f"Data loaded: {ms1_cnt} MS1 spectra and {ms2_cnt} MS2 spectra in {self.n_cycle} cycles")
        
        return True

    def init(self):
        '''
        Derive window settings and cycle information of current run.
        '''
        
        # extract all RTs
        self.ms1_RT = np.array([sp.RT for sp in self.ms1sp])
        self.ms2_RT = np.array([sp.RT for sp in self.ms2sp])
        
        #
        self.RT_min = np.min(self.ms1_RT)
        self.RT_max = np.max(self.ms2_RT)
        logging.info(f"RT range: {self.RT_min:.2f} - {self.RT_max:.2f} minutes") 
        
        # define ms1 windows as the m/z range of the ms1 spectra
        ms1_mz_min = INF
        ms1_mz_max = -INF
        for ms1 in self.ms1sp:
            if not ms1.size():
                ms1.window_low = ms1.window_high = 0.0
            else:
                ms1.window_low = ms1.peaks_mz[0] + 0.25
                ms1.window_high = ms1.peaks_mz[-1] - 0.25
                if ms1.window_low < ms1_mz_min:
                    ms1_mz_min = ms1.window_low
                if ms1.window_high > ms1_mz_max:
                    ms1_mz_max = ms1.window_high
        logging.info(f"Detected MS1 spectra m/z range: {(ms1_mz_min-0.25):.1f} - {(ms1_mz_max+0.25):.1f}")
        
        # get the ms2 spectra m/z range
        ms2_mz_min = INF
        ms2_mz_max = -INF
        for ms2 in self.ms2sp:
            if ms2.size():
                if ms2.peaks_mz[0] < ms2_mz_min:
                    ms2_mz_min = ms2.peaks_mz[0]
                if ms2.peaks_mz[-1] > ms2_mz_max:
                    ms2_mz_max = ms2.peaks_mz[-1]
        if ms2_mz_min < ms2_mz_max:
            self.min_ms2_mz = np.round(ms2_mz_min)
            self.max_ms2_mz = np.round(ms2_mz_max)
        logging.info(f"Detected MS2 spectra m/z range: {self.min_ms2_mz} - {self.max_ms2_mz}")
        
        # assigning the cycle number based on the cycle of ms2 spectrum
        ms2_cnt = len(self.ms2sp)
        self.ms2_cycle = np.zeros(ms2_cnt, dtype=np.int32)
        window_low_begin = self.ms2sp[0].window_low
        window_high_begin = self.ms2sp[0].window_high
        self.min_ms1_isolation_mz = window_low_begin
        self.max_ms1_isolation_mz = window_high_begin
        cycle = 0
        for i in range(1, ms2_cnt):
            # for overlapping m/z windows
            # Case #1:
            # o------o
            #    o------o
            if (self.ms2sp[i - 1].window_low < self.ms2sp[i].window_low - E
                and self.ms2sp[i - 1].window_high < self.ms2sp[i].window_high - E
                and self.ms2sp[i - 1].window_high > self.ms2sp[i].window_low + E):
                # make the border equal to the average 
                # o----o
                #      o----o
                self.ms2sp[i - 1].window_high = self.ms2sp[i].window_low = (self.ms2sp[i - 1].window_high + self.ms2sp[i].window_low) * 0.5
            # Case #2:
            #     o------o
            # o------o
            if (self.ms2sp[i - 1].window_low > self.ms2sp[i].window_low + E
                and self.ms2sp[i - 1].window_high > self.ms2sp[i].window_high + E
                and self.ms2sp[i - 1].window_low < self.ms2sp[i].window_high - E):
                # make the border equal to the average 
                #      o----o
                # o----o
                self.ms2sp[i - 1].window_low = self.ms2sp[i].window_high = (self.ms2sp[i - 1].window_low + self.ms2sp[i].window_high) * 0.5
            
            # assigning the cycle number to each ms2 spectrum
            if (self.ms2sp[i].window_high == window_high_begin
                and self.ms2sp[i].window_low == window_low_begin):
                cycle += 1
                self.ms2_cycle[i] = cycle
            else:
                self.ms2_cycle[i] = cycle

            # get the isolation window m/z range
            if self.ms2sp[i].window_low < self.min_ms1_isolation_mz:
                self.min_ms1_isolation_mz = self.ms2sp[i].window_low
            if self.ms2sp[i].window_high > self.max_ms1_isolation_mz:
                self.max_ms1_isolation_mz = self.ms2sp[i].window_high
        
        logging.info(f"Detected MS1 isolation m/z range: {self.min_ms1_isolation_mz:.1f} - {self.max_ms1_isolation_mz:.1f}")
        
        # number of total cycles
        self.n_cycle = cycle + 1
        
        # extract window settings
        window_idx = []
        window_lows = []
        window_highs = []
        for i, x in enumerate(self.ms2_cycle):
            if x == 0:
                window_idx.append(i)
                window_lows.append(self.ms2sp[i].window_low)
                window_highs.append(self.ms2sp[i].window_high)
            else:
                break 
        self.window_settings = {'Index': np.array(window_idx), 
                                'Low': np.array(window_lows), 
                                'High': np.array(window_highs)}
        
        lows, highs = self.window_settings['Low'], self.window_settings['High']
        windows_sorted = sorted(zip(lows, highs))
        compact = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in windows_sorted)
        logging.info("Isolation windows (n = %d): %s", len(windows_sorted), compact)
        
        # assigning the cycle number to ms1 spectra
        last_ms2_indices = list({val: i for i, val in enumerate(self.ms2_cycle)}.values())
        last_ms2_rt = self.ms2_RT[last_ms2_indices]
        self.ms1_cycle = [bisect.bisect_left(last_ms2_rt, rt) for rt in self.ms1_RT]
        
        # Count the number of ms1/ms2 spectra in a cycle
        def get_cycle_log(counts_per_cycle, cycle_nums, ms_level):
            # Sort by cycle_counts (descending)
            sorted_pairs = sorted(zip(counts_per_cycle, cycle_nums), key=lambda x: x[1], reverse=True)
            log_messages = [
                f"{n_cycle} {'cycle has' if n_cycle == 1 else 'cycles have'} {count} {ms_level} spectra" 
                for count, n_cycle in sorted_pairs
            ]
            cycle_log = "; ".join(log_messages)
            return f"For {ms_level}: " + cycle_log
        
        # MS1
        _, ms1_counts = np.unique(self.ms1_cycle, return_counts=True)
        ms1_per_cycle, cycle_counts = np.unique(ms1_counts, return_counts=True)
        logging.info(get_cycle_log(ms1_per_cycle, cycle_counts, 'MS1'))
        self.n_ms1_per_cycle = np.median(ms1_per_cycle)
        # MS2
        _, ms2_counts = np.unique(self.ms2_cycle, return_counts=True)
        ms2_per_cycle, cycle_counts = np.unique(ms2_counts, return_counts=True)
        logging.info(get_cycle_log(ms2_per_cycle, cycle_counts, 'MS2'))
        self.n_ms2_per_cycle = np.max(ms2_per_cycle)
        
        # Cycle time
        self.cycle_time = get_cycle_time(self.file_path, out_dir=out_int_dir)
        logging.info(f"Cycle time: {self.cycle_time:.4f} minutes ({(self.cycle_time*60):.2f} seconds)") 


    def get_window_index(self, mz):
        window_lows = self.window_settings['Low']
        window_highs = self.window_settings['High']
        
        indices = np.where((mz >= window_lows) & (mz < window_highs))[0]
        if indices.size > 0:
            return indices[0]
        
        return None
    

    def load_library(self, _lib):
        self.lib = _lib
        # filter precursors 
        cnt = 0
        for it in _lib.entries:
            window_index = self.get_window_index(it.prec.mz)
            if window_index is not None:
                entry = RunEntry()
                entry.name = it.name
                entry.init(self, it.prec)
                self.entries.append(entry)
                cnt += 1
        logging.info(f"{cnt} precursors in the library are within the MS1 isolation m/z range")
    
    
    def load_prec_rts(self, df, missing_status):
        if self.tims:         
            logging.info("Loading precursor RT and IM")
        else:
            logging.info("Loading precursor RT")
            
        for entry in self.entries:
            entry.mass_cal = df.loc[entry.name, 'mass_cal']
            entry.re_quan = df.loc[entry.name, 're_quan']
            if missing_status.loc[entry.name, self.run_name] == 'reported':
                entry.reported = 1
            
            run_RT = df.loc[entry.name, self.run_name+'_RT']
            if not np.isnan(run_RT):
                entry.RT = run_RT
            
            if self.tims:
                run_IM = df.loc[entry.name, self.run_name+'_IM']
                if not np.isnan(run_IM):
                    entry.IM = run_IM
    
    
    def determine_n_xic_points(self):
        logging.info("Determining the number of data points to extract for an XIC")
        
        rt_fit_metrics = pd.read_csv(
            out_int_dir / "rt_prediction/rt_production_fit_metrics.tsv",
            sep="\t",
        )
        
        fit_p95 = rt_fit_metrics.loc[
            rt_fit_metrics["run_name"] == self.run_name,
            "training_absolute_residual_p95",
        ].iloc[0]
        
        rt_error_inflation = 1.2
        prediction_error_points = int(
            np.ceil((fit_p95 * rt_error_inflation) / self.cycle_time)
        )
        
        half_window_points = max(
            MinXICPointsHalf,
            prediction_error_points * 2,
        )
        nXICPoints = 2 * half_window_points + 1
        
        logging.info(
            "XIC window: %d total points (%d points on each side)",
            nXICPoints,
            half_window_points,
        )
        
        self.rt_model_error = prediction_error_points 
        self.n_xic_points = nXICPoints
    
    
    def process(self, info=None):
        self.determine_n_xic_points()
        self.calibrate_mass()
        logging.info("Collecting XIC info for each precursor")
        for entry in progress_bar(self.entries, args.show_progress, args.inter_progress):
            if entry.re_quan == 1:
                entry.collect_xic_info(stage=1, n_xic_points=self.n_xic_points)
        self.export_xics(info=info)


    def calibrate_mass(self): 
        if Calibrate:
            self.MassAccuracyMS1 = MassCalAcc / 1000000.0
            self.MassAccuracyMS2 = MassCalAcc / 1000000.0
            logging.info(f"Initital mass accuracy used for calibration: {MassCalAcc}, {MassCalAcc} ppm (MS1, MS2)")
        
        if self.tims:
            if not args.fix_im_window:
                self.IM_window = IMWindowCal
                logging.info(f"Initital ion mobility window used for calibration: {self.IM_window}")
            else:
                self.IM_window = args.im_window
                logging.info(f"Ion mobility window for XIC extraction set to: {self.IM_window}")
            
        if Calibrate and not self.mz_calibrated:
            self.do_mass_calibration(save_cal_info=False)
            
        if self.tims and not args.fix_im_window:
            self.check_ion_mobility_error(save_cal_info=False)
        
        if self.acc_ms2_calibrated:
            logging.info(f"Refined mass accuracy used for calibration: {self.MassAccuracyMS1*1e6}, {self.MassAccuracyMS2*1e6} ppm (MS1, MS2)")
            if self.tims and not args.fix_im_window:
                logging.info(f"Refined ion mobility window: {self.IM_window}")
            self.do_mass_calibration(save_cal_info=SaveSupplements)
            
        if self.tims and not args.fix_im_window:
            self.check_ion_mobility_error(save_cal_info=SaveSupplements)
    
        # optimizing mass accuracy for XIC extraction
        if self.acc_ms2_calibrated and not args.fix_mass_acc:
            logging.info(f"Using XIC extraction mass accuracy calculated by data: {self.MassAccuracyMS1*1e6}, {self.MassAccuracyMS2*1e6} ppm (MS1, MS2)")
            args.mass_acc_ms1 = self.MassAccuracyMS1 * 1000000.0
            args.mass_acc_ms2 = self.MassAccuracyMS2 * 1000000.0
            
        if self.tims and not args.fix_im_window and not IndividualIMWindow:
            logging.info(f"Final ion mobility window for XIC extraction: {self.IM_window}")
            args.im_window = self.IM_window
        
        if args.fix_mass_acc:
            logging.info(f"XIC extraction mass accuracy set to: {args.mass_acc_ms1}, {args.mass_acc_ms2} ppm (MS1, MS2)")
            self.MassAccuracyMS1 = args.mass_acc_ms1 / 1000000.0
            self.MassAccuracyMS2 = args.mass_acc_ms2 / 1000000.0
        
        if not IndividualMassAcc:
            args.fix_mass_acc = 1
        
        if self.tims and not IndividualIMWindow:
            args.fix_im_window = 1
    
    
    def do_mass_calibration(self, save_cal_info=True):
        for entry in progress_bar(self.entries, args.show_progress, args.inter_progress):
            if entry.mass_cal == 1 and entry.reported == 1:
                entry.collect_xic_info(stage=0, n_xic_points=self.n_xic_points)
        
        mass_cali_data = {'theoretical_mz_ms1': [],
                          'theoretical_mz_ms2': [],
                          'observed_mz_ms1': [],
                          'observed_mz_ms2': [],
                          'mass_delta_ms1': [],
                          'mass_delta_ms2': [],
                          'experimental_rt': [],
                          }
        
        for entry in self.entries:
            if entry.mass_cal == 1:
                #break
                if np.abs(entry.info.mass_delta_ms2) > E and np.abs(entry.info.mass_delta_ms1) > E:
                    mass_cali_data['experimental_rt'].append(entry.RT)
                    # theoretical m/z
                    mass_cali_data['theoretical_mz_ms1'].append(entry.prec.mz)
                    mass_cali_data['theoretical_mz_ms2'].append(entry.info.best_ms2_xic_mz)
                    # mass difference between theoretical m/z and observed m/z 
                    mass_cali_data['mass_delta_ms1'].append(entry.info.mass_delta_ms1)
                    mass_cali_data['mass_delta_ms2'].append(entry.info.mass_delta_ms2)
                
        mass_cali_data['theoretical_mz_ms1'] = np.array(mass_cali_data['theoretical_mz_ms1'])
        mass_cali_data['theoretical_mz_ms2'] = np.array(mass_cali_data['theoretical_mz_ms2'])
        mass_cali_data['mass_delta_ms1'] = np.array(mass_cali_data['mass_delta_ms1'])
        mass_cali_data['mass_delta_ms2'] = np.array(mass_cali_data['mass_delta_ms2'])
        mass_cali_data['experimental_rt'] = np.array(mass_cali_data['experimental_rt'])
        
        mass_cali_data['observed_mz_ms1'] = mass_cali_data['theoretical_mz_ms1'] + mass_cali_data['mass_delta_ms1']
        mass_cali_data['observed_mz_ms2'] = mass_cali_data['theoretical_mz_ms2'] + mass_cali_data['mass_delta_ms2']

        mass_cali_data = pl.DataFrame(mass_cali_data)
                
        # sort by RT
        mass_cali_data = mass_cali_data.sort("experimental_rt")
        
        # setup for mass calibration bins
        (MassCalBins,
         self.MassCalSplit,
         self.MassCalCenter) = split_into_rt_bins(mass_cali_data['experimental_rt'].to_numpy(),
                                                  max_num_bins=MassCalBinsMax)
        self.MassCalSplit  = np.asarray(self.MassCalSplit,  dtype=np.float64)
        self.MassCalCenter = np.asarray(self.MassCalCenter, dtype=np.float64)
        
        # calibration       
        logging.info(f"{len(mass_cali_data)} MS1 m/z and MS2 m/z are used for mass calibration") 
        mass_cali_data = self.mass_calibration(mass_cali_data, level='ms1')
        mass_cali_data = self.mass_calibration(mass_cali_data, level='ms2')
        
        if save_cal_info:
            out_cal_dir = out_int_dir / 'mass_calibration'
            out_cal_dir.mkdir(parents=True, exist_ok=True)
            mass_cali_data.write_csv(out_cal_dir / (self.run_name + "_mass_cal.csv"))
            mass_calibration_plot(mass_cali_data, 'ms1', run_name=self.run_name, out_dir=out_cal_dir)
            mass_calibration_plot(mass_cali_data, 'ms2', run_name=self.run_name, out_dir=out_cal_dir)

        self.mz_calibrated = True


    def mass_calibration(self, mass_cali_data, level):
        assert level in ['ms1', 'ms2']
    
        theo_col = f'theoretical_mz_{level}'
        delta_col = f'mass_delta_{level}'
        obs_col = f'observed_mz_{level}'
        query_col = f'query_mz_{level}'
        
        # Step 1: Check mass error before correction
        theo = mass_cali_data[theo_col].to_numpy()
        b_r = mass_cali_data[delta_col].to_numpy() / theo
        mean = np.mean(b_r)
        sd = np.std(b_r)
        logging.info(f"Mean and SD of {level.upper()} mass error before calibration: {(mean * 1e6):2f} and {(sd * 1e6):2f} ppm")
        acc_no_cal = np.round(max(abs(mean+3*sd), abs(mean-3*sd)) * 1e6 + 5) / 1e6
        
        # Step 2: Build calibration model
        correction = build_mass_cali_model(
            theoretical_mzs = mass_cali_data[theo_col].to_numpy(),
            delta_mzs = mass_cali_data[delta_col].to_numpy(),
            rts = mass_cali_data['experimental_rt'].to_numpy(),
            RTBinCenter = self.MassCalCenter,
            nRTBins = MassCalBins
        )
        setattr(self, f'MassCorrection{level.upper()}', correction)
    
        # Step 3: Recalculate query m/z
        query_mz = predicted_mz_parallel(
            correction,
            mass_cali_data[theo_col].to_numpy().reshape(-1, 1),
            mass_cali_data['experimental_rt'].to_numpy(),
            self.MassCalCenter,
            MassCalBins
        ).reshape(-1)
        mass_cali_data = mass_cali_data.with_columns(
            pl.Series(name=query_col, values=query_mz)
        )
    
        # Step 4: Compute mass error
        observed = mass_cali_data[obs_col].to_numpy()
        b_r = (observed - query_mz) / theo
        mean = np.mean(b_r)
        sd = np.std(b_r)
        logging.info(f"Mean and SD of {level.upper()} mass error after calibration: {(mean * 1e6):2f} and {(sd * 1e6):2f} ppm")
        acc = np.round(max(abs(mean+3*sd), abs(mean-3*sd)) * 1e6 + 5) / 1e6
        setattr(self, f'MassAccuracy{level.upper()}', acc)
        setattr(self, f'acc_{level}_calibrated', True)
    
        if acc_no_cal < acc:
            logging.info(f"No {level.upper()} mass correction required")
            setattr(self, f'MassAccuracy{level.upper()}', acc_no_cal)
            setattr(self, f'MassCorrection{level.upper()}', np.zeros_like(correction))
        
        return mass_cali_data
    
    
    def check_ion_mobility_error(self, save_cal_info):
        
        im_cali_data = {'theoretical_mz': [],
                        'query_im': [],
                        'im_delta': [],
                        }
        
        for entry in self.entries:
            if entry.mass_cal == 1:
                if np.abs(entry.info.im_delta) > 0.0:
                    # theoretical m/z
                    im_cali_data['theoretical_mz'].append(entry.prec.mz)
                    # mass difference between theoretical m/z and observed m/z 
                    im_cali_data['query_im'].append(entry.IM)
                    im_cali_data['im_delta'].append(entry.info.im_delta)
                
        im_cali_data['theoretical_mz'] = np.array(im_cali_data['theoretical_mz'])
        im_cali_data['query_im'] = np.array(im_cali_data['query_im'])
        im_cali_data['im_delta'] = np.array(im_cali_data['im_delta'])

        im_cali_data = pl.DataFrame(im_cali_data) 
        im_cali_data = im_cali_data.sort("query_im")
        
        # calibration       
        logging.info(f"{len(im_cali_data)} IM values are used for IM window determination") 
        mean = im_cali_data['im_delta'].mean()
        sd = im_cali_data['im_delta'].std()
        logging.info(f"Mean and SD of ion mobility error: {mean:4g} and {sd:5g}")
        im_boundary = max(abs(mean+3*sd), abs(mean-3*sd)) * 1.25
        self.IM_window = np.round(im_boundary, 3)
        logging.info(f"Run-specific ion mobility window: {self.IM_window}")

        if save_cal_info:
            out_cal_dir = out_int_dir / 'mass_calibration'
            out_cal_dir.mkdir(parents=True, exist_ok=True)
            im_cali_data.write_csv(out_cal_dir / (self.run_name + "_im_error.csv"))
            im_error_plot(im_cali_data, run_name=self.run_name, out_dir=out_cal_dir)
        
    
    
    def export_xics(self, info=None):        
        rows = {
            "prec_index": [],
            "Precursor.Id": [],
            "reported": [],
        
            "n_isotopes": [],
            "n_prods": [],
            "isotope_dist": [],
        
            "initial_RT": [],
            "initial_apex": [],
        
            "ref_xic": [],
            "ref_xic_type": [],
            "ref_xic_score": [],
            "ref_xic_status": [],
            
            "apex_distance": [],
            "apex_source": [],
            "evidence_tier": [],
            
            "apex": [],
            "fwhm": [],
            
            'ms1_candidate_apex': [],
            'ms1_fwhm_left': [],
            'ms1_fwhm_right': [],
            
            'ms1_n_usable_xics_region': [],
            'ms2_n_usable_xics_region': [],
            'ms1_n_usable_xics_local': [],
            'ms2_n_usable_xics_local': [],
            
            # Strategy-support / joint XIC metrics
            "mono_ms1_xic_score": [],
            "best_ms1_xic_score": [],
            "best_ms2_xic_score": [],
            "best_ms1_xic_index": [],
            "best_ms2_xic_index": [],
            "cross_ms1_ms2_corr": [],
            "ms1_internal_corr": [],
            "ms2_internal_corr": [],
        
            "ms1_mz": [],
            "ms2_mz": [],
            "ms2_indices": [],
        
            "combined_score": [],
            "corr_score": [],
            "shape_score": [],
            "intensity_score": [],
        
            "ms1_rt": [],
            "ms2_rt": [],
            "ms1": [],
            "ms2": [],
            "ms1_i": [],
        }
        
        for xg in self.XIC_list:
            info_ = xg.info
            n_isotopes, n_prods = info_.n_isotopes, info_.n_prods
            
            rows["prec_index"].append(xg.pr_index)
            rows["Precursor.Id"].append(xg.pr_id)
            rows["reported"].append(xg.reported)
            
            rows["n_isotopes"].append(n_isotopes)
            rows["n_prods"].append(n_prods)
            rows["isotope_dist"].append(info_.isotope_dist)
            
            rows["initial_RT"].append(info_.initial_RT)
            rows["initial_apex"].append(info_.initial_apex)
            
            rows["ref_xic"].append(info_.ref_xic)
            rows["ref_xic_type"].append(info_.ref_xic_type)
            rows["ref_xic_score"].append(info_.ref_xic_score)
            rows["ref_xic_status"].append(info_.ref_xic_status)
            
            rows["apex_distance"].append(info_.apex_distance)
            rows["apex_source"].append(info_.apex_source)
            rows["evidence_tier"].append(info_.evidence_tier)
            
            rows["apex"].append(info_.apex)
            rows["fwhm"].append(info_.fwhm)
            
            rows['ms1_candidate_apex'].append(info_.ms1_candidate_apex)
            rows['ms1_fwhm_left'].append(info_.ms1_fwhm_left)
            rows['ms1_fwhm_right'].append(info_.ms1_fwhm_right)
            
            rows['ms1_n_usable_xics_region'].append(info_.ms1_n_usable_xics_region)
            rows['ms2_n_usable_xics_region'].append(info_.ms2_n_usable_xics_region)
            rows['ms1_n_usable_xics_local'].append(info_.ms1_n_usable_xics_local)
            rows['ms2_n_usable_xics_local'].append(info_.ms2_n_usable_xics_local)
            
            rows["mono_ms1_xic_score"].append(info_.mono_ms1_xic_score)
            rows["best_ms1_xic_score"].append(info_.best_ms1_xic_score)
            rows["best_ms2_xic_score"].append(info_.best_ms2_xic_score)
            rows["best_ms1_xic_index"].append(info_.best_ms1_xic_index)
            rows["best_ms2_xic_index"].append(info_.best_ms2_xic_index)
            rows["cross_ms1_ms2_corr"].append(info_.cross_ms1_ms2_corr)
            rows["ms1_internal_corr"].append(info_.ms1_internal_corr)
            rows["ms2_internal_corr"].append(info_.ms2_internal_corr)
            
            rows["ms1_mz"].append(info_.ms1_mzs.tolist())
            rows["ms2_mz"].append([fr.mz for fr in info_.products])
            rows["ms2_indices"].append(list(info_.prodi))          # copy — don't mutate
            
            rows["combined_score"].append(list(info_.scores['combined_score']))
            rows["corr_score"].append(list(info_.scores['corr_score']))
            rows["shape_score"].append(list(info_.scores['shape_score']))
            rows["intensity_score"].append(list(info_.scores['intensity_score']))
    
            # xg.xics columns: [ms1_rt, ms2_rt, ms1(n_isotopes), ms2(n_prods), ms1_i(n_isotopes)]
            xics = xg.xics
            rows["ms1_rt"].append(xics[:, 0].tolist())
            rows["ms2_rt"].append(xics[:, 1].tolist())
            rows["ms1"].append(xics[:, 2:2 + n_isotopes].tolist())
            rows["ms2"].append(xics[:, 2 + n_isotopes:2 + n_isotopes + n_prods].tolist())
            rows["ms1_i"].append(xics[:, 2 + n_isotopes + n_prods:].tolist())
        
        #rows["Run"] = [self.run_name] * len(rows["Precursor.Id"])
        #rows["rt_model_error"] = [self.rt_model_error] * len(rows["Precursor.Id"])
        df = pl.DataFrame(rows)
    
        out_xic = out_int_dir / "xic_files" / f"{self.run_name}.xic.parquet"
        out_xic.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.write_parquet(out_xic)
            logging.info(f"XIC data saved to {out_xic}\n")
        except IOError:
            logging.warning("WARNING: cannot save the .parquet file to the desired location; attempting to save in the current working folder.\n")
            fallback_file = f"{self.run_name}.xic.parquet"
            try:
                df.write_parquet(fallback_file)
                logging.info(f"XIC data saved to {fallback_file}\n")
            except IOError:
                logging.error(f"ERROR: cannot save the .parquet file to {fallback_file}. Check if the destination folder is write-protected or the file is in use.\n")
    
        if info is not None:
            info.accumulate_from_xic_list(self.XIC_list, self.run_index)
            
        self.XIC_list.clear()


