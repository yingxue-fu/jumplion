# builtin
import bisect
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
import logging
# external
import numpy as np
import polars as pl
# local
from jumpdiaq.arguments import Arguments, get_extension
from jumpdiaq.utils import (CorrRadius, CorrMethod, WeightByShape, WeightByIntensity, 
                            CorrWeight, ShapeWeight, IntensityWeight, MinShapeScore,
                            MinXicIntensity, MinApexIntensity, MinApexEvidence,
                            MassCalBins, MassCalBinsMax, MassCalScore, C13delta, 
                            fExclude, E, INF, estimate_isotopes, consolidate_by_ms1,
                            remove_extension, get_file_name, progress_bar)
from jumpdiaq.read_ms_files import load_thermo_raw, load_mzml, load_timsTOF_d
from jumpdiaq.calibration import (split_into_rt_bins, 
                                  predicted_mz, 
                                  predicted_mz_parallel, 
                                  build_mass_cali_model, 
                                  mass_calibration_plot, 
                                  im_error_plot)
from jumpdiaq.elution_peaks import interpolate_ms1, identify_optimal_xic

args = Arguments()
OutDir = args.out_dir
out_int_dir = Path(OutDir) / 'intermediate'
SaveSupplements = args.save_supplements
# XIC extraction
nXICPoints = args.n_xic_points
Calibrate = args.mass_calibration
MassCalAcc = args.mass_cal_acc
IndividualMassAcc = args.individual_mass_acc
IMWindowCal = args.im_window_cal
IndividualIMWindow = args.individual_im_window
# XIC quantification
MinFr = args.min_fr_quant
MaxFr = args.max_fr_quant
MaxPrIsotopes = args.max_pr_isotopes
OptimalXicSource = args.optimal_xic_source


@dataclass
class XICGroupInfo:
    # MS1 XIC m/zs
    ms1_mzs: List[float] = field(default_factory=list)
    n_isotopic_peaks: int = 0

    # MS2 ion info
    products: List[None] = field(default_factory=list)   # A list of product ions to extract XIC
    prodi: List[int] = field(default_factory=list)          # Product ion index
    n_frs: int = 0
    
    # XIC apex and peak width (fwhm)
    apex: int = 0
    fwhm: int = 0
    
    # XIC scores
    scores: Dict[str, List[float]] = field(default_factory=lambda: {
        'corr_score': [],
        'shape_score': [],
        'intensity_score': [],
        'score': []
    })

    # Optimal XIC for interference removal
    optimal_xic: int = 0               # Index of optimal XIC
    optimal_xic_type: int = 0          # 1 = MS1, 2 = MS2, 0 = Not found

    # For mass calibration
    best_ms2_xic: int = 0
    best_ms2_mz: float = 0.0
    mass_delta_ms2: float = 0.0
    mass_delta_ms1: float = 0.0
    
    im_delta: float = 0.0

    def clear(self):
        self.apex = 0
        self.fwhm = 0
        self.scores = {
            'corr_score': [],
            'shape_score': [],
            'intensity_score': [],
            'score': []
        }
        self.optimal_xic = 0
        self.optimal_xic_type = 0
        self.best_ms2_xic = 0
        self.best_ms2_mz = 0.0
        self.mass_delta_ms2 = 0.0
        self.mass_delta_ms1 = 0.0
        self.im_delta = 0.0

@dataclass
class XICGroup:
    run: int = -1       # run index
    pr_id: int = -1     # precursor index in the library
    pr_name: str = ""   # precursor id
    
    info: XICGroupInfo = field(default_factory=XICGroupInfo)  # XIC info
    xics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # RTs and intensities

    
class RunEntry:
    def __init__(self):
        self.run = None      # Run class
        self.name = ""       # precursor id
        self.prec = None     # Precursor class
        self.mass_cal = 0
        
        # RT and IM (ion mobility)
        self.RT = 0.0 
        self.IM = 0.0
        
        self.fInit = 0       # binary 000
        self.target_ms2_index = np.zeros(1)  # ms2 scan indices corresponding to the precursor
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
        
        self.info.n_frs = len(self.info.products)
        if self.info.n_frs < min_prod:  # default: at least 3 products
            self.info.products.clear()
            self.info.prodi.clear()
    
    def build_index(self):
        window_index = self.run.get_window_index(self.prec.mz)
        target_ms2_index = (np.arange(self.run.n_cycle) * self.run.n_ms2_per_cycle) + window_index
        self.target_ms2_index = target_ms2_index[target_ms2_index < len(self.run.ms2sp)]
    
    def extract_xic(self, from_idx: int, to_idx: int, report_mz_im: int):
        l = to_idx - from_idx
        
        ms1_rts = np.zeros(l)
        ms2_rts = np.zeros(l)
        
        # extracted intensities
        self.ms1 = np.zeros((l, MaxPrIsotopes))   # of isotopic peaks
        self.ms2 = np.zeros((l, MaxFr))
        
        # extracted mzs
        exp_ms1_mzs = np.zeros((l, MaxPrIsotopes))
        exp_ms2_mzs = np.zeros((l, MaxFr))
        exp_ims = np.zeros((l, MaxPrIsotopes))
                
        for k in range(from_idx, to_idx):
            
            i = self.target_ms2_index[k]
            rt = self.run.ms2_RT[i]
            ind = k - from_idx
            
            ms1_left = np.searchsorted(self.run.ms1_RT, rt) - 1 
            ms1_rts[ind] = self.run.ms1_RT[ms1_left]
            
            n_isotopes = min(MaxPrIsotopes, self.info.n_isotopic_peaks)
            for j in range(n_isotopes):
                query_mz = predicted_mz(self.run.MassCorrectionMS1, self.info.ms1_mzs[j], ms1_rts[ind], 
                                        self.run.MassCalCenter, MassCalBins)
                if self.run.tims: # return mobility
                    self.ms1[ind, j], exp_ms1_mzs[ind, j], exp_ims[ind, j] = self.run.ms1sp[ms1_left].level(report_mz_im, query_mz, self.run.MassAccuracyMS1, self.IM, self.run.IM_window)
                else:             
                    self.ms1[ind, j], exp_ms1_mzs[ind, j] = self.run.ms1sp[ms1_left].level(query_mz, self.run.MassAccuracyMS1)
            
            for fr in range(self.info.n_frs):
                ms2_rts[ind] = rt
                query_mz = predicted_mz(self.run.MassCorrectionMS2, self.info.products[fr].mz, ms2_rts[ind], 
                                        self.run.MassCalCenter, MassCalBins)
                if self.run.tims: # return mobility
                    self.ms2[ind, fr], exp_ms2_mzs[ind, fr], _ = self.run.ms2sp[i].level(report_mz_im, query_mz, self.run.MassAccuracyMS2, self.IM, self.run.IM_window)
                else:             
                    self.ms2[ind, fr], exp_ms2_mzs[ind, fr] = self.run.ms2sp[i].level(query_mz, self.run.MassAccuracyMS2)
                if self.ms2[ind, fr] < MinXicIntensity:
                    self.ms2[ind, fr] = 0.0
        
        self.ms1_rts, self.ms2_rts = consolidate_by_ms1(ms1_rts, ms2_rts, method='first')
        _, self.ms1 = consolidate_by_ms1(ms1_rts, self.ms1, method='first')
        _, self.ms2 = consolidate_by_ms1(ms1_rts, self.ms2, method='sum')
        _, exp_ms1_mzs = consolidate_by_ms1(ms1_rts, exp_ms1_mzs, method='first')
        _, exp_ms2_mzs = consolidate_by_ms1(ms1_rts, exp_ms2_mzs, method='max')
        _, exp_ims = consolidate_by_ms1(ms1_rts, exp_ims, method='first')
        
        return exp_ms1_mzs, exp_ms2_mzs, exp_ims
    

    def collect_calibration_data(self, apex_index, exp_ims,
                                exp_ms1_mzs, exp_ms2_mzs, 
                                best_ms2_score, mono_ms1_score):
        
        # collect data for mass calibration on the center point of best fragment
        ms1_mz = exp_ms1_mzs[apex_index, 0]
        ms2_mz = exp_ms2_mzs[apex_index, self.info.best_ms2_xic]
        if self.run.tims:
            im = exp_ims[apex_index, 0]
        
        if (best_ms2_score >= MassCalScore and 
            mono_ms1_score >= MassCalScore and 
            ms1_mz > E and ms2_mz > E and
            self.ms2[apex_index, self.info.best_ms2_xic] >= MinApexIntensity):
            self.info.mass_delta_ms1 = ms1_mz - self.prec.mz
            self.info.mass_delta_ms2 = ms2_mz - self.info.best_ms2_mz
            if self.run.tims:
                self.info.im_delta = im - self.IM
        else:
            self.info.mass_delta_ms1 = 0.0
            self.info.mass_delta_ms2 = 0.0
            if self.run.tims:
                self.info.im_delta = 0.0
            
    
    def get_optimal_xic(self, stage, n_xic_points):
        # for stage = 0, extract info for mass calibration
        # for stage = 1, perform quantification and export XIC
        if not self.fInit:
            # theoritical MS1 m/zs
            self.info.ms1_mzs = self.prec.mz + (C13delta * np.arange(0, MaxPrIsotopes) / self.prec.charge)
            self.info.n_isotopic_peaks = estimate_isotopes(self.prec.mz * self.prec.charge)
            # get products info and build ms2 scan index
            self.prepare_products(MinFr, MaxFr)
            if not self.info.n_frs:
                return
            self.build_index()
            self.fInit = 1
            
        self.info.clear()
        # determine initial peak based on RT
        target_ms2_RTs = self.run.ms2_RT[self.target_ms2_index]
        initial_apex = min(np.argmin(np.abs(target_ms2_RTs - self.RT)), 
                           len(self.target_ms2_index) - 3)
           
        # set XIC extraction boundaries 
        R = int(n_xic_points // 2)
        start = max(0, initial_apex - R)
        stop = min(len(self.target_ms2_index) - 2, initial_apex + R)
        
        # extract chromatograms
        report_mz_im = 1 if stage == 0 else 0
        exp_ms1_mzs, exp_ms2_mzs, exp_ims = self.extract_xic(start, stop + 1, report_mz_im)
        self.ms1_i = interpolate_ms1(self.ms1_rts, self.ms1, self.ms2_rts) # MS1 data interpolation
        
        # find optimal xic
        (corr_scores, 
         shape_scores, 
         intensity_scores, 
         scores, 
         optimal_xic, 
         optimal_xic_type, 
         elution_apex, fwhm) = identify_optimal_xic(
                                         self.ms1_i, 
                                         self.ms2, 
                                         apex=self.ms2.shape[0] // 2, 
                                         radius=CorrRadius, 
                                         corr_method=CorrMethod, 
                                         shape_weighted=WeightByShape,
                                         intensity_weighted=WeightByIntensity,
                                         corr_weight=CorrWeight,
                                         shape_weight=ShapeWeight,
                                         intensity_weight=IntensityWeight,
                                         optimal_xic_source=OptimalXicSource,
                                         min_shape_score=MinShapeScore,
                                         min_apex_intensity=MinApexIntensity,
                                         min_apex_evidence=MinApexEvidence)
        self.info.scores['corr_score'] = corr_scores
        self.info.scores['shape_score'] = shape_scores
        self.info.scores['intensity_score'] = intensity_scores
        self.info.scores['score'] = scores
        self.info.optimal_xic = optimal_xic
        self.info.optimal_xic_type = optimal_xic_type
        
        ms2_scores = scores[len(self.info.ms1_mzs):]
        best_fr = np.argmax(ms2_scores[0:self.info.n_frs])
        self.info.best_ms2_xic = best_fr
        self.info.best_ms2_mz = self.info.products[best_fr].mz
        self.info.apex = elution_apex
        self.info.fwhm = fwhm
        
        # if stage = 0, exit the function
        if stage == 0:
            self.collect_calibration_data(elution_apex, 
                                          exp_ims,
                                          exp_ms1_mzs, 
                                          exp_ms2_mzs, 
                                          ms2_scores[best_fr], 
                                          scores[0])
            return
        
        # save XICs
        self.save_xics()
        

    def save_xics(self):
   
        xg = XICGroup()

        xg.run = self.run.run_index
        xg.pr = self.prec.index
        xg.id = self.run.lib.entries[self.prec.index].name
        
        xg.info.ms1_mzs = self.info.ms1_mzs
        xg.info.products = self.info.products
        xg.info.prodi = self.info.prodi
        
        xg.info.apex = self.info.apex
        xg.info.fwhm = self.info.fwhm
        
        xg.info.n_isotopic_peaks = self.info.n_isotopic_peaks
        xg.info.n_frs = self.info.n_frs
        
        xg.info.scores = self.info.scores
        xg.info.optimal_xic = self.info.optimal_xic
        xg.info.optimal_xic_type = self.info.optimal_xic_type
        xg.info.best_ms2_xic = self.info.best_ms2_xic
        xg.info.best_ms2_mz = self.info.best_ms2_mz
        
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
        
        self.ms1_RT = None
        self.ms2_RT = None
        
        self.RT_min = 0.0
        self.RT_max = 0.0
        
        
        self.ms2_cycle = []
        self.n_cycle = 0
        self.n_ms2_per_cycle = 0
        self.window_settings = {}
        
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
    
    
    def load_prec_rts(self, df):
        if self.tims:         
            logging.info("Loading precursor RT and IM")
        else:
            logging.info("Loading precursor RT")
            
        for entry in self.entries:
            entry.mass_cal = df.loc[entry.name, 'mass_cal']
            run_RT = df.loc[entry.name, self.run_name+'_RT']
            if not np.isnan(run_RT):
                entry.RT = run_RT
            if self.tims:
                run_IM = df.loc[entry.name, self.run_name+'_IM']
                if not np.isnan(run_IM):
                    entry.IM = run_IM
    
    
    def process(self):
        self.calibrate_mass()
        logging.info("Identifying the optimal XIC for each precursor")
        for entry in progress_bar(self.entries, args.show_progress, args.inter_progress):   
            entry.get_optimal_xic(stage=1, n_xic_points=nXICPoints)
        self.export_xics()


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
            logging.info(f"Using XIC extraction mass accuracy calculated by 'Mean + 3*SD + 5': {self.MassAccuracyMS1*1e6}, {self.MassAccuracyMS2*1e6} ppm (MS1, MS2)")
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
        for entry in self.entries:   
            if entry.mass_cal == 1:
                entry.get_optimal_xic(stage=0, n_xic_points=nXICPoints)
        
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
                    mass_cali_data['theoretical_mz_ms2'].append(entry.info.best_ms2_mz)
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

        if save_cal_info:
            out_cal_dir = out_int_dir / 'mass_calibration'
            out_cal_dir.mkdir(parents=True, exist_ok=True)
            im_cali_data.write_csv(out_cal_dir / (self.run_name + "_im_error.csv"))
            im_error_plot(im_cali_data, run_name=self.run_name, out_dir=out_cal_dir)
        
        
    def export_xics(self):
        ms1_intensity_cols = [f'ms1_intensity_M{i}' for i in range(MaxPrIsotopes)]
        ms2_intensity_cols = [f'ms2_intensity_fr{i}' for i in range(MaxFr)]
        ms1_interpol_cols = [f'ms1_interpol_M{i}' for i in range(MaxPrIsotopes)]
        column_names = ['Precursor.Id', 'row_type', 'ms1_rt', 'ms2_rt'] + ms1_intensity_cols + ms2_intensity_cols + ms1_interpol_cols
        
        data_rows = []
        
        for xic_group in self.XIC_list:
            ms1_mzs = xic_group.info.ms1_mzs.tolist()
            ms1_indices = [xic_group.pr] * len(ms1_mzs)
            ms2_mzs = [fr.mz for fr in xic_group.info.products]
            ms2_mzs += [0.0] * (MaxFr - len(ms2_mzs))
            ms2_indices = xic_group.info.prodi
            ms2_indices += [0.0] * (MaxFr - len(ms2_indices))
            
            optimal_xic = xic_group.info.optimal_xic
            optimal_xic_type = xic_group.info.optimal_xic_type
            n_isotopic_peaks = xic_group.info.n_isotopic_peaks
            n_frs = xic_group.info.n_frs
            
            apex = xic_group.info.apex
            fwhm = xic_group.info.fwhm
            
            # mz_row: use 0.0 placeholders for rt, plus best fragment mz at the end
            index_row = [xic_group.id, 'index_row'] + [optimal_xic, optimal_xic_type] + ms1_indices + ms2_indices + ms1_indices
            mz_row = [xic_group.id, 'mz_row'] + [n_isotopic_peaks, n_frs] + ms1_mzs + ms2_mzs + ms1_mzs
            score_row = [xic_group.id, 'score_row'] + [apex, fwhm] + xic_group.info.scores['score'].tolist() + [0.0] * len(ms1_mzs)
            data_rows.append(index_row)
            data_rows.append(mz_row)
            data_rows.append(score_row)
                     
            # Prepare original and corrected xics
            xic_row_list = []
            for i, row in enumerate(xic_group.xics):
                xic_row_list.append([xic_group.id, 'xic_row'] + list(row))
            data_rows.extend(xic_row_list)
    
        # Create Polars DataFrame
        df = pl.DataFrame(data_rows, schema=column_names, orient="row")
    
        # Determine output file name
        out_xic = out_int_dir / "xic_files" / f"{self.run_name}.xic.parquet"
    
        # Save DataFrame with error handling
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
    
        self.XIC_list.clear()

            