# builtin
import logging
from typing import Tuple
# external
import numpy as np
from numba.typed import List
from numba import types
from numba.experimental import jitclass
import pyteomics.mzml
import timsrust_pyo3
from timsrust_pyo3 import MSLevel
# local
from jumpdiaq.arguments import Arguments
from jumpdiaq.utils import progress_bar
from jumpdiaq.ext.alpharaw_thermo import RawFileReader
from jumpdiaq.ext.alphatims_bruker import get_meta_bruker, get_mobility_mz

args = Arguments()
PEAK_MZ_DTYPE = np.float32
PEAK_IM_DTYPE = np.float32
PEAK_INTENSITY_DTYPE = np.float32

@jitclass([
    ("RT", types.float32), 
    ("window_low", types.float32), 
    ("window_high", types.float32), 
    ("n", types.int64), 
    ("type", types.int8), 
    ("peaks_mz", types.float32[:]),
    ("peaks_intensity", types.float32[:])
])
class Spectrum:
    def __init__(self):
        self.RT: float = 0.0                                 # RT of the Spectrum
        self.window_low: float = 0.0                         # min m/z of Precursor ions
        self.window_high: float = 0.0                        # max m/z of Precursor ions
        self.n: int = 0                                      # number of peaks in the Spectrum
        self.type: int = 0                                   # MS1/MS2 level
        self.peaks_mz: np.ndarray = np.empty(0, dtype=np.float32)         # array of peaks' m/zs
        self.peaks_intensity: np.ndarray = np.empty(0, dtype=np.float32)  # array of peaks' intensities
    
    def size(self) -> int:
        """Returns the number of peaks in the spectrum."""
        return self.n

    def __lt__(self, other: 'Spectrum') -> bool:
        """Compares this spectrum with another based on RT."""
        return self.RT < other.RT

    def has(self, mz: float, margin: float = 0.0) -> bool:
        """checks if a given mz value (with an optional margin) falls within the the window."""
        return self.window_low <= mz + margin and self.window_high > mz - margin

    def level(self, mz: float, accuracy: float) -> Tuple[float, float]:
        """
        Finds the highest intensity peak and corresponding m/z value within a given m/z and accuracy. 
        
        Args:
            mz (float): Target m/z value.
            accuracy (float): Relative accuracy for m/z range.
        
        Returns:
            Tuple[float, float]: (highest intensity, corresponding m/z value)
        
        """
        n_peaks = self.size()
        if n_peaks == 0:
            return (np.float32(0.0), np.float32(0.0))
        
        margin = mz * accuracy
        min_mz = mz - margin
        max_mz = mz + margin

        low = 0
        high = n_peaks
        
        while high > low:
            # Binary search
            middle = (high + low) // 2
            v = self.peaks_mz[middle]

            if v < max_mz:
                if v > min_mz:
                    
                    s = self.peaks_intensity[middle]
                    exp_mz = self.peaks_mz[middle]
                    
                    # find the peak with the highest intensity within the m/z range 
                    # for the right side of middle
                    i = middle + 1
                    while i < high and self.peaks_mz[i] < max_mz:
                        if self.peaks_intensity[i] > s:
                            s = self.peaks_intensity[i]
                            exp_mz = self.peaks_mz[i]
                        i += 1
                    if i < high:
                        high = i
                        
                    # for the left side of middle
                    i = middle - 1
                    while i >= low and self.peaks_mz[i] > min_mz:
                        if self.peaks_intensity[i] > s:
                            s = self.peaks_intensity[i]
                            exp_mz = self.peaks_mz[i]
                        i -= 1
                    if i >= low:
                        low = i + 1
                    
                    return (np.float32(s), np.float32(exp_mz))
                
                low = middle + 1
            else:
                high = middle
        
        return (np.float32(0.0), np.float32(0.0))


@jitclass([
    ("RT", types.float32), 
    ("window_low", types.float32), 
    ("window_high", types.float32), 
    ("n", types.int32), 
    ("type", types.int8), 
    ("peaks_mz", types.float32[:]),
    ("peaks_im", types.float32[:]),
    ("peaks_intensity", types.float32[:])
])
class timsSpectrum:
    def __init__(self):
        self.RT: float = 0.0                                 # RT of the Spectrum
        self.window_low: float = 0.0                         # min m/z of Precursor ions
        self.window_high: float = 0.0                        # max m/z of Precursor ions
        self.n: int = 0                                      # number of peaks in the Spectrum
        self.type: int = 0                                   # MS1/MS2 level
        self.peaks_mz: np.ndarray = np.empty(0, dtype=np.float32)         # array of peaks' m/zs
        self.peaks_im: np.ndarray = np.empty(0, dtype=np.float32)   # array of peaks' ion mobilities
        self.peaks_intensity: np.ndarray = np.empty(0, dtype=np.float32)  # array of peaks' intensities
    
    def size(self) -> int:
        """Returns the number of peaks in the spectrum."""
        return self.n

    def __lt__(self, other: 'timsSpectrum') -> bool:
        """Compares this spectrum with another based on RT."""
        return self.RT < other.RT

    def has(self, mz: float, mz_margin: float = 0.0) -> bool:
        """checks if a given mz value (with an optional margin) falls within the the window."""
        return self.window_low <= mz + mz_margin and self.window_high > mz - mz_margin

    def level(self, report_mz_im: int,
              mz: float, mz_accuracy: float, 
              im: float, im_accuracy: float) -> Tuple[float, float, float]:
        """
        Within the m/z range defined by a given m/z and m/z accuracy, filter peaks in 
        defined ion mobility range, returns the weighted m/z and ion mobility values, 
        and the sum of intensities for all peaks. 
        
        Args:
            mz (float): Target m/z value.
            mz_accuracy (float): Relative mass accuracy.
            
        Returns:
            Tuple[float, float, float]: (total intensity, m/z value, ion mobility value)
        
        """
        n_peaks = self.size()
        if n_peaks == 0:
            return (np.float32(0.0), np.float32(0.0), np.float32(0.0))
        
        mz_margin = mz * mz_accuracy
        min_mz = mz - mz_margin
        max_mz = mz + mz_margin
        
        min_im = im - im_accuracy
        max_im = im + im_accuracy
        
        low = 0
        high = n_peaks
        
        total_intensity = np.float32(0.0)
        total_intensity_mz = np.float32(0.0)
        total_intensity_im = np.float32(0.0)
        while high > low:
            # Binary search
            middle = (high + low) // 2
            v = self.peaks_mz[middle]

            if v < max_mz:
                if v > min_mz:
                    
                    middle_intensity = self.peaks_intensity[middle]
                    middle_mz = self.peaks_mz[middle]
                    middle_im = self.peaks_im[middle]
                    
                    if middle_im > min_im and middle_im < max_im:
                        total_intensity += middle_intensity
                        if report_mz_im:
                            total_intensity_mz += middle_intensity * middle_mz
                            total_intensity_im += middle_intensity * middle_im
                    
                    # find the peak with the highest intensity within the m/z range 
                    # for the right side of middle
                    i = middle + 1
                    while i < high and self.peaks_mz[i] < max_mz:
                        if self.peaks_im[i] > min_im and self.peaks_im[i] < max_im:
                            total_intensity += self.peaks_intensity[i]
                            if report_mz_im:
                                total_intensity_mz += self.peaks_intensity[i] * self.peaks_mz[i]
                                total_intensity_im += self.peaks_intensity[i] * self.peaks_im[i]
                        i += 1
                    if i < high:
                        high = i
                        
                    # for the left side of middle
                    i = middle - 1
                    while i >= low and self.peaks_mz[i] > min_mz:
                        if self.peaks_im[i] > min_im and self.peaks_im[i] < max_im:
                            total_intensity += self.peaks_intensity[i]
                            if report_mz_im:
                                total_intensity_mz += self.peaks_intensity[i] * self.peaks_mz[i]
                                total_intensity_im += self.peaks_intensity[i] * self.peaks_im[i]
                        i -= 1
                    if i >= low:
                        low = i + 1
                    
                    if total_intensity == 0.0:
                        return (np.float32(0.0), np.float32(0.0), np.float32(0.0))
                    else:
                        weighted_mz = total_intensity_mz/total_intensity
                        weighted_im = total_intensity_im/total_intensity
                        return (np.float32(total_intensity), np.float32(weighted_mz), np.float32(weighted_im))
                
                low = middle + 1
            else:
                high = middle
        
        return (np.float32(0.0), np.float32(0.0), np.float32(0.0))
    

def get_cycle_time(ms_file):
    if ms_file.endswith('.d'):
        meta_data, _, _, _ = get_meta_bruker(ms_file, False, False, False)
        cycle_time = meta_data['cycle_time']
    elif ms_file.endswith('.raw'):
        rawfile = RawFileReader(ms_file)
        ms1_rts = []
        for i in range(1, 1000):
            if rawfile.GetMSOrderForScanNum(i) == 1:
                ms1_rts.append(rawfile.RTFromScanNum(i))
        rt_diffs = np.diff(np.array(ms1_rts))    
        cycle_time = np.mean(rt_diffs[1:])
    elif ms_file.endswith('.mzML'):
        pass

    return cycle_time


def load_thermo_raw(raw_file_path: str, centroided: bool = True):
    rawfile = RawFileReader(raw_file_path)
    first_spectrum_number = rawfile.FirstSpectrumNumber
    last_spectrum_number = rawfile.LastSpectrumNumber

    ms1_spectra = List()
    ms2_spectra = List()
    for i in progress_bar(range(first_spectrum_number, last_spectrum_number + 1), 
                          args.show_progress, 
                          args.inter_progress):
        ms_order = rawfile.GetMSOrderForScanNum(i)
        rt = rawfile.RTFromScanNum(i)
        if not centroided:
            masses, intensities = rawfile.GetProfileMassListFromScanNum(i)
        else:
            masses, intensities = rawfile.GetCentroidMassListFromScanNum(i)

        spec = Spectrum()
        spec.type = np.int8(ms_order)
        spec.RT = np.float32(rt)
        spec.n = len(masses)
        spec.peaks_mz = masses.astype(PEAK_MZ_DTYPE)
        spec.peaks_intensity = intensities.astype(PEAK_INTENSITY_DTYPE)

        if spec.type == 1:
            ms1_spectra.append(spec)
        elif spec.type == 2:
            scan_event = rawfile.GetScanEventForScanNum(i)
            isolation_center = scan_event.GetReaction(0).PrecursorMass
            isolation_width = scan_event.GetIsolationWidth(0)
            
            spec.window_low = float(isolation_center - isolation_width / 2)
            spec.window_high = float(isolation_center + isolation_width / 2)
            ms2_spectra.append(spec)
    rawfile.Close()
    
    ms1_spectra = np.array(ms1_spectra, dtype=object)
    ms2_spectra = np.array(ms2_spectra, dtype=object)
    return ms1_spectra, ms2_spectra

# load data from a pyteomics.mzML object and sore it into a list of Spectrum() object
def load_mzml(file_name):
    mzML = pyteomics.mzml.MzML(file_name, use_index=True)
    start = 0
    stop = len(mzML)
    
    other_spectra = False
    ms1_spectra = List()
    ms2_spectra = List()
    for i in range(start, stop):
        sp = mzML[i]

        if sp["ms level"] != 1 and sp["ms level"] != 2:
            other_spectra = True
            continue
        else:
            spec = Spectrum()
            spec.n = len(sp["m/z array"])
            spec.type = np.int8(sp['ms level'])
            spec.RT = np.float32(sp['scanList']['scan'][0]['scan start time'])

            spec.peaks_mz = sp["m/z array"].astype(np.float32)
            spec.peaks_intensity = sp["intensity array"].astype(np.float32)
            
            if sp["ms level"] == 1:
                ms1_spectra.append(spec)
            elif sp["ms level"] == 2:
                middle = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z'].real
                lower_offset = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset'].real
                upper_offset = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window upper offset'].real
                
                spec.window_low = float(middle - lower_offset)
                spec.window_high = float(middle + upper_offset)
            
                ms2_spectra.append(spec)
    
    if other_spectra:
        logging.warning("Spectra other than MS1 and MS2 detected")
    
    ms1_spectra = np.array(ms1_spectra, dtype=object)
    ms2_spectra = np.array(ms2_spectra, dtype=object)
    
    return ms1_spectra, ms2_spectra

def load_timsTOF_d(bruker_d_folder_name):
    meta_data, ms1_frames_df, ms2_frames_df, _ = get_meta_bruker(bruker_d_folder_name, False, False)
    scan_max_index = int(ms1_frames_df.NumScans.max())
    mobility_values, mz_values = get_mobility_mz(bruker_d_folder_name, 
                                                 meta_data,
                                                 scan_max_index)
    
    reader = timsrust_pyo3.FrameReader(bruker_d_folder_name)
    frames = reader.read_all_frames()
    del reader
    
    ms1_spectra = np.empty(len(ms1_frames_df), dtype=object)
    ms2_spectra = np.empty(len(ms2_frames_df), dtype=object)
    
    ms1_idx = 0
    ms2_idx = 0
    for i, frame in enumerate(progress_bar(frames, args.show_progress, args.inter_progress)):
        mzs = mz_values[frame.tof_indices].astype(PEAK_MZ_DTYPE)
        ims = np.repeat(mobility_values, np.diff(frame.scan_offsets)).astype(PEAK_IM_DTYPE)
        intensities = np.array(frame.intensities, dtype=PEAK_INTENSITY_DTYPE)
        rt = frame.rt * (1.0 / 60)
        
        if frame.ms_level == MSLevel.MS1:  # MS1 spectra
            spec = timsSpectrum()
            spec.type = np.int8(1)
            spec.RT = np.float32(rt)
            spec.n = len(mzs)
            
            sort_idx = np.argsort(mzs)
            spec.peaks_mz = mzs[sort_idx]
            spec.peaks_im = ims[sort_idx]
            spec.peaks_intensity = intensities[sort_idx]
            ms1_spectra[ms1_idx] = spec
            ms1_idx += 1
        elif frame.ms_level == MSLevel.MS2:  # MS2 spectra
            quad_settings = frame.quadrupole_settings
            scan_offsets = np.asarray(frame.scan_offsets)
            starts = scan_offsets[quad_settings.scan_starts]
            ends = scan_offsets[quad_settings.scan_ends]
            #
            isolation_width_half = np.array(quad_settings.isolation_width) / 2
            mz_window_low = quad_settings.isolation_mz - isolation_width_half
            mz_window_high = quad_settings.isolation_mz + isolation_width_half

            for j in range(len(starts)):
                spec = timsSpectrum()
                spec.type = np.int8(2)
                spec.RT = np.float32(rt)
                mzs_sub = mzs[starts[j] : ends[j]]
                ims_sub = ims[starts[j] : ends[j]]
                intensities_sub = intensities[starts[j] : ends[j]]
                
                spec.n = len(mzs_sub)
                sort_idx = np.argsort(mzs_sub)
                spec.peaks_mz = mzs_sub[sort_idx]
                spec.peaks_im = ims_sub[sort_idx]
                spec.peaks_intensity = intensities_sub[sort_idx]
                spec.window_low = float(mz_window_low[j])
                spec.window_high = float(mz_window_high[j])
                ms2_spectra[ms2_idx] = spec
                ms2_idx += 1
    
        frames[i] = None
    frames.clear()
    del frames
    return ms1_spectra[:ms1_idx], ms2_spectra[:ms2_idx]
