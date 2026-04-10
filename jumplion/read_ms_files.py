# builtin
import logging
from typing import Tuple
# external
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numba as nb
from numba.typed import List
from numba import njit, types
from numba.experimental import jitclass
import pyteomics.mzml
import timsrust_pyo3
from timsrust_pyo3 import MSLevel
# local
from jumplion.arguments import Arguments
from jumplion.utils import progress_bar
from jumplion.ext.alpharaw_thermo import RawFileReader
from jumplion.ext.alphatims_bruker import get_meta_bruker, get_mobility_mz

args = Arguments()
ShowProgress = args.show_progress
InterProgress = args.inter_progress
TimsCentroid = args.tims_centroid
WeightedCentroid = args.weighted_centroid
MassAccCentroid = args.mass_acc_centroid * 1e-6
IMWindowCentroid = args.im_window_centroid

PeakMzDtype = np.float32
PeakIMDtype = np.float32
PeakIntensityDtype = np.float32

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


def calculate_cycle_time(df):
    '''
    Define cycles based on MS2 window pattern, and calculate cycle time.
    '''
    ms2_data = df[df['MS_type'] == 'MS2']
    
    # Group all MS2 scans by RT
    ms2_groups = ms2_data.groupby('RT').agg({
        'Window_low': lambda x: sorted(x.values),  # Sort windows
        'Window_high': lambda x: sorted(x.values)
    }).reset_index()
    
    # First window 
    first_window_low = tuple(ms2_groups.iloc[0]['Window_low'])
    
    # Find all occurrences of this pattern
    ms2_groups['window_pattern'] = ms2_groups['Window_low'].apply(tuple)
    cycle_matches = ms2_groups[ms2_groups['window_pattern'] == first_window_low]
    
    # Calculate cycle times
    cycle_rts = cycle_matches['RT'].values
    cycle_time = np.mean(np.diff(cycle_rts[1:]))
    cycle1_start = df[df['RT'] == cycle_rts[1]].index[0]
    cycle2_start = df[df['RT'] == cycle_rts[2]].index[0]
    
    return cycle_time, cycle1_start, cycle2_start


def draw_acquisition_plot(df, cycle_time, cycle1_start, cycle2_start, out_dir, save_fig=False, figsize=(10, 10)):
    
    # cycle start
    if df.loc[cycle1_start-1]['MS_type'] == 'MS1':
        cycle_start_idx = cycle1_start-1
    else:
        cycle_start_idx = cycle1_start
    # cycle end
    if df.loc[cycle2_start-1]['MS_type'] == 'MS1':
        cycle_end_idx = cycle2_start-2
    else:
        cycle_end_idx = cycle2_start-1

    one_cycle = df.loc[cycle_start_idx:cycle_end_idx].copy().reset_index(drop=True)
    
    ms1_count = len(one_cycle[one_cycle['MS_type'] == 'MS1'])
    ms2_count = len(one_cycle[one_cycle['MS_type'] == 'MS2'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate rectangle boundaries
    # Each rectangle spans from its RT to the next scan's RT
    rt_values = one_cycle['RT'].values
    ms1_indices = one_cycle[one_cycle['MS_type'] == 'MS1'].index
    ms2_indices = one_cycle[one_cycle['MS_type'] == 'MS2'].index

    ms1_rt_width = np.median(rt_values[ms1_indices+1] - rt_values[ms1_indices])
    ms2_rt_diffs = rt_values[ms2_indices[:-1]+1] - rt_values[ms2_indices[:-1]]
    ms2_rt_width = np.median(ms2_rt_diffs[ms2_rt_diffs != 0])
    
    # Plot all rectangles
    for i, (idx, row) in enumerate(one_cycle.iterrows()):
        height = row['Window_high'] - row['Window_low']
        
        if row['MS_type'] == 'MS1':
            # MS1 scan - blue
            rect = Rectangle((rt_values[i], row['Window_low']), 
                             ms1_rt_width, height,
                             linewidth=1, edgecolor='darkblue', 
                             facecolor='cornflowerblue', alpha=0.8)
        else:
            # MS2 scan - red
            rect = Rectangle((rt_values[i], row['Window_low']), 
                             ms2_rt_width, height,
                             linewidth=0.5, edgecolor='darkred', 
                             facecolor='salmon', alpha=0.7)
        
        ax.add_patch(rect)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='cornflowerblue', edgecolor='darkblue', alpha=0.8, label='MS1'),
        Patch(facecolor='salmon', edgecolor='darkred', alpha=0.8, label='MS2')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, -0.2), fontsize=13, frameon=False)
    
    # Set axis labels and title
    ax.set_xlabel('Retention Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('m/z', fontsize=13, fontweight='bold')
    ax.set_title('Data Acquisition Cycle', 
                 fontsize=15, fontweight='bold', pad=10)
    
    # Set axis limits
    rt_min, rt_max = one_cycle['RT'].min(), one_cycle['RT'].max()
    mz_min = one_cycle['Window_low'].min()
    mz_max = one_cycle['Window_high'].max()
    
    ax.set_xlim(rt_min - cycle_time * 0.05, rt_max + ms2_rt_width + cycle_time * 0.05)
    ax.set_ylim(mz_min - 50, mz_max + 50)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add text annotation
    ax.text(0.05, -0.2, 
            f'Cycle time: {cycle_time*60:.2f} s\nTotal scans: {len(one_cycle)}\nMS1: {ms1_count}, MS2: {ms2_count}', 
            transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', edgecolor=None, facecolor='white', alpha=0))
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(out_dir / 'data_acquisition_cycle.png', dpi=300, bbox_inches='tight')


def get_cycle_time(ms_file, save_supplements=True, out_dir=''):
    if ms_file.endswith('.d'):
        (meta_data, 
         ms1_frames_df, 
         ms2_frames_df, 
         isolation_windows) = get_meta_bruker(ms_file, False, False, False)
        # get ms2 scans
        ms2_scans = ms2_frames_df.loc[0:1000, ['Time', 
                                               'quad_low_mz', 
                                               'quad_high_mz']]
        ms2_scans.rename(columns={'Time': 'RT', 
                                  'quad_low_mz': 'Window_low',
                                  'quad_high_mz': 'Window_high'},
                         inplace=True)
        ms2_scans['MS_type'] = 'MS2'
        # get ms1 scans
        ms1_scans = pd.DataFrame({'RT': ms1_frames_df.loc[ms1_frames_df['Time'] < ms2_scans['RT'].max(), 'Time']})
        ms1_scans['Window_low'] = ms2_scans['Window_low'].min()
        ms1_scans['Window_high'] = ms2_scans['Window_high'].max()
        ms1_scans['MS_type'] = 'MS1'
        # combine ms1 and ms2
        eg1000 = pd.concat([ms1_scans, ms2_scans])
        eg1000['RT'] = eg1000['RT'] / 60
    elif ms_file.endswith('.raw'):
        rawfile = RawFileReader(ms_file)
        ms1_rts = []
        ms2_rts = []
        ms2_lows = []
        ms2_highs = []
        for i in range(1, 1001):
            if rawfile.GetMSOrderForScanNum(i) == 1:
                ms1_rts.append(rawfile.RTFromScanNum(i))
            elif rawfile.GetMSOrderForScanNum(i) == 2:
                scan_event = rawfile.GetScanEventForScanNum(i)
                isolation_center = scan_event.GetReaction(0).PrecursorMass
                isolation_width = scan_event.GetIsolationWidth(0)
                ms2_rts.append(rawfile.RTFromScanNum(i))
                ms2_lows.append(float(isolation_center - isolation_width / 2))
                ms2_highs.append(float(isolation_center + isolation_width / 2))
        # get ms2 scans
        ms2_scans = pd.DataFrame({'RT': ms2_rts,
                                  'Window_low': ms2_lows,
                                  'Window_high': ms2_highs})
        ms2_scans['MS_type'] = 'MS2'
        # get ms1 scans
        ms1_scans = pd.DataFrame({'RT': ms1_rts})
        ms1_scans['Window_low'] = ms2_scans['Window_low'].min()
        ms1_scans['Window_high'] = ms2_scans['Window_high'].max()
        ms1_scans['MS_type'] = 'MS1'
        # combine ms1 and ms2
        eg1000 = pd.concat([ms1_scans, ms2_scans])
    elif ms_file.endswith('.mzML'):
        mzML = pyteomics.mzml.MzML(ms_file, use_index=True)
        ms1_rts = []
        ms2_rts = []
        ms2_lows = []
        ms2_highs = []
        for i in range(0, 1000):
            sp = mzML[i]
            rt = np.float32(sp['scanList']['scan'][0]['scan start time'])
            if sp["ms level"] == 1:
                ms1_rts.append(rt)
            elif sp["ms level"] == 2:
                ms2_rts.append(rt)
                isolation_center = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z'].real
                lower_offset = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset'].real
                upper_offset = sp['precursorList']['precursor'][0]['isolationWindow']['isolation window upper offset'].real
                ms2_lows.append(float(isolation_center - lower_offset))
                ms2_highs.append(float(isolation_center + upper_offset))
        # get ms2 scans
        ms2_scans = pd.DataFrame({'RT': ms2_rts,
                                  'Window_low': ms2_lows,
                                  'Window_high': ms2_highs})
        ms2_scans['MS_type'] = 'MS2'
        # get ms1 scans
        ms1_scans = pd.DataFrame({'RT': ms1_rts})
        ms1_scans['Window_low'] = ms2_scans['Window_low'].min()
        ms1_scans['Window_high'] = ms2_scans['Window_high'].max()
        ms1_scans['MS_type'] = 'MS1'
        # combine ms1 and ms2
        eg1000 = pd.concat([ms1_scans, ms2_scans])
    eg1000.sort_values('RT', inplace=True, ignore_index=True)
    cycle_time, cycle1_start, cycle2_start = calculate_cycle_time(eg1000)
    if save_supplements:
        eg1000.to_csv(out_dir / "data_acquisition_first1000scans.csv", index=False)
        draw_acquisition_plot(eg1000, cycle_time, cycle1_start, cycle2_start, save_fig=True, figsize=(4,5), out_dir=out_dir)
    
    return cycle_time


def load_thermo_raw(raw_file_path: str, centroided: bool = True):
    rawfile = RawFileReader(raw_file_path)
    first_spectrum_number = rawfile.FirstSpectrumNumber
    last_spectrum_number = rawfile.LastSpectrumNumber

    ms1_spectra = List()
    ms2_spectra = List()
    for i in progress_bar(range(first_spectrum_number, last_spectrum_number + 1), 
                          ShowProgress, 
                          InterProgress):
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
        spec.peaks_mz = masses.astype(PeakMzDtype)
        spec.peaks_intensity = intensities.astype(PeakIntensityDtype)

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
    mobility_scan_max_index = int(ms1_frames_df['NumScans'].max())
    mobility_values, mz_values = get_mobility_mz(bruker_d_folder_name, 
                                                 meta_data,
                                                 mobility_scan_max_index)
    
    reader = timsrust_pyo3.FrameReader(bruker_d_folder_name)
    frames = reader.read_all_frames()
    del reader
    
    ms1_spectra = np.empty(len(ms1_frames_df), dtype=object)
    ms2_spectra = np.empty(len(ms2_frames_df), dtype=object)
    
    ms1_idx = 0
    ms2_idx = 0
    rt_conversion = 1.0 / 60.0
    for i, frame in enumerate(progress_bar(frames, ShowProgress, InterProgress)):
        mzs = mz_values[frame.tof_indices].astype(PeakMzDtype)
        ims = np.repeat(mobility_values, np.diff(frame.scan_offsets)).astype(PeakIMDtype)
        intensities = np.array(frame.intensities, dtype=PeakIntensityDtype)
        rt = np.float32(frame.rt * rt_conversion)
        
        if frame.ms_level == MSLevel.MS1:  # MS1 spectra
            spec = timsSpectrum()
            spec.type = np.int8(1)
            spec.RT = rt
            
            mzs_cen, ims_cen, intensities_cen = centroid_and_sort(
                mzs, ims, intensities, 
                centroid=TimsCentroid,
                use_weighted_mean=WeightedCentroid,
                mz_rel_threshold=MassAccCentroid, 
                im_abs_threshold=IMWindowCentroid)
            
            spec.n = len(mzs_cen)
            spec.peaks_mz = mzs_cen
            spec.peaks_im = ims_cen
            spec.peaks_intensity = intensities_cen
            
            ms1_spectra[ms1_idx] = spec
            ms1_idx += 1
        elif frame.ms_level == MSLevel.MS2:  # MS2 spectra
            quad_settings = frame.quadrupole_settings
            scan_offsets = np.asarray(frame.scan_offsets)
            starts = scan_offsets[quad_settings.scan_starts]
            ends = scan_offsets[quad_settings.scan_ends]
            #
            isolation_width_half = np.array(quad_settings.isolation_width) * 0.5
            mz_window_low = quad_settings.isolation_mz - isolation_width_half
            mz_window_high = quad_settings.isolation_mz + isolation_width_half

            for j in range(len(starts)):
                spec = timsSpectrum()
                spec.type = np.int8(2)
                spec.RT = rt
                
                start_idx, end_idx = starts[j], ends[j]
                mzs_sub = mzs[start_idx:end_idx]
                ims_sub = ims[start_idx:end_idx]
                intensities_sub = intensities[start_idx:end_idx]
                
                # Skip empty spectra
                if len(mzs_sub) == 0:
                    continue
                
                mzs_cen, ims_cen, intensities_cen = centroid_and_sort(
                    mzs_sub, 
                    ims_sub, 
                    intensities_sub, 
                    centroid=TimsCentroid,
                    use_weighted_mean=WeightedCentroid,
                    mz_rel_threshold=MassAccCentroid, 
                    im_abs_threshold=IMWindowCentroid)
                
                spec.n = len(mzs_cen)
                spec.peaks_mz = mzs_cen
                spec.peaks_im = ims_cen
                spec.peaks_intensity = intensities_cen
                
                spec.window_low = float(mz_window_low[j])
                spec.window_high = float(mz_window_high[j])
                ms2_spectra[ms2_idx] = spec
                ms2_idx += 1
    
        frames[i] = None
    frames.clear()
    del frames
    return ms1_spectra[:ms1_idx], ms2_spectra[:ms2_idx]

@njit
def centroid_and_sort(mzs, ims, intensities, 
                      centroid=0, 
                      use_weighted_mean=0,
                      mz_rel_threshold=5e-6, 
                      im_abs_threshold=0.005):
    """
    Pre-sort by mz for spatial locality + early termination
    """    
    # Sort by mz first for spatial locality
    mz_indices = np.argsort(mzs)
    n = len(mzs)
    
    if not centroid or n==0:
        return mzs[mz_indices], ims[mz_indices], intensities[mz_indices]

    # Get intensity order within mz-sorted data
    sorted_intensities = intensities[mz_indices]
    intensity_order = np.argsort(-sorted_intensities)
    
    used = np.zeros(n, dtype=nb.boolean)
    
    # Pre-allocate result arrays
    mzs_result = np.empty(n, dtype=np.float32)
    ims_result = np.empty(n, dtype=np.float32)
    intensities_result = np.empty(n, dtype=np.float32)
    result_count = 0
    
    # Process in intensity order within mz-sorted data
    for idx in range(n):
        i_sorted = intensity_order[idx]
        i = mz_indices[i_sorted]
        
        if used[i_sorted]:
            continue
        
        # Cache frequently used values
        current_mz = mzs[i]
        current_im = ims[i]
        current_intensity = intensities[i]
        
        # Initialize accumulators
        mz_weighted_sum = current_mz * current_intensity
        im_weighted_sum = current_im * current_intensity
        intensity_sum = current_intensity
        used[i_sorted] = True
        
        mz_threshold_abs = current_mz * mz_rel_threshold
        
        # Search nearby points in mz-sorted order (both directions)
        # Search left (lower mz)
        for k in range(i_sorted - 1, -1, -1):
            if used[k]:
                continue
            j = mz_indices[k]
            if current_mz - mzs[j] > mz_threshold_abs:
                break  # Early termination
                
            if abs(ims[j] - current_im) < im_abs_threshold:
                intensity_j = intensities[j]
                mz_weighted_sum += mzs[j] * intensity_j
                im_weighted_sum += ims[j] * intensity_j
                intensity_sum += intensity_j
                used[k] = True
        
        # Search right (higher mz)
        for k in range(i_sorted + 1, n):
            if used[k]:
                continue
            j = mz_indices[k]
            if mzs[j] - current_mz > mz_threshold_abs:
                break  # Early termination
                
            if abs(ims[j] - current_im) < im_abs_threshold:
                intensity_j = intensities[j]
                mz_weighted_sum += mzs[j] * intensity_j
                im_weighted_sum += ims[j] * intensity_j
                intensity_sum += intensity_j
                used[k] = True
        
        # Store results
        if use_weighted_mean:
            mzs_result[result_count] = mz_weighted_sum / intensity_sum
            ims_result[result_count] = im_weighted_sum / intensity_sum
        else:
            mzs_result[result_count] = current_mz
            ims_result[result_count] = current_im
        intensities_result[result_count] = intensity_sum
        result_count += 1
    
    # Sort results by mz coordinate before returning
    if result_count > 1:
        sort_indices = np.argsort(mzs_result[:result_count])
        return (mzs_result[:result_count][sort_indices], 
                ims_result[:result_count][sort_indices], 
                intensities_result[:result_count][sort_indices])
    else:
        return mzs_result[:result_count], ims_result[:result_count], intensities_result[:result_count]
        