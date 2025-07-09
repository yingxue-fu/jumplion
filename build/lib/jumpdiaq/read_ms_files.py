# builtin
import json
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
from jumpdiaq.utils import progress_bar

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
    

# load data from a pyteomics.mzML object and sore it into a list of Spectrum() object
def load_mzml_data(file_name):
    
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

# bruker_d_folder_name = 'C:/Users/yfu/Downloads/miss_dia_test_data/raw_files/S13.d'
def get_meta_bruker(bruker_d_folder_name, 
                    show_logs=False,
                    save_meta_data=False, 
                    export_windows=False):
    
    (acquisition_mode, 
     global_meta_data, 
     frames_df, 
     fragment_frames_df, 
     precursors) = read_bruker_sql(bruker_d_folder_name)
    
    meta_data = dict(zip(global_meta_data.Key, global_meta_data.Value))
    ms1_frames_df = frames_df.loc[frames_df['MsMsType'] == 0, :]
    ms2_frames_df = fragment_frames_df.copy()
    
    # calculate isolation windows
    fragment_frames_df['quad_low_mz'] = fragment_frames_df['IsolationMz'] - fragment_frames_df['IsolationWidth'] / 2
    fragment_frames_df['quad_high_mz'] = fragment_frames_df['IsolationMz'] + fragment_frames_df['IsolationWidth'] / 2
    isolation_windows = fragment_frames_df.drop(columns=['Frame']).drop_duplicates(ignore_index=True)
    
    # calculate cycle time
    cycle_times = ms1_frames_df['Time'].diff().reset_index(drop=True)
    cycle_time = np.round(cycle_times[2:len(cycle_times)].mean() / 60, 4)
    
    meta_data['acquisition_mode'] = acquisition_mode
    meta_data['cycle_time'] = cycle_time
    if 'DigitizerSaturationHandling' in meta_data.keys():
        del meta_data['DigitizerSaturationHandling'] 
        
    if show_logs:
        logging.info(f"Data acquisition mode: {acquisition_mode}")
        logging.info(f"Total number of frames: {len(frames_df)} ({len(ms1_frames_df)} MS1 and {len(frames_df)-len(ms1_frames_df)} MS2)")
        logging.info(f"Cycle time: {cycle_time} ({np.round(cycle_time*60, 3)} seconds)")
    
    if save_meta_data:
        with open(bruker_d_folder_name + "_meta_info.json", "w") as f:
            for key, value in meta_data.items():
                json.dump({key: value}, f)
                f.write("\n") 
    
    if export_windows:
        isolation_windows.to_csv(bruker_d_folder_name + "_iso_windows.csv", index=False)
    
    return meta_data, ms1_frames_df, ms2_frames_df, isolation_windows


def get_cycle_time(ms_file):
    if ms_file.endswith('.d'):
        meta_data, _, _, _ = get_meta_bruker(ms_file, False, False, False)
        cycle_time = meta_data['cycle_time']
    elif ms_file.endswith('.raw'):
        pass
    elif ms_file.endswith('.mzML'):
        pass

    return cycle_time


def load_timsTOF_data(bruker_d_folder_name):
    # bruker_d_folder_name = r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\LFQbech_raw\202206012_K2E20_ZI_500ul60C3cm5min_P1-B5_1_9415.d"
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
    for i, frame in enumerate(progress_bar(frames)):
        mzs = mz_values[frame.tof_indices]
        out_ims = np.repeat(mobility_values, np.diff(frame.scan_offsets))
        intensities = np.array(frame.intensities)
        rt = frame.rt * (1.0 / 60)
        
        if frame.ms_level == MSLevel.MS1:  # MS1 spectra
            spec = timsSpectrum()
            spec.type = np.int8(1)
            spec.RT = np.float32(rt)
            spec.n = len(mzs)
            
            sort_idx = np.argsort(mzs)
            spec.peaks_mz = mzs[sort_idx].astype(np.float32)
            spec.peaks_im = out_ims[sort_idx].astype(np.float32)
            spec.peaks_intensity = intensities[sort_idx].astype(np.float32)
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
                ims_sub = out_ims[starts[j] : ends[j]]
                intensities_sub = intensities[starts[j] : ends[j]]
                
                spec.n = len(mzs_sub)
                sort_idx = np.argsort(mzs_sub)
                spec.peaks_mz = mzs_sub[sort_idx].astype(np.float32)
                spec.peaks_im = ims_sub[sort_idx].astype(np.float32)
                spec.peaks_intensity = intensities_sub[sort_idx].astype(np.float32)
                spec.window_low = float(mz_window_low[j])
                spec.window_high = float(mz_window_high[j])
                ms2_spectra[ms2_idx] = spec
                ms2_idx += 1
    
        frames[i] = None
    frames.clear()
    del frames
    return ms1_spectra[:ms1_idx], ms2_spectra[:ms2_idx]


################################## timsTOF data ###############################
import os
import sys
import contextlib
import multiprocessing
import ctypes
import pandas as pd
import sqlite3

BASE_PATH = os.path.dirname(__file__)
EXT_PATH = os.path.join(BASE_PATH, "bruker_ext")
MAX_THREADS = multiprocessing.cpu_count()

# define the path for the bruker .dll file
if sys.platform[:5] == "win32":
    BRUKER_DLL_FILE_NAME = os.path.join(EXT_PATH, "timsdata.dll")
elif sys.platform[:5] == "linux":
    BRUKER_DLL_FILE_NAME = os.path.join(EXT_PATH, "timsdata.so")
else:
    BRUKER_DLL_FILE_NAME = ""


#
def init_bruker_dll(bruker_dll_file_name: str = BRUKER_DLL_FILE_NAME):
    """Open a bruker.dll in Python.

    Five functions are defined for this dll:

        - tims_open: [c_char_p, c_uint32] -> c_uint64
        - tims_close: [c_char_p, c_uint32] -> c_uint64
        - tims_read_scans_v2: [c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32] -> c_uint32
        - tims_index_to_mz: [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32] -> None
        - tims_scannum_to_oneoverk0: Same as "tims_index_to_mz"

    Parameters
    ----------
    bruker_dll_file_name : str
        The absolute path to the timsdata.dll.
        Default is alphatims.utils.BRUKER_DLL_FILE_NAME.

    Returns
    -------
    : ctypes.cdll
        The Bruker dll library.
    """
    import ctypes
    bruker_dll = ctypes.cdll.LoadLibrary(
        os.path.realpath(bruker_dll_file_name)
    )
    bruker_dll.tims_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    bruker_dll.tims_open.restype = ctypes.c_uint64
    bruker_dll.tims_close.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_close.restype = None
    bruker_dll.tims_read_scans_v2.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32
    ]
    bruker_dll.tims_read_scans_v2.restype = ctypes.c_uint32
    bruker_dll.tims_index_to_mz.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_index_to_mz.restype = ctypes.c_uint32
    bruker_dll.tims_scannum_to_oneoverk0.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_scannum_to_oneoverk0.restype = ctypes.c_uint32
    bruker_dll.tims_set_num_threads.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_set_num_threads.restype = None
    bruker_dll.tims_set_num_threads(MAX_THREADS)
    # multiple threads is equally fast as just 1 for io?
    # bruker_dll.tims_set_num_threads(1)
    return bruker_dll


@contextlib.contextmanager
def open_bruker_d_folder(
    bruker_d_folder_name: str,
    bruker_dll_file_name=BRUKER_DLL_FILE_NAME,
) -> tuple:
    """A context manager for a bruker dll connection to a .d folder.

    Parameters
    ----------
    bruker_d_folder_name : str
        The name of a Bruker .d folder.
    bruker_dll_file_name : str, ctypes.cdll
        The path to Bruker' timsdata.dll library.
        Alternatively, the library itself can be passed as argument.
        Default is alphatims.utils.BRUKER_DLL_FILE_NAME,
        which in itself is dependent on the OS.

    Returns
    -------
    : tuple (ctypes.cdll, int).
        The opened bruker dll and identifier of the .d folder.
    """
    try:
        if isinstance(bruker_dll_file_name, str):
            bruker_dll = init_bruker_dll(bruker_dll_file_name)
        #logger.info(f"Opening handle for {bruker_d_folder_name}")
        bruker_d_folder_handle = bruker_dll.tims_open(
            bruker_d_folder_name.encode('utf-8'),
            0
        )
        yield bruker_dll, bruker_d_folder_handle
    finally:
        #logger.info(f"Closing handle for {bruker_d_folder_name}")
        bruker_dll.tims_close(bruker_d_folder_handle)
        
        
def read_bruker_sql(bruker_d_folder_name):
    with sqlite3.connect(
        os.path.join(bruker_d_folder_name, "analysis.tdf")
    ) as sql_database_connection:
        global_meta_data = pd.read_sql_query(
            "SELECT * from GlobalMetaData",
            sql_database_connection
        )
        frames = pd.read_sql_query(
            "SELECT * FROM Frames",
            sql_database_connection
        )
        if 9 in frames.MsMsType.values:
            acquisition_mode = "diaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * FROM DiaFrameMsMsInfo",
                sql_database_connection
            )
            fragment_frame_groups = pd.read_sql_query(
                "SELECT * from DiaFrameMsMsWindows",
                sql_database_connection
            )
            fragment_frames = fragment_frames.merge(
                fragment_frame_groups,
                how="left"
            )
            fragment_frames.rename(
                columns={"WindowGroup": "Precursor"},
                inplace=True
            )
            precursors = None
        elif 8 in frames.MsMsType.values:
            acquisition_mode = "ddaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * from PasefFrameMsMsInfo",
                sql_database_connection
            )
            precursors = pd.read_sql_query(
                "SELECT * from Precursors",
                sql_database_connection
            )
        else:
            acquisition_mode = "noPASEF"
            fragment_frames = pd.DataFrame(
                {
                    "Frame": np.array([0]),
                    "ScanNumBegin": np.array([0]),
                    "ScanNumEnd": np.array([0]),
                    "IsolationWidth": np.array([0]),
                    "IsolationMz": np.array([0]),
                    "Precursor": np.array([0]),
                }
            )
            precursors = None

        frames = pd.DataFrame(
            {
                col: pd.to_numeric(
                    frames[col]
                ) for col in frames if col != "Polarity"
            }
        )

        return (
            acquisition_mode,
            global_meta_data,
            frames,
            fragment_frames,
            precursors
        )
    
    
def get_mobility_mz(bruker_d_folder_name, 
                    meta_data,
                    scan_max_index, 
                    mobility_estimation_from_frame: int = 1,
                    mz_estimation_from_frame: int = 1, 
                    use_calibrated_mz_values_as_default: int = 0):
    bruker_dll_available = BRUKER_DLL_FILE_NAME != ""
    tof_max_index = int(meta_data["DigitizerNumSamples"]) + 1
    mobility_min_value = float(meta_data["OneOverK0AcqRangeLower"])
    mobility_max_value = float(meta_data["OneOverK0AcqRangeUpper"])
    
    if (mobility_estimation_from_frame != 0) and bruker_dll_available:
        with open_bruker_d_folder(
            bruker_d_folder_name
        ) as (bruker_dll, bruker_d_folder_handle):
            indices = np.arange(scan_max_index).astype(np.float64)
            mobility_values = np.empty_like(indices)
            
            bruker_dll.tims_scannum_to_oneoverk0(
                bruker_d_folder_handle,
                mobility_estimation_from_frame,
                indices.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                mobility_values.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                scan_max_index
            )
    else:
        if (mobility_estimation_from_frame != 0):
            logging.warning("Bruker DLL not available, estimating mobility values")
        mobility_values = mobility_max_value - (
            mobility_max_value - mobility_min_value
        ) / scan_max_index * np.arange(scan_max_index)

    mz_min_value = float(meta_data["MzAcqRangeLower"])
    mz_max_value = float(meta_data["MzAcqRangeUpper"])
    
    if meta_data["AcquisitionSoftware"] == "Bruker otofControl":
        logging.warning(
            "WARNING: Acquisition software is Bruker otofControl, "
            "mz min/max values are assumed to be 5 m/z wider than "
            "defined in analysis.tdf!"
        )
        mz_min_value -= 5
        mz_max_value += 5
        
    tof_intercept = np.sqrt(mz_min_value)
    tof_slope = (
        np.sqrt(mz_max_value) - tof_intercept
    ) / tof_max_index
    
    if (mz_estimation_from_frame != 0) and bruker_dll_available:
        with open_bruker_d_folder(
            bruker_d_folder_name
        ) as (bruker_dll, bruker_d_folder_handle):
            indices = np.arange(tof_max_index).astype(np.float64)
            mz_values = np.empty_like(indices)
            bruker_dll.tims_index_to_mz(
                bruker_d_folder_handle,
                mz_estimation_from_frame,
                indices.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                mz_values.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                tof_max_index
            )
    else:
        if (mz_estimation_from_frame != 0):
            logging.warning("Bruker DLL not available, estimating mz values")
        mz_values = (
            tof_intercept + tof_slope * np.arange(tof_max_index)
        )**2
        
    return mobility_values, mz_values