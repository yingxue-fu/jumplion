# builtin
import os
import sys
import json
import contextlib
import multiprocessing
import ctypes
import logging
# external
import numpy as np
import pandas as pd
import sqlite3

EXT_PATH = os.path.join(os.path.dirname(__file__), "bruker")
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