# builtin
import os
import re
import math
from typing import List
# external
from tqdm import tqdm
import numpy as np
import pandas as pd
from numba import njit

# 
MSFormatsExt = {"mzml", "raw", "d"}  
MaxLibFrNum = 20

# Mass calibration
MassCalBins = 1
MassCalBinsMax = 1
MassCalScore = 0.9

# Optimal XIC
CorrRadius = 5
CorrMethod = 0
WeightByShape = 1
WeightByIntensity = 1
CorrWeight = 0.5
ShapeWeight = 0.2
IntensityWeight = 0.3
MinShapeScore = 0.3
MinXicIntensity = 10
MinApexIntensity = 30
MinApexEvidence = 0.99

#
E = np.float64(0.000000001)
INF = np.float64(10000000.0)

#
C13delta = 1.003355
proton = 1.007276
OH = 17.00273965

#
fTypeB = 1 << 0      # 00000001 (binary) or 1 (decimal)
fTypeY = 1 << 1      # 00000010 (binary) or 2 (decimal)
char_of_fType = ['?', 'b', 'y']
fExclude = 1 << 6   

# 
name_of_loss = ["noloss", "H2O", "NH3", "CO", "unknown", "unknown"]
loss_none, loss_H2O, loss_NH3, loss_CO, loss_N, loss_other = np.arange(6)
Loss = np.array([0.0, 18.011113035, 17.026548, 27.994915, 0.0, 0.0], dtype=np.float32)


def estimate_isotopes(peptide_mass, threshold=0.1):
    """
    Estimate number of observable isotopic peaks (>= threshold relative intensity).
    
    Parameters
    ----------
    peptide_mass : float
        Peptide mass in Da.
    threshold : float
        Relative intensity cutoff (default 0.1 = 10%).
    
    Returns
    -------
    n_peaks : int
        Number of observable peaks (M to M+n).
    """
    # Mean heavy isotopes using averagine model (C/N contributions)
    lam = 0.00052 * peptide_mass
    
    # Poisson mode (most probable isotope count)
    mode = int(lam)
    max_prob = math.exp(-lam) * lam**mode / math.factorial(mode)
    
    # Always include mono isotopic peak
    n_peaks = 1
    k = 1
    while True:
        prob = math.exp(-lam) * lam**k / math.factorial(k)
        rel_intensity = prob / max_prob
        if rel_intensity >= threshold:
            n_peaks = k + 1  # count peaks from M to M+k
            k += 1
        else:
            break
    return n_peaks


def consolidate_by_ms1(indices, data, method='sum'):
    """
    Consolidate data array based on duplicate values in indices array.
    
    Parameters:
    -----------
    indices : array-like
        Array containing potentially duplicate index values
    data : array-like
        1-D or 2-D array to consolidate (must have same length as indices in first dimension)
    method : str
        'sum' - sum values at duplicate indices
        'first' - keep first occurrence of duplicate indices
    
    Returns:
    --------
    unique_indices : array
        Unique index values
    consolidated_data : array
        Consolidated data array
    """
    indices = np.array(indices)
    data = np.array(data)
    
    # Find unique indices and their inverse mapping
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    has_duplicates = len(unique_indices) < len(indices)
    if not has_duplicates:
        return indices, data
    
    if method == 'sum':
        # Sum values at duplicate indices
        if data.ndim == 1:
            consolidated = np.zeros(len(unique_indices))
            for i, val in enumerate(data):
                consolidated[inverse[i]] += val
        else:  # 2-D array
            consolidated = np.zeros((len(unique_indices), data.shape[1]))
            for i, row in enumerate(data):
                consolidated[inverse[i]] += row
    elif method == 'max':
        # Keep maximum value at duplicate indices
        if data.ndim == 1:
            consolidated = np.full(len(unique_indices), -np.inf)
            for i, val in enumerate(data):
                consolidated[inverse[i]] = max(consolidated[inverse[i]], val)
        else:  # 2-D array
            consolidated = np.full((len(unique_indices), data.shape[1]), -np.inf)
            for i, row in enumerate(data):
                consolidated[inverse[i]] = np.maximum(consolidated[inverse[i]], row)
    elif method == 'first':
        # Keep first occurrence
        if data.ndim == 1:
            consolidated = np.zeros(len(unique_indices))
            seen = set()
            for i, val in enumerate(data):
                if inverse[i] not in seen:
                    consolidated[inverse[i]] = val
                    seen.add(inverse[i])
        else:  # 2-D array
            consolidated = np.zeros((len(unique_indices), data.shape[1]))
            seen = set()
            for i, row in enumerate(data):
                if inverse[i] not in seen:
                    consolidated[inverse[i]] = row
                    seen.add(inverse[i])
    else:
        raise ValueError("method must be 'sum', 'first', or 'max'")
    
    return unique_indices, consolidated


@njit
def weighted_average_mz(mzs, intensities):
    """
    Calculate the weighted average of mzs based on intensities.
    
    Args:
        mzs: NumPy array of numerical mzs
        intensities: NumPy array of intensities corresponding to each value
    
    Returns:
        Tuple of (total_intensities, weighted_average)
    """
    # Numba doesn't support exceptions, so we handle edge cases differently
    if len(mzs) != len(intensities):
        return 0.0, 0.0  # Return zeros for error case
    
    if len(mzs) == 0:
        return 0.0, 0.0  # Return zeros for empty arrays
    
    # Calculate weighted sum and total weight
    weighted_sum = 0.0
    total_intensities = 0.0
    
    for i in range(len(mzs)):
        weighted_sum += mzs[i] * intensities[i]
        total_intensities += intensities[i]
    
    if total_intensities == 0.0:
        return 0.0, 0.0
    
    return total_intensities, weighted_sum / total_intensities


def generate_group_indices(sample_names: List[str], 
                           sample_meta: pd.DataFrame, 
                           sample_column: str = 'Sample', 
                           group_column: str = 'Group') -> List[List[int]]:
    """
    Generate group indices from sample names and sample meta.
    
    Parameters:
    -----------
    sample_names : List[str]
        List of sample names in the order they appear in the matrix
    sample_meta : pd.DataFrame
        DataFrame with sample names and their corresponding groups
    sample_column : str
        Name of the column in sample_meta that contains sample names
    group_column : str
        Name of the column in sample_meta that contains group assignments
        
    Returns:
    --------
    List[List[int]]
        List of lists, where each inner list contains column indices for a group
    """
    # Create a mapping from column name to its index in the matrix
    col_name_to_index = {name: idx for idx, name in enumerate(sample_names)}
    
    # Create a dictionary to store groups
    group_dict = {}
    
    # Process each row in the group DataFrame
    for _, row in sample_meta.iterrows():
        col_name = row[sample_column]
        group = row[group_column]
        
        # Check if column name exists in our matrix
        if col_name in col_name_to_index:
            if group not in group_dict:
                group_dict[group] = []
            group_dict[group].append(col_name_to_index[col_name])
        else:
            print(f"Warning: Column '{col_name}' not found in matrix columns")
    
    # Convert to list of lists (sorted by group keys for consistency)
    group_indices = [group_dict[group] for group in sorted(group_dict.keys())]
    
    return group_indices


def missing_value_imputation(
        data: np.ndarray, 
        sample_names, 
        sample_meta,
        sample_column, 
        group_column,
        fill_method='global_min',
        fill_factor=1.0) -> np.ndarray:
    """
    Perform missing value imputation on a numpy array.
    For each row and each group, impute missing values with group minimum (across all rows)
    Parameters:
    -----------
    data : np.ndarray
        2D numpy array with missing values (NaN)
    sample_names : array-like
        Sample names corresponding to columns
    sample_meta : DataFrame or dict
        Metadata for samples
    sample_column : str
        Column name in sample_meta to match sample_names
    group_column : str
        Column name in sample_meta for grouping
    fill_method : str, optional
        Method for imputation: 'global_min', 'global_mean', 'local_min', or 'local_mean'
        Default is 'global_min'
    fill_factor : float, optional
        Multiplication factor for fill value. Default is 1.0
        
    Returns:
    --------
    np.ndarray
        Imputed array
    """
    # Create a copy to avoid modifying original data
    imputed_data = data.copy()
    
    # Calculate global statistics
    global_min = np.nanmin(imputed_data)
    global_mean = np.nanmean(imputed_data)
    
    if fill_method == 'global_min':
        imputed_data = np.nan_to_num(imputed_data, nan=fill_factor*global_min)
    
    elif fill_method == 'global_mean':
        imputed_data = np.nan_to_num(imputed_data, nan=fill_factor*global_mean)
    
    elif fill_method in ['local_min', 'local_mean']:
        group_indices = generate_group_indices(sample_names, sample_meta, sample_column, group_column)
        
        # Calculate group statistics across all rows
        group_stats = {}
        for group_idx, col_indices in enumerate(group_indices):
            group_data = data[:, col_indices]
            if np.all(np.isnan(group_data)):
                # Handle case where all values in group are NaN
                group_stats[group_idx] = global_min 
            else:
                # Get minimum or mean value across all rows and columns in this group
                if fill_method == 'local_min':
                    group_stats[group_idx] = np.nanmin(group_data)
                else:  # local_mean
                    group_stats[group_idx] = np.nanmean(group_data)
        
        # Process each row
        for row_idx in range(data.shape[0]):
            # Process each group
            for group_idx, col_indices in enumerate(group_indices):
                # Get values for this group in current row
                group_values = data[row_idx, col_indices]
                # Find missing value positions
                missing_mask = np.isnan(group_values)
                # Impute if there are missing values
                if np.any(missing_mask):  
                    fill_value = group_stats[group_idx]
                    # Fill missing values
                    for i, col_idx in enumerate(col_indices):
                        if missing_mask[i]:
                            imputed_data[row_idx, col_idx] = fill_factor*fill_value
    
    return imputed_data


def progress_bar(iterable, show=True, intermittent=False):
    total = len(iterable)
    disable = not show
    if not intermittent:
        for item in tqdm(iterable, total=total, disable=disable, leave=True):
            yield item
    else:
        steps=5
        percent_points = [int(round(p)) for p in 
                          [i * 100 / (steps - 1) for i in range(1, steps)]]
        milestones = sorted({int(total * p / 100) for p in percent_points})

        with tqdm(total=total, disable=disable, leave=False) as pbar:
            for count, item in enumerate(iterable, 1):
                yield item
                if count in milestones:
                    pbar.n = count
                    pbar.refresh()


def fast_trim(string):
    return string.strip()

def get_file_name(file):
    return os.path.basename(file)

def get_extension(file):
    basename = os.path.basename(file)
    pos = basename.rfind('.')
    if pos == -1:
        return ''
    extension = basename[pos + 1:]
    extension = extension.rstrip('\\')
    return extension.lower()

def remove_extension(file):
    pos = file.rfind('.')
    if pos == -1:
        return file
    else:
        return file[:pos]

def location_to_file_name(loc, temp_folder=""):
    result = loc
    for c in [':', '/', '\\', '.']:
        result = result.replace(c, '_')
    return temp_folder + result

def get_format(value):
    if isinstance(value, (int, np.int32, np.uint8)):
        return 'i'
    elif isinstance(value, (float, np.float32, np.float64)):
        return 'f'
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
        
def extract_unique_parts(strings):
    if not strings:
        return strings
    # Step 1: Split each string by '_' and '-'
    split_strings = [re.split(r'[-_]', s) for s in strings]
    # Step 2: Transpose the list to identify common elements **at the same position**
    transposed = list(zip(*split_strings))
    # Step 3: Identify indices where all elements are identical
    common_indices = {i for i, values in enumerate(transposed) if len(set(values)) == 1}
    # Step 4: Remove elements at common indices
    unique_parts = [
        [value for i, value in enumerate(parts) if i not in common_indices]
        for parts in split_strings
    ]
    # Step 5: Reconstruct the unique strings
    return ['_'.join(parts) for parts in unique_parts]