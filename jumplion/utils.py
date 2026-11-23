# builtin
import os
import re
import math
import logging
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
MassCalScore = 0.8

# XIC evaluation
InitialRadius = 4
CorrMethod = 0
#MinShapeScore = 0.5
MinMs1Ms2Corr = 0.8
MinXicIntensity = 10
MinApexIntensity = 30
#MinApexEvidence = 0.99

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


def define_species(prot_names):
    if pd.isna(prot_names) or not prot_names.strip():
        return None

    species = set()
    for prot_name in prot_names.split(';'):
        parts = prot_name.strip().split('_')
        if len(parts) > 1:
            species.add(parts[1].strip())

    return ";".join(sorted(species)) if species else None


# print out df in logs
def indent_df(df, spaces=4, index=False):
    """Return a DataFrame string with left indentation for logging."""
    indent = " " * spaces
    df_str = df.to_string(index=index)
    return "\n".join(indent + line for line in df_str.splitlines())


def logging_histogram(count_data, max_count, n_bins=20, max_bar_width=20):
    n_bins = n_bins if max_count > n_bins else max_count - 1
    n_bins = max(n_bins, 1)

    data_min, data_max = int(np.min(count_data)), int(np.max(count_data))
    bin_edges = np.linspace(data_min, data_max + 1, n_bins + 1)
    bin_edges = np.unique(np.round(bin_edges).astype(int))

    counts, bin_edges = np.histogram(count_data, bins=bin_edges)
    max_count = counts.max()

    for i in range(len(counts)):
        bar_len = int((counts[i] / max_count) * max_bar_width) if max_count > 0 else 0
        bar = '#' * bar_len
        right_bracket = ']' if i == len(counts) - 1 else ')'
        logging.info(f"[{bin_edges[i]:5d}, {bin_edges[i+1]:5d}{right_bracket} | {bar} ({counts[i]})")


def estimate_isotopes(peptide_mass, threshold=0.1):
    """
    Estimate the isotopic envelope of a peptide using a Poisson/
    averagine approximation.

    Parameters
    ----------
    peptide_mass : float
        Peptide mass in Da.
    threshold : float
        Relative intensity cutoff relative to the most intense
        isotope peak. Default = 0.1 (10%).

    Returns
    -------
    n_peaks : int
        Number of peaks from M through the last isotope peak
        above the threshold.

    rel_intensities : list of float
        Relative intensities of M, M+1, ..., normalized so that
        the most intense isotope peak = 1.0.
    """
    if peptide_mass <= 0:
        raise ValueError("peptide_mass must be > 0")

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be > 0 and <= 1")

    # Mean heavy-isotope count from averagine approximation
    lam = 0.00052 * peptide_mass

    # Poisson mode
    mode = int(math.floor(lam))

    # log(P(k)) except for the constant -lambda, which cancels
    # when calculating relative probabilities.
    def log_prob(k):
        if lam == 0:
            return 0.0 if k == 0 else -math.inf
        return k * math.log(lam) - math.lgamma(k + 1)

    # Most intense isotope
    log_pmax = log_prob(mode)

    rel_intensities = []

    k = 0

    while True:
        # Relative intensity compared with the Poisson mode
        log_rel = log_prob(k) - log_pmax
        rel = math.exp(log_rel)

        rel_intensities.append(rel)

        # We can only stop AFTER passing the mode.
        # Before the mode, intensities may still be increasing.
        if k > mode and rel < threshold:
            # Remove first peak below threshold
            rel_intensities.pop()
            break

        k += 1

    n_peaks = len(rel_intensities)

    return n_peaks, rel_intensities


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
