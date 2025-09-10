# builtin
import os
import re
import math
from typing import List
# external
from tqdm import tqdm
import numpy as np
import pandas as pd

#
MSFormatsExt = {"mzml", "raw", "d"}  
MaxLibFrNum = 20

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
proton = 1.007825035
OH = 17.003288

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


def generate_group_indices(column_names: List[str], group_df: pd.DataFrame, 
                          col_name_column: str = 'Sample', 
                          group_column: str = 'Group') -> List[List[int]]:
    """
    Generate group indices from column names and grouping DataFrame.
    
    Parameters:
    -----------
    column_names : List[str]
        List of column names in the order they appear in your matrix
    group_df : pd.DataFrame
        DataFrame with column names and their corresponding groups
    col_name_column : str
        Name of the column in group_df that contains column names
    group_column : str
        Name of the column in group_df that contains group assignments
        
    Returns:
    --------
    List[List[int]]
        List of lists, where each inner list contains column indices for a group
    """
    # Create a mapping from column name to its index in the matrix
    col_name_to_index = {name: idx for idx, name in enumerate(column_names)}
    
    # Create a dictionary to store groups
    group_dict = {}
    
    # Process each row in the group DataFrame
    for _, row in group_df.iterrows():
        col_name = row[col_name_column]
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


def group_wise_imputation(data: np.ndarray, group_indices: List[List[int]]) -> np.ndarray:
    """
    Perform group-wise imputation on a numpy array.
    
    For each row and each group:
    - If group has ≥2 non-missing values: impute missing values with group mean
    - If group has 0 or 1 non-missing value: impute missing values with group minimum (across all rows)
    - If group has 0 non-missing values: keep as NaN
    
    Parameters:
    -----------
    data : np.ndarray
        2D numpy array with missing values (NaN)
    group_indices : List[List[int]]
        List of lists, where each inner list contains column indices for a group
        
    Returns:
    --------
    np.ndarray
        Imputed array
    """
    # Create a copy to avoid modifying original data
    imputed_data = data.copy()
    
    # Calculate group minimums across all rows for fallback imputation
    group_mins = {}
    for group_idx, col_indices in enumerate(group_indices):
        group_data = data[:, col_indices]
        # Get minimum value across all rows and columns in this group
        # Handle case where all values in group are NaN
        if np.all(np.isnan(group_data)):
            group_mins[group_idx] = np.nan  # or any default value you prefer
            #print(f"Warning: All values in group {group_idx} are NaN. Using 0.0 as fallback.")
        else:
            group_mins[group_idx] = np.nanmin(group_data)
    
    # Process each row
    for row_idx in range(data.shape[0]):
        # Process each group
        for group_idx, col_indices in enumerate(group_indices):
            # Get values for this group in current row
            group_values = data[row_idx, col_indices]
            
            # Find non-missing values
            non_missing_mask = ~np.isnan(group_values)
            non_missing_values = group_values[non_missing_mask]
            non_missing_count = len(non_missing_values)
            
            # Find missing value positions
            missing_mask = np.isnan(group_values)
            
            if np.any(missing_mask):  # Only impute if there are missing values
                if non_missing_count >= 2:
                    # Use mean of non-missing values in this row's group
                    #fill_value = np.mean(non_missing_values)
                    fill_value = group_mins[group_idx]
                else:
                    # Use minimum value of this group across all rows
                    fill_value = group_mins[group_idx]
                
                # Fill missing values
                for i, col_idx in enumerate(col_indices):
                    if missing_mask[i]:
                        imputed_data[row_idx, col_idx] = fill_value
    
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