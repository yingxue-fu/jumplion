# external
import numpy as np
import pandas as pd
from numba import njit
# local
from jumplion.utils import E

@njit
def interpolate_ms1(ms1_rts: np.ndarray,
                    ms1: np.ndarray,
                    ms2_rts: np.ndarray,
                    pad_value: float = 0.0) -> np.ndarray:
    n, m = ms1.shape
    ms1_new = np.full((n, m), pad_value, dtype=ms1.dtype)

    for j in range(m):
        for i in range(n - 1):
            t0 = ms1_rts[i]
            t1 = ms1_rts[i + 1]
            v0 = ms1[i, j]
            v1 = ms1[i + 1, j]
            t = ms2_rts[i]
            delta_t = t1 - t0
            if delta_t != 0:
                slope = (v1 - v0) / delta_t
                ms1_new[i, j] = v0 + slope * (t - t0)
            else:
                ms1_new[i, j] = v0  # or pad_value or np.nan, depending on preference

    # Last row remains as pad_value
    return ms1_new


@njit
def rankdata(x):
    n = len(x)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = 1
        for j in range(n):
            if x[j] < x[i]:
                r += 1
        ranks[i] = r
    return ranks

@njit
def corr(x, y, method):
    """
    Compute Pearson or Spearman correlation between two arrays.
    Set method = 0 for Pearson, 1 for Spearman.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if method == 1:
        x = rankdata(x).astype(np.float64)
        y = rankdata(y).astype(np.float64)

    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x * x)
    sy2 = np.sum(y * y)
    sxy = np.sum(x * y)

    numerator = n * sxy - sx * sy
    denominator_x = n * sx2 - sx * sx
    denominator_y = n * sy2 - sy * sy
    denominator = np.sqrt(max(E, denominator_x * denominator_y))

    return numerator / denominator

@njit
def get_corr_matrix(x, method):
    """
    Compute correlation matrix (Pearson or Spearman) for matrix x (n, p).
    """
    n, p = x.shape
    corr_matrix = np.zeros((p, p))

    for i in range(p):
        for j in range(i + 1, p):
            r = corr(x[:, i], x[:, j], method)
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix

@njit
def smooth(src):
    """
    Smooths a 1D array using a weighted average.

    For boundary points, it applies specific weights:
    - First point: 2/3 of itself + 1/3 of the next point.
    - Last point: 2/3 of itself + 1/3 of the previous point.

    For all interior points, it applies a smoothing kernel:
    - Current value gets 50% weight, neighbors get 25% each.

    Parameters:
    src (np.ndarray): Input 1D array of numerical values.

    Returns:
    np.ndarray: Smoothed 1D array of the same shape.
    """
    n = len(src)  
    dst = np.zeros_like(src)  
    
    # Apply smoothing to the first element
    dst[0] = (2.0 / 3.0) * src[0] + (1.0 / 3.0) * src[1]
    
    # Apply smoothing to the last element
    dst[n - 1] = (2.0 / 3.0) * src[n - 1] + (1.0 / 3.0) * src[n - 2]
    
    # Apply smoothing to the interior elements
    for i in range(1, n - 1):
        dst[i] = 0.5 * src[i] + 0.25 * (src[i - 1] + src[i + 1])

    return dst

@njit
def get_fwhm(xic):
    half_max = np.max(xic) / 2.0
    count = 0
    first = -1
    last = -1
    for i in range(xic.shape[0]):
        if xic[i] >= half_max:
            if first == -1:
                first = i
            last = i
            count += 1
    if first == -1 or last == -1 or count < 2:
        return 0
    return last - first + 1

@njit
def gaussian_curve(size, center, fwhm, max_value):
    """
    Generate a Gaussian curve using FWHM in data points.

    Parameters:
        size (int): Number of points in the curve.
        center (int or float): Index position of the peak (0-based). If -1, centered in the middle.
        fwhm (float): Full Width at Half Maximum in number of data points.
        max_value (float): Peak height.

    Returns:
        np.ndarray: Gaussian curve of given size.
    """
    if size < 1:
        return np.empty(0)

    if center < 0:
        center = (size - 1) / 2.0

    # Convert FWHM to sigma in index units
    fwhm = max(1, fwhm)
    sigma = fwhm / 2.35482 

    y = np.empty(size)
    for i in range(size):
        x = i
        y[i] = max_value * np.exp(-0.5 * ((x - center) / sigma)**2)

    return y

@njit
def calculate_shape_score(xic, corr_method):
    xic_fwhm = max(1, get_fwhm(xic))
    apex = np.argmax(xic)
    
    # Bell curve
    bell = gaussian_curve(len(xic), apex, xic_fwhm, np.max(xic))
    bell_corr = corr(xic, bell, corr_method)
    
    # Tail Ratio
    area_left = 0.0
    for i in range(apex+1):
        area_left += xic[i]
    area_left = max(E, area_left)
    
    area_right = 0.0
    for i in range(apex, len(xic)):
        area_right += xic[i]
    area_right = max(E, area_right)
    
    tail_balance = min(area_left, area_right) / max(area_left, area_right)
    
    # Combined shape score
    shape_score = bell_corr * tail_balance
    
    return shape_score


def calculate_xic_score(xics_region, n_ms1, n_ms2, corr_method=0, 
                        shape_weighted=1, intensity_weighted=1):
    # Get correlation matrix between ions
    corr_matrix = get_corr_matrix(xics_region, corr_method)
    corr_matrix[corr_matrix < 0] = E
    corr_scores = np.sum(corr_matrix, axis=1)
    max_corr = np.max(corr_scores)
    corr_scores = corr_scores / max_corr if max_corr > 0 else corr_scores
    
    # weight by shape
    if shape_weighted:
        # shape score
        shape_scores = np.zeros(xics_region.shape[1])
        for i in range(xics_region.shape[1]):
            xic = xics_region[:, i]
            xic_smooth = smooth(xic)
            shape_scores[i] = calculate_shape_score(xic_smooth, corr_method)
        max_score = max(shape_scores)
        shape_scores = shape_scores / max_score if max_score > 0 else shape_scores
        shape_scores[shape_scores < 0] = E
    else:
        shape_scores = np.ones(xics_region.shape[1])
        
    # weight by intensity
    if intensity_weighted:
        # Intensity (area under curve) per ion
        intensities = np.sum(xics_region, axis=0)
        # Norm ms1 intensity
        if n_ms1 > 0 and n_ms2 > 0:
            ms1_intensities = intensities[0:n_ms1]
            ms2_intensities = intensities[n_ms1:]
            
            mono_ms1 = ms1_intensities[0]
            max_ms2 = np.max(ms2_intensities)
            if mono_ms1 > 0 and max_ms2 > 0 and mono_ms1 > max_ms2:
                ms1_intensities = ms1_intensities / (mono_ms1 / max_ms2)
                
            intensities = np.concatenate((ms1_intensities, ms2_intensities), axis=0)

        # get relative ratio of intensity
        log2_intensities = np.log2(intensities + 1)
        max_log2_intensity = np.max(log2_intensities)
        if max_log2_intensity > 0:
            intensity_scores = log2_intensities / max_log2_intensity
        else:
            intensity_scores = np.ones(xics_region.shape[1])
    else:
        intensity_scores = np.ones(xics_region.shape[1])
    
    return corr_scores, shape_scores, intensity_scores


def identify_optimal_xic(ms1, ms2, 
                        apex, radius, 
                        corr_method, 
                        shape_weighted,
                        intensity_weighted,
                        corr_weight,
                        shape_weight,
                        intensity_weight,
                        optimal_xic_source, 
                        min_shape_score,
                        min_apex_intensity, 
                        min_apex_evidence):
    
    '''
    1. Select XICs to scan (MS2 or MS1+MS2).
    2. Compute score for each trace within the window.
    3. For each high-score trace:
       a. Refine apex using smoothed profile.
       b. Check for bell-shaped peak and local dominance.
       c. If all checks pass, return this ion.
    4. Fallback: use bell curve.
    5. Correlate selected elution profile to all MS2 and MS1 traces.
    6. Return best ion info.
    '''
    
    found = False
    region = slice(max(0, apex - radius), 
                   min(ms1.shape[0] - 2, apex + radius + 1))
    
    xics = np.concatenate((ms1, ms2), axis = 1)
    n_ms1 = ms1.shape[1]
    n_ms2 = ms2.shape[1]
    
    (corr_scores, 
     shape_scores, 
     intensity_scores) = calculate_xic_score(xics[region, :], 
                                             n_ms1, 
                                             n_ms2, 
                                             corr_method, 
                                             shape_weighted, 
                                             intensity_weighted)
    # Sort descending by combined score
    scores = ((corr_weight*corr_scores) + 
              (intensity_weight*intensity_scores) + 
              (shape_weight*shape_scores))
    indices = np.argsort(scores)[::-1]
    
    ms1_indices = np.arange(n_ms1)
    ms2_indices = np.arange(n_ms1, xics.shape[1])
    if optimal_xic_source == 'ms1':
        indices_f = np.array([idx for idx in indices if idx not in ms2_indices])
    elif optimal_xic_source == 'ms2':
        indices_f = np.array([idx for idx in indices if idx not in ms1_indices]) 
    elif optimal_xic_source == 'ms1_ms2':
        indices_f = indices
        
    for idx in indices_f:
        ion_xic = xics[region, idx]
        elution = smooth(ion_xic)
        elution_apex = np.argmax(elution)
            
        refined_apex = region.start + elution_apex
        new_region = slice(max(0, refined_apex - radius), min(ms2.shape[0], refined_apex + radius + 1))
        elution = smooth(xics[new_region, idx])
        elution_apex = np.argmax(elution)
        elution_apex_global = new_region.start + elution_apex
        fwhm = max(1, get_fwhm(elution))
        
        # shape score
        shape_score = calculate_shape_score(elution, corr_method)
        if shape_score < min_shape_score:
            continue
        
        apex_intensity = elution[elution_apex]
        if apex_intensity < min_apex_intensity:
            continue

        # Check for peak shoulder: no nearby points should be much higher
        window_length = new_region.stop - new_region.start
        flank_start = max(elution_apex - elution_apex // 3, 1)
        flank_end = min(elution_apex + elution_apex // 3 + 1, window_length)
        flank = elution[flank_start:flank_end]
        if np.any(flank > apex_intensity * 1.5):
            continue

        # Check wider evidence: apex should be distinct
        left = elution[:elution_apex-1] if elution_apex > 1 else np.array([])
        right = elution[elution_apex+2:] if elution_apex+2 < len(elution) else np.array([])
        max_neighbor = np.max(np.concatenate([left, right])) if left.size + right.size > 0 else 0
        if apex_intensity < max_neighbor * min_apex_evidence:
            continue

        # All checks passed
        found = True
        break

    # Fallback if no fragment passes
    if not found:
        optimal_xic = -1
        optimal_xic_type = 0
        elution_apex_global = apex
        fwhm = radius
    else:
        if optimal_xic_source == 'ms1':
            optimal_xic_type = 1
            optimal_xic = idx
        elif optimal_xic_source == 'ms2':
            optimal_xic_type = 2
            optimal_xic = idx - n_ms1
        elif optimal_xic_source == 'ms1_ms2':
            if idx >= n_ms1:
                optimal_xic_type = 2
                optimal_xic = idx - n_ms1
            else:
                optimal_xic_type = 1
                optimal_xic = idx
    
    return (corr_scores, 
            shape_scores, 
            intensity_scores, 
            scores, 
            optimal_xic, 
            optimal_xic_type, 
            elution_apex_global, 
            fwhm)


def calculate_quantity(ms1, ms2, 
                       begin, end, apex, 
                       optimal_xic, 
                       optimal_xic_type, 
                       corr_method=0, 
                       intf_removal=True, 
                       intf_factor_ms1=1.2,
                       intf_factor_ms2=1.5):
    
    ms1_ms2 = np.concatenate((ms1, ms2), axis=1)
    n_ms1 = ms1.shape[1]
    len_ = end - begin + 1
    p = ms1_ms2.shape[1]
    
    if optimal_xic_type == 1:
        elution = smooth(ms1[:, optimal_xic])
    elif optimal_xic_type == 2:
        elution = smooth(ms2[:, optimal_xic])
        optimal_xic = optimal_xic + ms1.shape[1]
    elif optimal_xic_type == 0:
        elution_middle = gaussian_curve(len_, -1, len_//3, np.max(ms2[begin:end+1, :]))
        elution = np.concatenate((np.zeros(begin), 
                                  elution_middle, 
                                  np.zeros(ms2.shape[0] - end - 1)), 
                                 axis=0)
    
    # Initialize variables
    quant_corr = np.zeros((p, 2), dtype=np.float64)
    quant_quantity = np.zeros((p, 2), dtype=np.float64)
    ms1_ms2_filtered = ms1_ms2.copy()
    
    # Compute total intensity for each fragment and precursor
    quant_corr[:, 0] = np.array([corr(elution[begin:end+1], ms1_ms2[begin:end+1, ion], corr_method) for ion in range(p)])
    quant_quantity[:, 0] = np.sum(ms1_ms2[begin:end+1, :], axis=0)
    
    # Initialize filtered values
    quant_quantity[:, 1] = quant_quantity[:, 0] 
    quant_corr[:, 1] = quant_corr[:, 0]
    
    # Apply interference removal
    if optimal_xic_type != 0:
        for ion in range(p):
            if ion == optimal_xic:
                continue
            if intf_removal:
                w = max(E, np.sum(elution[apex - 1:apex + 2] ** 2))
                scaled = np.sum(ms1_ms2[apex - 1:apex + 2, :] * elution[apex - 1:apex + 2, None] ** 2, axis=0) / w
                if scaled[optimal_xic]:
                    # Compute ratio r
                    r = scaled[ion] / scaled[optimal_xic]
        
                    # Adjust ratio if correlation is below 0.8
                    if quant_corr[ion, 0] < 0.8:
                        # initial r
                        r = min(r, ms1_ms2[apex, ion] / max(E, ms1_ms2[apex, optimal_xic]))
                        # adjust r
                        radj = r
                        e = elution[apex] * 0.5
                        for pos in range(apex - 1, -1, -1): # for the left side of apex point
                            if elution[pos] < e:
                                break
                            radj = min(radj, ms1_ms2[pos, ion] / max(E, ms1_ms2[pos, optimal_xic]))
                        for pos in range(apex + 1, ms1_ms2.shape[0]): # for the right side of apex point
                            if elution[pos] < e:
                                break
                            radj = min(radj, ms1_ms2[pos, ion] / max(E, ms1_ms2[pos, optimal_xic]))
                        # final r
                        r = max(r * 0.5, min(r, radj))
        
                    # Apply interference filtering
                    if ion < n_ms1: # for MS1 XICs
                        ms1_ms2_filtered[begin:end+1, ion] = np.minimum(ms1_ms2[begin:end+1, ion], intf_factor_ms1 * r * elution[begin:end+1])
                    else:           # for MS2 XICs
                        ms1_ms2_filtered[begin:end+1, ion] = np.minimum(ms1_ms2[begin:end+1, ion], intf_factor_ms2 * r * elution[begin:end+1])
                    quant_quantity[ion, 1] = np.sum(ms1_ms2_filtered[begin:end+1, ion])
        
        # calculate correlation after correction
        quant_corr[:, 1] = np.array([corr(elution[begin:end+1], ms1_ms2_filtered[begin:end+1, ion], corr_method) for ion in range(p)])
    
    # corrected XIC region
    ms1_ms2_filtered = ms1_ms2_filtered[begin:end+1, :]
    
    return ms1_ms2_filtered, elution, quant_corr, quant_quantity


def extract_ms1_ms2_from_xic_file(xic_file, precursor_id):

    xic_data = pd.read_parquet(xic_file, filters=[('Precursor.Id', '==', precursor_id)])
    if xic_data.empty:
        raise ValueError("No XIC data found for selected precursor")
        
    xic_mask = xic_data['row_type'] == 'xic_row'
    mz_mask = xic_data['row_type'] == 'mz_row'
    ms1_intensity_cols = xic_data.filter(like='ms1_intensity').columns.to_list()
    ms2_intensity_cols = xic_data.filter(like='ms2_intensity').columns.to_list()
    ms1_interpol_cols = xic_data.filter(like='ms1_interpol').columns.to_list()
    
    ms1_df = xic_data.loc[xic_mask, ['ms1_rt'] + ms1_intensity_cols]
    ms2_df = xic_data.loc[xic_mask, ['ms2_rt'] + ms2_intensity_cols]
    ms1_i_df = xic_data.loc[xic_mask, ['ms2_rt'] + ms1_interpol_cols]
    
    ms1_mzs = np.round(xic_data.loc[mz_mask, ms1_intensity_cols].values[0], 4).tolist()
    ms1_df.columns = ['ms1_rt'] + ms1_mzs
    ms1_i_df.columns = ['ms2_rt'] + ms1_mzs
    
    ms2_mzs = np.round(xic_data.loc[mz_mask, ms2_intensity_cols].values[0], 4).tolist()
    ms2_df.columns = ['ms2_rt'] + ms2_mzs
    
    ms1_df.reset_index(drop=True, inplace=True)
    ms1_i_df.reset_index(drop=True, inplace=True)
    ms2_df.reset_index(drop=True, inplace=True)
    
    return ms1_df, ms2_df, ms1_i_df


def update_xic_file(
    xic_file,
    precursor_ids,
    ms1_ms2_filtered_list,
    elution_list,
    start_list,
    stop_list
):
    # Load the full dataset once
    full_data = pd.read_parquet(xic_file)

    for precursor_id, ms1_ms2_filtered, elution, begin, end in zip(
        precursor_ids, ms1_ms2_filtered_list, elution_list, start_list, stop_list
    ):
        # Subset the data for this precursor
        xic_data = full_data[full_data['Precursor.Id'] == precursor_id]
        if xic_data.empty:
            raise ValueError(f"No XIC data found for precursor {precursor_id}")

        original_mask = xic_data['xic_type'] == 'original'
        corrected_mask = xic_data['xic_type'] == 'corrected'
        ms1_intensity_columns = xic_data.filter(like='ms1_intensity').columns.tolist()
        ms2_intensity_columns = xic_data.filter(like='ms2_intensity').columns.tolist()

        # Modify the subset
        xic_data.loc[corrected_mask, ms1_intensity_columns + ms2_intensity_columns] = ms1_ms2_filtered
        xic_data.loc[corrected_mask, ['ms1_rt', 'ms2_rt']] = xic_data.loc[original_mask, ['ms1_rt', 'ms2_rt']].values[begin:end+1, :]
        xic_data.loc[original_mask, 'elution'] = elution
        xic_data.loc[corrected_mask, 'elution'] = elution[begin:end+1]

        # Replace in the full dataset
        full_data.loc[xic_data.index] = xic_data

    # Save the updated full dataset back to file
    full_data.to_parquet(xic_file, index=False)
    
