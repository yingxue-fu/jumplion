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
def get_fwhm_bounds(xic):
    """Return inclusive first/last indices at >= half maximum and width."""
    half_max = np.max(xic) / 2.0
    first = -1
    last = -1
    count = 0
    for i in range(xic.shape[0]):
        if xic[i] >= half_max:
            if first == -1:
                first = i
            last = i
            count += 1
    if first == -1 or last == -1 or count < 2:
        return -1, -1, 0
    return first, last, last - first + 1


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


def _mean_finite(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    return float(np.mean(values)) if values.size else np.nan


def _weighted_available(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    w = weights[valid]
    w = w / w.sum()
    v = np.nan_to_num(values[valid], nan=0.0, posinf=1.0, neginf=0.0)
    v = np.clip(v, 0.0, 1.0)
    return float(np.sum(v * w))


def _calculate_shape_scores(xics_region, corr_method):
    n_ions = xics_region.shape[1]

    shape_scores = np.full(n_ions, np.nan, dtype=float)

    for i in range(n_ions):
        try:
            value = float(
                calculate_shape_score(
                    smooth(xics_region[:, i]),
                    corr_method,
                )
            )
        except Exception:
            value = np.nan

        if np.isfinite(value):
            shape_scores[i] = max(0.0, value)

    return shape_scores


def _calculate_intensity_scores(xics_region, n_ms1, n_ms2):
    n_ions = xics_region.shape[1]

    intensities = np.nansum(xics_region, axis=0)

    if n_ms1 > 0 and n_ms2 > 0:
        ms1_intensities = intensities[:n_ms1].copy()
        ms2_intensities = intensities[n_ms1:].copy()

        mono_ms1 = (
            ms1_intensities[0]
            if ms1_intensities.size
            else 0.0
        )
        max_ms2 = (
            np.nanmax(ms2_intensities)
            if ms2_intensities.size
            else 0.0
        )

        if mono_ms1 > 0 and max_ms2 > 0 and mono_ms1 > max_ms2:
            ms1_intensities = ms1_intensities / (mono_ms1 / max_ms2)

        intensities = np.concatenate([
            ms1_intensities,
            ms2_intensities,
        ])

    intensities = np.where(
        np.isfinite(intensities),
        intensities,
        0.0,
    )

    log2_intensities = np.log2(
        np.maximum(intensities, 0.0) + 1.0
    )

    max_log2 = np.max(log2_intensities)

    if max_log2 > 0:
        return log2_intensities / max_log2

    return np.ones(n_ions, dtype=float)


def _joint_selection_scores(
    xics_region,
    n_ms1,
    n_ms2,
    corr_method=0,
    corr_weight=0.50,
    shape_weight=0.30,
    intensity_weight=0.20,
    within_modality_weight=0.50,
    cross_modality_weight=0.50,
    cross_top_k=3,
):
    """
    Stage 1 scoring: select the XIC that best represents the joint
    chromatographic event.

    Cross-MS1/MS2 correlation is intentionally INCLUDED here.
    """
    xics_region = np.asarray(xics_region, dtype=float)
    if xics_region.ndim != 2:
        raise ValueError("xics_region must be a 2D array.")

    n_ions = xics_region.shape[1]
    if n_ms1 + n_ms2 != n_ions:
        raise ValueError(
            "n_ms1 + n_ms2 must equal the number of XIC columns."
        )
    if n_ions == 0:
        raise ValueError("At least one XIC is required.")
    if cross_top_k < 1:
        raise ValueError("cross_top_k must be >= 1.")

    corr_matrix = np.asarray(
        get_corr_matrix(xics_region, corr_method),
        dtype=float,
    )
    corr_matrix[~np.isfinite(corr_matrix)] = np.nan
    corr_matrix[corr_matrix <= 0] = np.nan

    modality = np.concatenate([
        np.zeros(n_ms1, dtype=int),
        np.ones(n_ms2, dtype=int),
    ])

    within_corr = np.full(n_ions, np.nan, dtype=float)
    cross_corr = np.full(n_ions, np.nan, dtype=float)
    joint_corr = np.full(n_ions, np.nan, dtype=float)

    for i in range(n_ions):
        row = corr_matrix[i].copy()
        row[i] = np.nan

        same_mask = modality == modality[i]
        same_mask[i] = False
        cross_mask = ~same_mask
        cross_mask[i] = False

        within_values = row[same_mask]
        cross_values = row[cross_mask]

        within_values = within_values[np.isfinite(within_values)]
        cross_values = cross_values[np.isfinite(cross_values)]

        if within_values.size:
            within_corr[i] = float(np.mean(within_values))

        if cross_values.size:
            k = min(cross_top_k, cross_values.size)
            cross_corr[i] = float(
                np.median(np.sort(cross_values)[-k:])
            )

        joint_corr[i] = _weighted_available(
            [within_corr[i], cross_corr[i]],
            [within_modality_weight, cross_modality_weight],
        )

    shape_scores = _calculate_shape_scores(
        xics_region,
        corr_method,
    )
    intensity_scores = _calculate_intensity_scores(
        xics_region,
        n_ms1,
        n_ms2,
    )

    combined_scores = np.full(n_ions, np.nan, dtype=float)
    base_weights = np.asarray([
        corr_weight,
        shape_weight,
        intensity_weight,
    ], dtype=float)

    if np.any(~np.isfinite(base_weights)) or np.any(base_weights < 0):
        raise ValueError("Candidate selection weights must be finite and non-negative.")
    if base_weights.sum() <= 0:
        raise ValueError("At least one candidate selection weight must be positive.")

    for i in range(n_ions):
        combined_scores[i] = _weighted_available(
            [joint_corr[i], shape_scores[i], intensity_scores[i]],
            base_weights,
        )

    return {
        "selection_corr_scores": joint_corr,
        "selection_shape_scores": shape_scores,
        "selection_intensity_scores": intensity_scores,
        "selection_combined_scores": combined_scores,
    }


def _modality_specific_evaluation(
    xics_region,
    n_ms1,
    n_ms2,
    corr_method=0,
    corr_weight=0.50,
    shape_weight=0.30,
    intensity_weight=0.20,
    cross_top_k=3,
):
    """
    Stage 2 scoring on the FINAL apex-centered region.

    MS1 and MS2 candidate scores use only within-modality correlation,
    shape, and intensity. Cross-modality correlation is independent.
    """
    xics_region = np.asarray(xics_region, dtype=float)
    n_ions = xics_region.shape[1]

    if n_ms1 + n_ms2 != n_ions:
        raise ValueError(
            "n_ms1 + n_ms2 must equal the number of XIC columns."
        )
    if cross_top_k < 1:
        raise ValueError("cross_top_k must be >= 1.")

    corr_matrix = np.asarray(
        get_corr_matrix(xics_region, corr_method),
        dtype=float,
    )
    corr_matrix[~np.isfinite(corr_matrix)] = np.nan
    corr_matrix[corr_matrix <= 0] = np.nan

    modality = np.concatenate([
        np.zeros(n_ms1, dtype=int),
        np.ones(n_ms2, dtype=int),
    ])

    within_corr = np.full(n_ions, np.nan, dtype=float)
    #cross_corr = np.full(n_ions, np.nan, dtype=float)

    for i in range(n_ions):
        row = corr_matrix[i].copy()
        row[i] = np.nan

        same_mask = modality == modality[i]
        same_mask[i] = False
        #cross_mask = ~same_mask
        #cross_mask[i] = False

        within_values = row[same_mask]
        #cross_values = row[cross_mask]

        within_values = within_values[np.isfinite(within_values)]
        #cross_values = cross_values[np.isfinite(cross_values)]

        if within_values.size:
            within_corr[i] = float(np.mean(within_values))

        # if cross_values.size:
        #     k = min(cross_top_k, cross_values.size)
        #     cross_corr[i] = float(
        #         np.median(np.sort(cross_values)[-k:])
        #     )

    shape_scores = _calculate_shape_scores(
        xics_region,
        corr_method,
    )
    intensity_scores = _calculate_intensity_scores(
        xics_region,
        n_ms1,
        n_ms2,
    )

    base_weights = np.asarray([
        corr_weight,
        shape_weight,
        intensity_weight,
    ], dtype=float)

    if np.any(~np.isfinite(base_weights)) or np.any(base_weights < 0):
        raise ValueError("Modality score weights must be finite and non-negative.")
    if base_weights.sum() <= 0:
        raise ValueError("At least one modality score weight must be positive.")

    combined_scores = np.full(n_ions, np.nan, dtype=float)
    for i in range(n_ions):
        combined_scores[i] = _weighted_available(
            [within_corr[i], shape_scores[i], intensity_scores[i]],
            base_weights,
        )

    ms1_indices = np.arange(n_ms1, dtype=int)
    ms2_indices = np.arange(n_ms1, n_ions, dtype=int)

    ms1_values = combined_scores[ms1_indices]
    ms2_values = combined_scores[ms2_indices]

    if np.isfinite(ms1_values).any():
        best_ms1_index = int(
            ms1_indices[
                np.nanargmax(ms1_values)
            ]
        )
        best_ms1_score = float(combined_scores[best_ms1_index])
        mono_ms1_score = float(combined_scores[0])
    else:
        best_ms1_index = -1
        best_ms1_score = np.nan
        mono_ms1_score = np.nan

    if np.isfinite(ms2_values).any():
        best_ms2_index = int(
            ms2_indices[
                np.nanargmax(ms2_values)
            ]
        )
        best_ms2_score = float(combined_scores[best_ms2_index])
        best_ms2_index = best_ms2_index - n_ms1 # convert to an index within ms2
    else:
        best_ms2_index = -1
        best_ms2_score = np.nan

    if n_ms1 > 0 and n_ms2 > 0:
        cross_block = corr_matrix[:n_ms1, n_ms1:]
        cross_values = cross_block[np.isfinite(cross_block)]

        if cross_values.size:
            k = min(cross_top_k, cross_values.size)
            cross_ms1_ms2_corr = float(
                np.median(np.sort(cross_values)[-k:])
            )
        else:
            cross_ms1_ms2_corr = np.nan
    else:
        cross_ms1_ms2_corr = np.nan

    if n_ms1 > 1:
        block = corr_matrix[:n_ms1, :n_ms1]
        tri = np.triu_indices(n_ms1, k=1)
        ms1_internal_corr = _mean_finite(block[tri])
    else:
        ms1_internal_corr = np.nan

    if n_ms2 > 1:
        block = corr_matrix[n_ms1:, n_ms1:]
        tri = np.triu_indices(n_ms2, k=1)
        ms2_internal_corr = _mean_finite(block[tri])
    else:
        ms2_internal_corr = np.nan

    return {
        "corr_scores": within_corr,
        "shape_scores": shape_scores,
        "intensity_scores": intensity_scores,
        "combined_scores": combined_scores,
        "mono_ms1_xic_score": mono_ms1_score,
        "best_ms1_xic_score": best_ms1_score,
        "best_ms2_xic_score": best_ms2_score,
        "best_ms1_xic_index": best_ms1_index,
        "best_ms2_xic_index": best_ms2_index,
        "cross_ms1_ms2_corr": cross_ms1_ms2_corr,
        "ms1_internal_corr": ms1_internal_corr,
        "ms2_internal_corr": ms2_internal_corr,
    }


def _empty_result(n_ms1, n_ms2, center, radius, status="not_found"):
    """Single source of truth for the failure-path dict."""
    n = n_ms1 + n_ms2
    return {
        "corr_scores": np.full(n, np.nan),
        "shape_scores": np.full(n, np.nan),
        "intensity_scores": np.full(n, np.nan),
        "combined_scores": np.full(n, np.nan),
        "mono_ms1_xic_score": np.nan,
        "best_ms1_xic_score": np.nan,
        "best_ms2_xic_score": np.nan,
        "best_ms1_xic_index": -1,
        "best_ms2_xic_index": -1,
        "cross_ms1_ms2_corr": np.nan,
        "ms1_internal_corr": np.nan,
        "ms2_internal_corr": np.nan,
        "ref_xic": -1,
        "ref_xic_type": 0,
        "ref_xic_score": np.nan,
        "ref_xic_status": status,
        "apex": int(center),
        "fwhm": int(radius),
        "apex_distance": -1,
        "apex_source": "none",
        "apex_score": np.nan,
        "apex_margin": np.nan,
        "n_candidates": 0,
        # Optional integer MS1 fallback-localization primitives.
        # None preserves nullable integer types in downstream Polars tables.
        "ms1_candidate_apex": None,
        "ms1_fwhm_left": None,
        "ms1_fwhm_right": None,
        # Search-derived counts are always integers; zero is a real count.
        "ms1_n_usable_xics_region": 0,
        "ms1_n_usable_xics_local": 0,
        "ms2_n_usable_xics_region": 0,
        "ms2_n_usable_xics_local": 0,
        "evidence_tier": "prediction_only"
    }
 
 
def _evaluate_at_apex(
    xics,
    n_ms1,
    n_ms2,
    apex,
    radius,
    corr_method,
    ref_xic_source,
    corr_weight,
    shape_weight,
    intensity_weight,
    within_modality_weight,
    cross_modality_weight,
    cross_top_k,
):
    """
    Full Stage 1 + Stage 2 evaluation for one candidate apex.
 
    Stage 1 runs on `apex +/- radius` rather than on the original centre,
    so the reference XIC is chosen from a window that actually contains
    the candidate peak.
 
    Returns None if no reference XIC could be selected here.
    """
    n_rt = xics.shape[0]
 
    region = slice(max(0, apex - radius), min(n_rt, apex + radius + 1))
 
    selection = _joint_selection_scores(
        xics[region, :],
        n_ms1=n_ms1,
        n_ms2=n_ms2,
        corr_method=corr_method,
        corr_weight=corr_weight,
        shape_weight=shape_weight,
        intensity_weight=intensity_weight,
        within_modality_weight=within_modality_weight,
        cross_modality_weight=cross_modality_weight,
        cross_top_k=cross_top_k,
    )
 
    selection_scores = selection["selection_combined_scores"]
    valid = np.where(np.isfinite(selection_scores))[0]
    if valid.size == 0:
        return None
 
    ranked = valid[np.argsort(selection_scores[valid])[::-1]]
 
    if ref_xic_source == "ms1":
        candidates = ranked[ranked < n_ms1]
    elif ref_xic_source == "ms2":
        candidates = ranked[ranked >= n_ms1]
    elif ref_xic_source == "ms1_ms2":
        candidates = ranked
    else:
        raise ValueError("ref_xic_source must be 'ms1', 'ms2', or 'ms1_ms2'.")
 
    if candidates.size == 0:
        return None
 
    ref_abs_index = int(candidates[0])
    ref_xic_score = float(selection_scores[ref_abs_index])
 
    if ref_abs_index < n_ms1:
        ref_xic_type, ref_xic = 1, ref_abs_index
    else:
        ref_xic_type, ref_xic = 2, ref_abs_index - n_ms1
 
    # width from the selected reference trace
    elution = smooth(np.ascontiguousarray(xics[region, ref_abs_index], dtype=np.float64))
    fwhm = max(1, get_fwhm(elution))
 
    final_region = slice(max(0, apex - fwhm), min(n_rt, apex + fwhm + 1))
 
    modality = _modality_specific_evaluation(
        xics[final_region, :],
        n_ms1=n_ms1,
        n_ms2=n_ms2,
        corr_method=corr_method,
        corr_weight=corr_weight,
        shape_weight=shape_weight,
        intensity_weight=intensity_weight,
        cross_top_k=cross_top_k,
    )
 
    return {
        "corr_scores": selection["selection_corr_scores"],
        "shape_scores": selection["selection_shape_scores"],
        "intensity_scores": selection["selection_intensity_scores"],
        "combined_scores": selection["selection_combined_scores"],
        "mono_ms1_xic_score": modality["mono_ms1_xic_score"],
        "best_ms1_xic_score": modality["best_ms1_xic_score"],
        "best_ms2_xic_score": modality["best_ms2_xic_score"],
        "best_ms1_xic_index": modality["best_ms1_xic_index"],
        "best_ms2_xic_index": modality["best_ms2_xic_index"],
        "cross_ms1_ms2_corr": modality["cross_ms1_ms2_corr"],
        "ms1_internal_corr": modality["ms1_internal_corr"],
        "ms2_internal_corr": modality["ms2_internal_corr"],
        "ref_xic": ref_xic,
        "ref_xic_type": ref_xic_type,
        "ref_xic_score": ref_xic_score,
        "ref_xic_status": "selected",
        "apex": int(apex),
        "fwhm": int(fwhm),
        "ms1_candidate_apex": None,
        "ms1_fwhm_left": None,
        "ms1_fwhm_right": None,
    }
 

def _proximity_prior(apex, center, sigma, floor):
    """Gaussian decay with a floor, so a distant but much stronger
    peak can still win."""
    if sigma <= 0:
        return 1.0
    d = abs(apex - center)
    return floor + (1.0 - floor) * np.exp(-0.5 * (d / sigma) ** 2)


def evaluate_xics(
    reported,
    ms1,
    ms2,
    center,
    radius,
    corr_method,
    # Final-region/reference-XIC evaluation
    corr_weight=0.50,
    shape_weight=0.30,
    intensity_weight=0.20,
    ref_xic_source="ms1_ms2",
    within_modality_weight=0.50,
    cross_modality_weight=0.50,
    cross_top_k=3,
    min_ref_xic_score=0.0,
    # Candidate extraction / ranking
    max_candidates=3,
    min_apex_score=0.0,
    min_separation=None,
    rt_prediction_error=3,
    prior_apex_floor=0.6,
    # Recovery-route controls
    scan_ms1_xics=False,
    allow_ms1_only=True,
    ms1_only_min_score=0.0,
    # Shared raw-signal threshold
    min_peak_height=0.0,
    # MS2 localization controls (find_apex)
    ms2_half_window=None,
    ms2_top_k=4,
    ms2_min_fragments=1,
    min_ms2_points=1,
    ms2_min_centrality=0.5,
    ms2_w_coelution=1.0,
    ms2_w_intensity=0.7,
    ms2_w_bell=0.3,
    ms2_w_centrality=1.0,
    ms2_w_ms1_bonus=0.5,
    # MS1 fallback controls (find_apex_ms1)
    ms1_half_window=None,
    ms1_top_k=4,
    ms1_min_channels=2,
    min_ms1_points=2,
    ms1_min_coelution=0.05,
    ms1_min_centrality=0.5,
    ms1_fwhm_tolerance=1,
    ms1_max_apex_distance=3,
    ms1_require_mono=True,
    ms1_mono_index=0,
    ms1_w_coelution=1.0,
    ms1_w_mono=0.8,
    ms1_w_intensity=0.7,
    ms1_w_bell=0.3,
    ms1_w_centrality=1.0,
    ms1_w_envelope=0.2,
):
    """
    Two-stage XIC evaluation.

    All peak-localization parameters used by the MS2 and MS1 search routes are
    exposed here so callers can configure recovery behavior from one place.
    `ms2_*` parameters control MS2 co-elution localization; `ms1_*` parameters
    control the MS1 fallback.  `radius` still controls final-region evaluation.
 
    reported=True
        Trust `center`. One evaluation at that apex.
 
    reported=False
        Search the whole RT range with `find_peaks`, take up to
        `max_candidates` apexes, evaluate each, and keep the best that
        passes the `min_ref_xic_score` quality gate. Candidates are ranked
        by localisation score; see the module docstring for why the
        Stage 1/2 scores cannot be used for that ranking.
 
    Additional returned keys
        apex_score     localisation score at the chosen apex (NaN when reported)
        apex_margin    gap to the runner-up candidate; small values mean the
                       apex assignment is ambiguous and worth flagging
        n_candidates   how many candidates were evaluated
        evidence_tier  reported / ms2_coelution / ms1_peak / prediction_only
        ms1_candidate_apex, ms1_fwhm_left, ms1_fwhm_right
                       primitive MS1 localization diagnostics for the fallback
                       route. A candidate qualifies when the predicted center
                       lies within `ms1_fwhm_tolerance` scans of its mono FWHM
                       and its apex is within `ms1_max_apex_distance` scans of
                       the predicted center. For ms1_peak, the qualifying candidate apex is also used
                       as the final apex; prediction_only remains centered on
                       the predicted coordinate.
        ms1_n_usable_xics_region, ms2_n_usable_xics_region
                       usable-XIC counts across the full extracted RT region
        ms1_n_usable_xics_local, ms2_n_usable_xics_local
                       usable-XIC counts around the route-specific evaluated
                       location, using the same search parameters

    Output typing
        Optional integer localization fields use None when not applicable so
        downstream Polars tables can retain nullable integer columns. Search
        counts use integer zero when no usable XIC is present.
    """
    ms1 = np.asarray(ms1, dtype=float)
    ms2 = np.asarray(ms2, dtype=float)
 
    if ms1.ndim != 2 or ms2.ndim != 2:
        raise ValueError("ms1 and ms2 must be 2D arrays.")
    if ms1.shape[0] != ms2.shape[0]:
        raise ValueError("MS1 and MS2 XICs must have the same number of RT points.")
 
    # By default, preserve the historical behavior in which the peak-search
    # half-window and peak-separation scale follow `radius`.  They can now be
    # controlled independently from this top-level function.
    if ms2_half_window is None:
        ms2_half_window = radius
    if ms1_half_window is None:
        ms1_half_window = radius
    if min_separation is None:
        min_separation = radius

    xics = np.concatenate([ms1, ms2], axis=1)
    n_ms1, n_ms2 = ms1.shape[1], ms2.shape[1]
 
    # Region-wide usable-XIC counts use the same thresholds/persistence rules
    # as the peak-search routes. Local counts are added immediately before each
    # return, using the location actually evaluated by that route.
    ms1_n_usable_xics_region, _, _ = _usable_xic_summary(
        ms1, min_peak_height, min_ms1_points
    )
    ms2_n_usable_xics_region, _, _ = _usable_xic_summary(
        ms2, min_peak_height, min_ms2_points
    )

    def add_usable_xic_diagnostics(result, local_center):
        local_center = int(local_center)
        n_rt = ms1.shape[0]

        ms1_lo = max(0, local_center - int(ms1_half_window))
        ms1_hi = min(n_rt, local_center + int(ms1_half_window) + 1)
        ms2_lo = max(0, local_center - int(ms2_half_window))
        ms2_hi = min(n_rt, local_center + int(ms2_half_window) + 1)

        ms1_local, _, _ = _usable_xic_summary(
            ms1[ms1_lo:ms1_hi, :], min_peak_height, min_ms1_points
        )
        ms2_local, _, _ = _usable_xic_summary(
            ms2[ms2_lo:ms2_hi, :], min_peak_height, min_ms2_points
        )

        result.update(
            ms1_n_usable_xics_region=int(ms1_n_usable_xics_region),
            ms2_n_usable_xics_region=int(ms2_n_usable_xics_region),
            ms1_n_usable_xics_local=int(ms1_local),
            ms2_n_usable_xics_local=int(ms2_local),
        )
        return result
 
    common = dict(
        corr_method=corr_method,
        ref_xic_source=ref_xic_source,
        corr_weight=corr_weight,
        shape_weight=shape_weight,
        intensity_weight=intensity_weight,
        within_modality_weight=within_modality_weight,
        cross_modality_weight=cross_modality_weight,
        cross_top_k=cross_top_k,
    )
 
    # ------------------------------------------------------------------
    # reported: single evaluation at the supplied centre
    # ------------------------------------------------------------------
    if reported:
        result = _evaluate_at_apex(xics, n_ms1, n_ms2, center, radius, **common)
        if result is None:
            empty = _empty_result(n_ms1, n_ms2, center, radius)
            return add_usable_xic_diagnostics(empty, center)
        result.update(apex_distance=0,
                      apex_source="reported",
                      apex_score=np.nan,
                      apex_margin=np.nan,
                      n_candidates=1,
                      evidence_tier="reported")
        return add_usable_xic_diagnostics(result, center)
 
    # ------------------------------------------------------------------
    # unreported: locate, then evaluate each candidate
    # ------------------------------------------------------------------
    found = find_peaks_any(
        ms2,
        ms1=ms1,
        center=center,
        scan_ms1_xics=scan_ms1_xics,
        allow_ms1_only=allow_ms1_only,
        ms1_only_min_score=ms1_only_min_score,
        corr_method=corr_method,
        half_window=ms2_half_window,
        top_k=ms2_top_k,
        min_fragments=ms2_min_fragments,
        min_separation=min_separation,
        max_peaks=max_candidates,
        min_score=min_apex_score,
        min_peak_height=min_peak_height,
        min_ms2_points=min_ms2_points,
        min_centrality=ms2_min_centrality,
        w_coelution=ms2_w_coelution,
        w_intensity=ms2_w_intensity,
        w_bell=ms2_w_bell,
        w_centrality=ms2_w_centrality,
        w_ms1_bonus=ms2_w_ms1_bonus,
        # MS1-only parameters are routed separately inside find_peaks_any.
        ms1_half_window=ms1_half_window,
        ms1_top_k=ms1_top_k,
        min_channels=ms1_min_channels,
        min_ms1_points=min_ms1_points,
        min_coelution=ms1_min_coelution,
        ms1_min_centrality=ms1_min_centrality,
        ms1_fwhm_tolerance=ms1_fwhm_tolerance,
        ms1_max_apex_distance=ms1_max_apex_distance,
        require_mono=ms1_require_mono,
        mono_index=ms1_mono_index,
        w_mono=ms1_w_mono,
        w_envelope=ms1_w_envelope,
        ms1_w_coelution=ms1_w_coelution,
        ms1_w_intensity=ms1_w_intensity,
        ms1_w_bell=ms1_w_bell,
        ms1_w_centrality=ms1_w_centrality,
    )
 
    peaks = found.get("peaks", [])
    tier = found.get("evidence_tier")

    if not scan_ms1_xics and tier in {"ms1_peak", "prediction_only"}:
        center_common = dict(
            common,
            ref_xic_source="ms1",
            within_modality_weight=1.0,
            cross_modality_weight=0.0,
        )

        if tier == "ms1_peak":
            # The qualifying MS1 candidate is constrained by the predicted
            # center: its monoisotopic FWHM must contain `center`. Once that
            # criterion is met, use the measured MS1 candidate apex as the
            # final localization rather than retaining the predicted center.
            candidate_pool = found.get("center_overlapping_ms1_peaks", [])
            cand = candidate_pool[0] if candidate_pool else {}
            eval_apex = int(cand["apex"]) if cand.get("apex") is not None else int(center)
            apex_source = "ms1"
            apex_distance = abs(eval_apex - int(center))
        else:
            # No qualifying peak supports moving away from the prediction.
            # Keep the best global MS1 candidate only as a diagnostic.
            candidate_pool = found.get("all_ms1_peaks", [])
            cand = candidate_pool[0] if candidate_pool else {}
            eval_apex = int(center)
            apex_source = "center"
            apex_distance = 0

        result = _evaluate_at_apex(
            xics, n_ms1, n_ms2, eval_apex, radius, **center_common
        )
        if result is None:
            result = _empty_result(
                n_ms1, n_ms2, eval_apex, radius, status="no_signal"
            )

        result.update(
            apex_source=apex_source,
            apex_distance=int(apex_distance),
            apex_score=(
                float(cand["score"])
                if tier == "ms1_peak" and cand.get("score") is not None
                else np.nan
            ),
            apex_margin=np.nan,
            n_candidates=len(peaks),
            # For ms1_peak these describe the qualifying mono-defined peak
            # used for final localization. For prediction_only they describe
            # the best global/off-center candidate, when one exists.
            ms1_candidate_apex=(
                int(cand["apex"]) if cand.get("apex") is not None else None
            ),
            ms1_fwhm_left=(
                int(cand["ms1_fwhm_left"])
                if cand.get("ms1_fwhm_left") is not None
                and int(cand["ms1_fwhm_left"]) >= 0
                else None
            ),
            ms1_fwhm_right=(
                int(cand["ms1_fwhm_right"])
                if cand.get("ms1_fwhm_right") is not None
                and int(cand["ms1_fwhm_right"]) >= 0
                else None
            ),
            evidence_tier=tier,
        )
        return add_usable_xic_diagnostics(result, eval_apex)

    if not peaks:
        result = _empty_result(
            n_ms1, n_ms2, center, radius, status="no_signal"
        )
        result["evidence_tier"] = tier or "prediction_only"
        return add_usable_xic_diagnostics(result, center)

    evaluated = []
    for peak in peaks:
        result = _evaluate_at_apex(
            xics, n_ms1, n_ms2, peak["apex"], radius, **common
        )
        if result is None:
            continue
        if not np.isfinite(result["ref_xic_score"]):
            continue
        if result["ref_xic_score"] < min_ref_xic_score:
            continue
        result["apex_distance"] = abs(peak["apex"] - center)
        result["apex_prior"] = _proximity_prior(
            peak["apex"], center, rt_prediction_error, prior_apex_floor)
        result["apex_score"] = peak["score"]              # raw, for diagnostics
        result["ranking_score"] = peak["score"] * result["apex_prior"]
        evaluated.append(result)
 
    if not evaluated:
        empty = _empty_result(n_ms1, n_ms2, center, radius, status="failed_quality_gate")
        return add_usable_xic_diagnostics(empty, center)
 
    # rank by localisation score: the only term with an absolute
    # across-position intensity anchor
    evaluated.sort(key=lambda r: r["ranking_score"], reverse=True)
 
    best = evaluated[0]
    best["n_candidates"] = len(evaluated)
    best["apex_source"] = "located"
    best["apex_margin"] = (
        best["ranking_score"] - evaluated[1]["ranking_score"] if len(evaluated) > 1 else np.inf
    )
    best["evidence_tier"] = found.get("evidence_tier")
    return add_usable_xic_diagnostics(best, best["apex"])

# def evaluate_xics(
#     reported,
#     ms1,
#     ms2,
#     center,
#     radius,
#     corr_method,
#     corr_weight=0.50,
#     shape_weight=0.30,
#     intensity_weight=0.20,
#     ref_xic_source="ms1_ms2",
#     apex_tolerance=0,
#     within_modality_weight=0.50,
#     cross_modality_weight=0.50,
#     cross_top_k=3,
# ):
#     """
#     Two-stage XIC evaluation.

#     Stage 1
#     -------
#     Select the reference XIC using the full MS1+MS2 XIC pool.
#     Cross-modality correlation contributes to this selection because the
#     purpose is chromatographic localization.

#     Stage 2
#     -------
#     Use the refined apex to define the final region. Calculate MS1/MS2
#     modality-specific scores on that final region. Cross-MS1/MS2
#     concordance is reported separately and is not included in those scores.

#     Returns only one reference-XIC identity:
#         ref_xic
#         ref_xic_type
#         ref_xic_score
#     plus apex/FWHM and Stage-2 modality metrics.
#     """
#     ms1 = np.asarray(ms1, dtype=float)
#     ms2 = np.asarray(ms2, dtype=float)

#     if ms1.ndim != 2 or ms2.ndim != 2:
#         raise ValueError("ms1 and ms2 must be 2D arrays.")
#     if ms1.shape[0] != ms2.shape[0]:
#         raise ValueError(
#             "MS1 and MS2 XICs must have the same number of RT points."
#         )

#     xics = np.concatenate([ms1, ms2], axis=1)
#     n_ms1 = ms1.shape[1]
#     n_ms2 = ms2.shape[1]

#     # --------------------------------------------------------------
#     # Initial search region for joint reference-XIC selection.
#     # --------------------------------------------------------------
#     region = slice(
#         max(0, center - radius),
#         min(ms1.shape[0] - 2, center + radius + 1),
#     )

#     selection = _joint_selection_scores(
#         xics[region, :],
#         n_ms1=n_ms1,
#         n_ms2=n_ms2,
#         corr_method=corr_method,
#         corr_weight=corr_weight,
#         shape_weight=shape_weight,
#         intensity_weight=intensity_weight,
#         within_modality_weight=within_modality_weight,
#         cross_modality_weight=cross_modality_weight,
#         cross_top_k=cross_top_k,
#     )

#     selection_scores = selection["selection_combined_scores"]
#     valid_indices = np.where(np.isfinite(selection_scores))[0]

#     if valid_indices.size == 0:
#         # No reference XIC could be selected.
#         return {
#             "corr_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "shape_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "intensity_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "combined_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "mono_ms1_xic_score": np.nan,
#             "best_ms1_xic_score": np.nan,
#             "best_ms2_xic_score": np.nan,
#             "best_ms1_xic_index": -1,
#             "best_ms2_xic_index": -1,
#             "cross_ms1_ms2_corr": np.nan,
#             "ms1_internal_corr": np.nan,
#             "ms2_internal_corr": np.nan,
#             "ref_xic": -1,
#             "ref_xic_type": 0,
#             "ref_xic_score": np.nan,
#             "ref_xic_status": "not_found",
#             "apex": center,
#             "fwhm": radius,
#         }

#     ranked = valid_indices[
#         np.argsort(selection_scores[valid_indices])[::-1]
#     ]

#     if ref_xic_source == "ms1":
#         candidate_indices = ranked[ranked < n_ms1]
#     elif ref_xic_source == "ms2":
#         candidate_indices = ranked[ranked >= n_ms1]
#     elif ref_xic_source == "ms1_ms2":
#         candidate_indices = ranked
#     else:
#         raise ValueError(
#             "ref_xic_source must be 'ms1', 'ms2', or 'ms1_ms2'."
#         )

#     if candidate_indices.size == 0:
#         return {
#             "corr_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "shape_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "intensity_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "combined_scores": np.full(n_ms1 + n_ms2, np.nan),
#             "mono_ms1_xic_score": np.nan,
#             "best_ms1_xic_score": np.nan,
#             "best_ms2_xic_score": np.nan,
#             "best_ms1_xic_index": -1,
#             "best_ms2_xic_index": -1,
#             "cross_ms1_ms2_corr": np.nan,
#             "ms1_internal_corr": np.nan,
#             "ms2_internal_corr": np.nan,
#             "ref_xic": -1,
#             "ref_xic_type": 0,
#             "ref_xic_score": np.nan,
#             "ref_xic_status": "not_found",
#             "apex": center,
#             "fwhm": radius,
#         }

#     ref_abs_index = int(candidate_indices[0])
#     ref_xic_score = float(
#         selection_scores[ref_abs_index]
#     )

#     if ref_abs_index < n_ms1:
#         ref_xic_type = 1
#         ref_xic = ref_abs_index
#     else:
#         ref_xic_type = 2
#         ref_xic = ref_abs_index - n_ms1

#     # --------------------------------------------------------------
#     # Determine apex/FWHM from the reference XIC.
#     # There is NO post-selection shape rejection.
#     # --------------------------------------------------------------
#     elution = smooth(xics[region, ref_abs_index])
#     apex = center
#     fwhm = max(1, get_fwhm(elution))
    
#     if not reported:
#         temp_apex = np.argmax(elution) # todo: use local maximum rather than simple max
#         apex_offset = min(
#             np.abs(temp_apex - radius), # radius is the center point of elution
#             apex_tolerance,
#         )
#         elution_apex = (
#             radius + apex_offset
#             if temp_apex >= radius
#             else radius - apex_offset
#         )
#         refined_apex = region.start + elution_apex

#         refined_region = slice(
#             max(0, refined_apex - radius),
#             min(ms2.shape[0], refined_apex + radius + 1),
#         )

#         elution = smooth(
#             xics[refined_region, ref_abs_index]
#         )
#         apex = refined_apex
#         fwhm = max(1, get_fwhm(elution))
        
#     final_region = slice(
#         max(0, apex - fwhm),
#         min(ms2.shape[0], apex + fwhm + 1),
#     )

#     # --------------------------------------------------------------
#     # Stage 2: modality-specific scores on the FINAL region.
#     # --------------------------------------------------------------
#     modality_scores = _modality_specific_evaluation(
#         xics[final_region, :],
#         n_ms1=n_ms1,
#         n_ms2=n_ms2,
#         corr_method=corr_method,
#         corr_weight=corr_weight,
#         shape_weight=shape_weight,
#         intensity_weight=intensity_weight,
#         cross_top_k=cross_top_k,
#     )

#     return {
#         "corr_scores": selection["selection_corr_scores"],
#         "shape_scores": selection["selection_shape_scores"],
#         "intensity_scores": selection["selection_intensity_scores"],
#         "combined_scores": selection["selection_combined_scores"],
#         "mono_ms1_xic_score": modality_scores["mono_ms1_xic_score"],
#         "best_ms1_xic_score": modality_scores["best_ms1_xic_score"],
#         "best_ms2_xic_score": modality_scores["best_ms2_xic_score"],
#         "best_ms1_xic_index": modality_scores["best_ms1_xic_index"],
#         "best_ms2_xic_index": modality_scores["best_ms2_xic_index"],
#         "cross_ms1_ms2_corr": modality_scores["cross_ms1_ms2_corr"],
#         "ms1_internal_corr": modality_scores["ms1_internal_corr"],
#         "ms2_internal_corr": modality_scores["ms2_internal_corr"],       
#         "ref_xic": ref_xic,
#         "ref_xic_type": ref_xic_type,
#         "ref_xic_score": ref_xic_score,
#         "ref_xic_status": "selected",
#         "apex": apex,
#         "fwhm": fwhm,
#     }


def calculate_quantity(
    ms1, ms2,
    begin, end, apex,
    ref_xic,
    ref_xic_type,
    corr_method=0,
):
    
    ms1_ms2 = np.concatenate((ms1, ms2), axis=1)
    len_ = end - begin + 1
    p = ms1_ms2.shape[1]
    
    if ref_xic_type == 1:
        elution = smooth(ms1[:, ref_xic])
    elif ref_xic_type == 2:
        elution = smooth(ms2[:, ref_xic])
        ref_xic = ref_xic + ms1.shape[1]
    elif ref_xic_type == 0:
        elution_middle = gaussian_curve(len_, -1, len_//3, np.max(ms2[begin:end+1, :]))
        elution = np.concatenate((np.zeros(begin), 
                                  elution_middle, 
                                  np.zeros(ms2.shape[0] - end - 1)), 
                                 axis=0)
    
    # Initialize variables
    quant_corr = np.zeros(p, dtype=np.float64)
    quant_quantity = np.zeros(p, dtype=np.float64)
    
    # Compute total intensity for each fragment and precursor
    quant_corr = np.array([corr(elution[begin:end+1], ms1_ms2[begin:end+1, ion], corr_method) for ion in range(p)])
    quant_quantity = np.sum(ms1_ms2[begin:end+1, :], axis=0)
    
    return elution, quant_corr, quant_quantity


def calculate_quantity_with_intf_removal(
    ms1, ms2,
    begin, end, apex,
    best_ms1_xic,
    best_ms2_xic,
    corr_method=0,
    intf_removal=True,
    intf_factor_ms1=1.2,
    intf_factor_ms2=1.5
):
    ms1_ms2 = np.concatenate((ms1, ms2), axis=1)
    n_ms1 = ms1.shape[1]
    p = ms1_ms2.shape[1]

    ms1_elution = smooth(ms1[:, best_ms1_xic])
    ms2_elution = smooth(ms2[:, best_ms2_xic])

    # reference columns expressed in ms1_ms2's combined column space
    ref_ms1 = best_ms1_xic
    ref_ms2 = best_ms2_xic + n_ms1

    quant_corr = np.zeros((p, 2), dtype=np.float64)
    quant_quantity = np.zeros((p, 2), dtype=np.float64)
    ms1_ms2_filtered = ms1_ms2.copy()

    # uncorrected quantity/correlation, each ion scored against its own level's elution
    quant_quantity[:, 0] = np.sum(ms1_ms2[begin:end+1, :], axis=0)
    for ion in range(p):
        elution = ms1_elution if ion < n_ms1 else ms2_elution
        quant_corr[ion, 0] = corr(elution[begin:end+1], ms1_ms2[begin:end+1, ion], corr_method)

    quant_quantity[:, 1] = quant_quantity[:, 0]
    quant_corr[:, 1] = quant_corr[:, 0]

    if intf_removal:
        for ion in range(p):
            is_ms1 = ion < n_ms1
            elution = ms1_elution if is_ms1 else ms2_elution
            ref_xic = ref_ms1 if is_ms1 else ref_ms2
            if ion == ref_xic:
                continue

            w = max(E, np.sum(elution[apex - 1:apex + 2] ** 2))
            scaled = np.sum(ms1_ms2[apex - 1:apex + 2, :] * elution[apex - 1:apex + 2, None] ** 2, axis=0) / w
            if not scaled[ref_xic]:
                continue

            r = scaled[ion] / scaled[ref_xic]

            if quant_corr[ion, 0] < 0.8:
                r = min(r, ms1_ms2[apex, ion] / max(E, ms1_ms2[apex, ref_xic]))
                radj = r
                e = elution[apex] * 0.5
                for pos in range(apex - 1, -1, -1):
                    if elution[pos] < e:
                        break
                    radj = min(radj, ms1_ms2[pos, ion] / max(E, ms1_ms2[pos, ref_xic]))
                for pos in range(apex + 1, ms1_ms2.shape[0]):
                    if elution[pos] < e:
                        break
                    radj = min(radj, ms1_ms2[pos, ion] / max(E, ms1_ms2[pos, ref_xic]))
                r = max(r * 0.5, min(r, radj))

            factor = intf_factor_ms1 if is_ms1 else intf_factor_ms2
            ms1_ms2_filtered[begin:end+1, ion] = np.minimum(
                ms1_ms2[begin:end+1, ion], factor * r * elution[begin:end+1]
            )
            quant_quantity[ion, 1] = np.sum(ms1_ms2_filtered[begin:end+1, ion])

        for ion in range(p):
            elution = ms1_elution if ion < n_ms1 else ms2_elution
            quant_corr[ion, 1] = corr(elution[begin:end+1], ms1_ms2_filtered[begin:end+1, ion], corr_method)

    ms1_ms2_filtered = ms1_ms2_filtered[begin:end+1, :]

    return ms1_ms2_filtered, ms1_elution, ms2_elution, quant_corr, quant_quantity


def extract_ms1_ms2_from_xic_file(xic_file, precursor_id):

    xic_data = pd.read_parquet(xic_file, filters=[('Precursor.Id', '==', precursor_id)])
    if xic_data.empty:
        raise ValueError("No XIC data found for selected precursor")
    
    def xic_to_df(xic_data, idx, rt_col, intensity_col, prefix='iso'):
        ms1 = np.vstack(xic_data.loc[idx, intensity_col])
        cols = [f'{prefix}_{i}' for i in range(ms1.shape[1])]
        df = pd.DataFrame(ms1, columns=cols)
        df.insert(0, 'rt', xic_data.loc[idx, rt_col])
        return df
    
    ms1_df = xic_to_df(xic_data, idx=0, rt_col='ms1_rt', intensity_col='ms1', prefix='iso')
    ms2_df = xic_to_df(xic_data, idx=0, rt_col='ms2_rt', intensity_col='ms2',  prefix='fr')
    ms1_i_df = xic_to_df(xic_data, idx=0, rt_col='ms2_rt', intensity_col='ms1_i', prefix='iso')
    
    ms1_mzs = np.round(xic_data.loc[0, 'ms1_mz'], 4).tolist()
    ms1_df.columns = ['ms1_rt'] + ms1_mzs
    ms1_i_df.columns = ['ms2_rt'] + ms1_mzs
    
    ms2_mzs = np.round(xic_data.loc[0, 'ms2_mz'], 4).tolist()
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
    


def _as_f64(a):
    """Contiguous float64 view, required by the njit primitives."""
    return np.ascontiguousarray(a, dtype=np.float64)
 
 
def _mean_pairwise_corr(block, corr_method):
    """Mean off-diagonal correlation of an (n_rt, k) block.
 
    Negative correlations are clipped to 0 rather than dropped: an
    anti-correlated fragment is evidence against co-elution, and
    discarding it would inflate the mean.
    """
    k = block.shape[1]
    if k < 2:
        return np.nan
 
    cm = np.asarray(get_corr_matrix(_as_f64(block), corr_method), dtype=float)
    tri = np.triu_indices(k, k=1)
    vals = cm[tri]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(np.clip(vals, 0.0, 1.0)))
 
 
def _geometric_mean(components, weights):
    """Weighted geometric mean over components with finite values.
 
    Unlike `_weighted_available`, a non-finite component is dropped and
    its weight redistributed, rather than being counted as zero.
    """
    components = np.asarray(components, dtype=float)
    weights = np.asarray(weights, dtype=float)
 
    keep = (weights > 0) & np.isfinite(weights) & np.isfinite(components)
    if not np.any(keep):
        return np.nan
    if np.any(components[keep] <= 0.0):
        return np.nan
    v = np.clip(components[keep], E, 1.0)
    w = weights[keep]
    return float(np.exp(np.sum(w * np.log(v)) / np.sum(w)))
 



def find_apex(
    ms2,
    ms1=None,
    corr_method=0,
    half_window=4,
    top_k=4,
    min_fragments=2,
    min_ms2_points=1,
    min_peak_height=0.0,
    min_centrality=0.5,
    w_coelution=1.0,
    w_intensity=0.7,
    w_bell=0.3,
    w_centrality=1.0,
    w_ms1_bonus=0.3,
):
    """
    Score every scan as a candidate apex and return the best.
 
    Windows are truncated at the array edges rather than skipped, so a
    peak eluting near the start or end of the range remains findable.
 
    Parameters
    ----------
    ms2 : (n_rt, p) array
        Fragment XICs.
    ms1 : (n_rt,) or (n_rt, k) array, optional
        Precursor XIC(s). If 2D, column 0 (monoisotopic) is used.
        MS1 is used only as supporting evidence: positive MS1-MS2
        correlation increases the MS2-derived score, whereas absent or
        non-positive correlation does not penalize the candidate.
    corr_method : int
        0 = Pearson, 1 = Spearman. Passed through to `corr`.
    half_window : int
        Scans either side of the candidate centre.
    # min_window : int
    #     Minimum usable width; narrower positions score NaN.
    top_k : int
        Fragments (by in-window intensity) entering the co-elution mean.
        Fixed so scores are comparable across precursors.
    min_fragments : int
        Positions with fewer usable fragments score NaN.
    min_peak_height : float
        Fragments never reaching this within the window are excluded.
    w_ms1_bonus : float
        Maximum fractional score bonus from MS1 support. For example,
        0.5 allows an MS1 correlation of 1.0 to increase the MS2 score
        by up to 50%.    
    
    Returns
    -------
    dict with `apex`, `score`, `status`, the combined `scores` profile,
    per-component profiles, and the fragments/window used at the apex.
    """
    ms2 = np.asarray(ms2, dtype=float)
    if ms2.ndim != 2:
        raise ValueError("ms2 must be 2D (n_rt, n_fragments).")
 
    n_rt, p = ms2.shape
    # if n_rt < min_window:
    #     raise ValueError(
    #         f"RT range ({n_rt}) is shorter than min_window ({min_window})."
    #     )
    full_window = 2 * half_window + 1
    if n_rt < full_window:
        raise ValueError(
            f"RT range ({n_rt}) is shorter than the full window ({full_window}); "
            f"reduce half_window."
        )
    
    ms1_available = ms1 is not None
    if ms1_available:
        ms1 = np.asarray(ms1, dtype=float)
        ms1 = ms1[:, 0] if ms1.ndim == 2 else ms1.ravel()
        if ms1.size != n_rt:
            raise ValueError("ms1 and ms2 must share the same RT axis.")
 
    coelution = np.full(n_rt, np.nan)
    ms1_corr = np.full(n_rt, np.nan)
    raw_intensity = np.full(n_rt, np.nan)
    bell = np.full(n_rt, np.nan)
    centrality = np.full(n_rt, np.nan)
    fragments_used = [None] * n_rt
    windows = [None] * n_rt
 
    # ------------------------------------------------------------------
    # pass 1: components computable from a single window
    # ------------------------------------------------------------------
    for c in range(n_rt):
        if c < half_window or c >= n_rt - half_window:
            continue
        start = max(0, c - half_window)
        stop = min(n_rt, c + half_window + 1)
        # if stop - start < min_window:
        #     continue
 
        window = ms2[start:stop, :]
        centre = c - start
 
        totals = window.sum(axis=0)
        #usable = np.where(window.max(axis=0) > min_peak_height)[0]
        usable = _usable_columns(window, min_peak_height, min_ms2_points)
        if usable.size < min_fragments:
            continue
 
        order = usable[np.argsort(totals[usable])[::-1]]
        selected = order[: min(top_k, order.size)]
        if selected.size < min_fragments:
            continue
 
        block = window[:, selected]
 
        mc = _mean_pairwise_corr(block, corr_method)
        if not np.isfinite(mc):
            continue
        coelution[c] = mc
 
        summed = _as_f64(block.sum(axis=1))
 
        if ms1_available:
            c1 = corr(summed, _as_f64(ms1[start:stop]), corr_method)
            ms1_corr[c] = np.clip(c1, 0.0, 1.0) if np.isfinite(c1) else 0.0
 
        raw_intensity[c] = np.log2(1.0 + max(summed.sum(), 0.0))
 
        smoothed = np.asarray(smooth(summed), dtype=float)
 
        # peak-likeness of the summed trace, independent of where the
        # candidate centre sits
        try:
            bs = float(calculate_shape_score(smoothed, corr_method))
        except Exception:
            bs = np.nan
        bell[c] = np.clip(bs, 0.0, 1.0) if np.isfinite(bs) else 0.0
 
        # does the candidate centre actually sit on the maximum
        top = smoothed.max()
        centrality[c] = float(np.clip(smoothed[centre] / top, 0.0, 1.0)) if top > E else 0.0
        
        fragments_used[c] = selected
        windows[c] = slice(start, stop)
 
    # ------------------------------------------------------------------
    # pass 2: normalise intensity across positions, then combine
    # ------------------------------------------------------------------
    finite = np.isfinite(raw_intensity)
    intensity = np.full(n_rt, np.nan)
 
    if np.any(finite):
        peak_intensity = raw_intensity[finite].max()
        intensity[finite] = (
            raw_intensity[finite] / peak_intensity if peak_intensity > E else 0.0
        )
 
    weights = np.array(
        [w_coelution, w_intensity, w_bell, w_centrality], dtype=float
    )
 
    scores = np.full(n_rt, np.nan)
    for c in range(n_rt):
        if not np.isfinite(coelution[c]):
            continue
        # hard gate: a candidate apex must sit near the top of its own window
        if centrality[c] < min_centrality:      # 0.6-0.7
            continue
        ms2_score = _geometric_mean(
            [coelution[c], intensity[c], bell[c], centrality[c]],
            weights,
        )
        if not np.isfinite(ms2_score):
            continue
        if ms1_available and np.isfinite(ms1_corr[c]):
            ms1_bonus = 1.0 + w_ms1_bonus * ms1_corr[c]
        else:
            ms1_bonus = 1.0
        
        scores[c] = ms2_score * ms1_bonus
        
    valid = np.where(np.isfinite(scores))[0]
    if valid.size == 0:
        return {
            "apex": -1,
            "score": np.nan,
            "status": "no_signal",
            "scores": scores,
            "coelution": coelution,
            "ms1_corr": ms1_corr,
            "intensity": intensity,
            "bell": bell,
            "centrality": centrality,
            "best_fragments": np.array([], dtype=int),
            "window": None,
        }
 
    apex = int(valid[np.argmax(scores[valid])])
    return {
        "apex": apex,
        "score": float(scores[apex]),
        "status": "found",
        "scores": scores,
        "coelution": coelution,
        "ms1_corr": ms1_corr,
        "intensity": intensity,
        "bell": bell,
        "centrality": centrality,
        "best_fragments": fragments_used[apex],
        "window": windows[apex],
    }
 
 
def find_peaks(
    ms2, ms1=None, max_peaks=3, min_separation=None, min_score=0.0, **kwargs
):
    """
    Greedy multi-apex extraction from a single score profile.
 
    Takes the highest-scoring scan, suppresses everything within
    `min_separation`, repeats. Defaults to `half_window` separation so
    two scans on the same chromatographic peak are not both returned.
    """
    result = find_apex(ms2, ms1=ms1, **kwargs)
    scores = result["scores"].copy()
 
    if min_separation is None:
        min_separation = kwargs.get("half_window", 4)
 
    peaks = []
    while len(peaks) < max_peaks:
        valid = np.where(np.isfinite(scores))[0]
        if valid.size == 0:
            break
        idx = int(valid[np.argmax(scores[valid])])
        if scores[idx] < min_score:
            break
        peaks.append({"apex": idx, "score": float(scores[idx])})
        lo = max(0, idx - min_separation)
        hi = min(scores.size, idx + min_separation + 1)
        scores[lo:hi] = np.nan
 
    result["peaks"] = peaks
    return result


# def _has_ms2_evidence(ms2, min_peak_height=0.0, min_fragments=2, min_scans=1):
#     """True when at least `min_scans` scans carry >= `min_fragments`
#     co-occurring fragments above threshold.
 
#     Strict `>` so that all-zero channels never count, which is what let
#     empty fragments satisfy `min_fragments` in the original code.
#     """
#     ms2 = np.asarray(ms2, dtype=float)
#     if ms2.ndim != 2 or ms2.size == 0:
#         return False
#     per_scan = np.sum(ms2 > min_peak_height, axis=1)
#     return int(np.sum(per_scan >= min_fragments)) >= min_scans


def _max_run(mask):
    """Longest run of consecutive True values."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def _usable_columns(window, min_peak_height=0.0, min_points=2):
    """Columns with at least `min_points` consecutive scans above threshold.

    Strict `>` so all-zero columns never qualify, and a run requirement so
    isolated spikes -- which correlate perfectly with each other -- cannot
    contribute co-elution evidence.
    """
    out = []
    for i in range(window.shape[1]):
        if _max_run(window[:, i] > min_peak_height) >= min_points:
            out.append(i)
    return np.asarray(out, dtype=int)

def _has_ms2_evidence(ms2, min_peak_height=0.0, min_fragments=1,
                      min_ms2_points=1, min_scans=1):
    ms2 = np.asarray(ms2, dtype=float)
    if ms2.ndim != 2 or ms2.size == 0:
        return False
    keep = _usable_columns(ms2, min_peak_height, min_ms2_points)
    if keep.size < min_fragments:
        return False
    per_scan = np.sum(ms2[:, keep] > min_peak_height, axis=1)
    return int(np.sum(per_scan >= min_fragments)) >= min_scans


def _usable_xic_summary(xics, min_peak_height=0.0, min_points=2):
    """Count usable XIC channels using the same persistence rule as peak search."""
    xics = np.asarray(xics, dtype=float)
    if xics.ndim != 2:
        raise ValueError("xics must be 2D.")
    n_total = int(xics.shape[1])
    if n_total == 0:
        return 0, np.nan, np.array([], dtype=int)
    usable = _usable_columns(xics, min_peak_height, min_points)
    return int(usable.size), float(usable.size / n_total), usable


def find_apex_ms1(
    ms1,
    corr_method=0,
    half_window=4,
    top_k=4,
    min_channels=2,
    min_ms1_points=2,
    min_peak_height=0.0,
    min_coelution=0.05,
    min_centrality=0.5,
    #min_mono_corr=0.5,          # [CHANGE 1]
    require_mono=True,
    mono_index=0,
    w_coelution=1.0,
    w_mono=0.8,                 # [CHANGE 1]
    w_intensity=0.7,
    w_bell=0.3,
    w_centrality=1.0,
    w_envelope=0.2,
):
    """
    Score every scan as a candidate apex using MS1 isotope channels alone.
 
    The monoisotopic channel anchors the call: it must be present, it must
    agree with the rest of the envelope, and it -- not the summed trace --
    defines where the apex sits and how wide the peak is. A heavy channel
    carrying a co-isolated interference can otherwise dominate the sum and
    pull the region onto the wrong feature.
 
    Parameters
    ----------
    ms1 : (n_rt, k) array
        Isotope channel XICs. Column `mono_index` is the monoisotopic.
    half_window : int
        Scans either side of the candidate centre. Positions closer than
        this to either end of the range are skipped: an apex whose rising
        or falling edge was never acquired is not determinable.
    top_k : int
        Channels (by in-window area) entering the co-elution mean.
    min_channels : int
        Minimum contributing channels. 2 is the floor.
    min_ms1_points : int
        Minimum run of consecutive scans above `min_peak_height` for a
        channel to count. Guards against isolated spikes, which correlate
        perfectly with each other.
    min_coelution : float
        Mean pairwise correlation floor across the selected channels.
    min_centrality : float
        The candidate scan must sit at least this fraction of the mono
        trace's window maximum. Rejects shoulder positions, which
        otherwise score well because their window still contains the peak.
    # min_mono_corr : float
    #     Mono must correlate at least this well with the other selected
    #     channels. This is the guard against heavy-channel-anchored
    #     interferences.
    w_envelope : float
        Weight on the fraction of channels contributing. A weak prior
        only -- a low-abundance precursor showing two isotopes is
        ordinary, and agreement matters more than channel count.
 
    Returns
    -------
    dict with `apex`, `score`, `status`, `evidence_tier="ms1_only"`, the
    combined `scores` profile, per-component profiles (including
    `mono_agreement`), `n_channels`, and the channels/window used at the
    apex. There is no `ms1_corr` -- no independent modality remains to
    correlate against.
    """
    ms1 = np.asarray(ms1, dtype=float)
    if ms1.ndim != 2:
        raise ValueError("ms1 must be 2D (n_rt, n_channels).")
 
    n_rt, n_ch = ms1.shape
    full_window = 2 * half_window + 1
    if n_rt < full_window:
        raise ValueError(
            f"RT range ({n_rt}) is shorter than the full window ({full_window}); "
            f"reduce half_window."
        )
    if n_ch < min_channels:
        return _ms1_no_signal(n_rt)
    if require_mono and not (0 <= mono_index < n_ch):
        raise ValueError("mono_index out of range.")
 
    coelution = np.full(n_rt, np.nan)
    mono_agreement = np.full(n_rt, np.nan)      # [CHANGE 2]
    raw_intensity = np.full(n_rt, np.nan)
    bell = np.full(n_rt, np.nan)
    centrality = np.full(n_rt, np.nan)
    envelope = np.full(n_rt, np.nan)
    n_channels = np.zeros(n_rt, dtype=int)
    channels_used = [None] * n_rt
    windows = [None] * n_rt
 
    # ------------------------------------------------------------------
    # pass 1
    # ------------------------------------------------------------------
    for c in range(n_rt):
        if c < half_window or c >= n_rt - half_window:
            continue
 
        start = c - half_window
        stop = c + half_window + 1
 
        window = ms1[start:stop, :]
        centre = c - start
 
        usable = _usable_columns(window, min_peak_height, min_ms1_points)
        # if c == 15:
        #     print(window)
        #     print(usable)
        #     break
 
        # [CHANGE 3] recorded before any gate, so a rejected position
        # still says WHY it was rejected when you inspect the profiles
        n_channels[c] = int(usable.size)
        envelope[c] = float(usable.size) / float(n_ch)
 
        if usable.size < min_channels:
            continue
        if require_mono and mono_index not in usable:
            continue
 
        totals = window.sum(axis=0)
        order = usable[np.argsort(totals[usable])[::-1]]
        selected = order[: min(top_k, order.size)]
 
        # keep the anchor in the block even if it is not top-k by area
        if require_mono and mono_index not in selected:
            selected = np.append(selected[: max(0, top_k - 1)], mono_index)
 
        if selected.size < min_channels:
            continue
 
        block = window[:, selected]
 
        mc = _mean_pairwise_corr(block, corr_method)
        if not np.isfinite(mc) or mc <= min_coelution:
            continue
        coelution[c] = mc
 
        # [CHANGE 4] mono handled separately from the sum
        mono_trace = _as_f64(window[:, mono_index])
 
        # [CHANGE 5] mono must AGREE with the envelope, not merely exist
        partners = [i for i in selected if i != mono_index]
        mono_corrs = [
            corr(mono_trace, _as_f64(window[:, i]), corr_method) for i in partners
        ]
        mono_corrs = [v for v in mono_corrs if np.isfinite(v)]
        if not mono_corrs:
            continue
        mono_agreement[c] = float(np.clip(np.mean(mono_corrs), 0.0, 1.0))
        # if mono_agreement[c] < min_mono_corr:
        #     continue
 
        # intensity stays on the SUM: total envelope signal
        summed = _as_f64(block.sum(axis=1))
        raw_intensity[c] = np.log2(1.0 + max(summed.sum(), 0.0))
 
        # [CHANGE 6] shape and centrality now come from MONO
        mono_smoothed = np.asarray(smooth(mono_trace), dtype=float)
 
        try:
            bs = float(calculate_shape_score(mono_smoothed, corr_method))
        except Exception:
            bs = np.nan
        bell[c] = np.clip(bs, 0.0, 1.0) if np.isfinite(bs) else 0.0
 
        top = mono_smoothed.max()
        centrality[c] = (
            float(np.clip(mono_smoothed[centre] / top, 0.0, 1.0)) if top > E else 0.0
        )
 
        # [CHANGE 8] gate here rather than in pass 2, so a gated position
        # does not leave a half-filled row of components behind
        if centrality[c] < min_centrality:
            continue
 
        channels_used[c] = selected
        windows[c] = slice(start, stop)
 
    # ------------------------------------------------------------------
    # pass 2: normalise intensity across positions, then combine
    # ------------------------------------------------------------------
    finite = np.isfinite(raw_intensity)
    intensity = np.full(n_rt, np.nan)
    if np.any(finite):
        peak_intensity = raw_intensity[finite].max()
        intensity[finite] = (
            raw_intensity[finite] / peak_intensity if peak_intensity > E else 0.0
        )
 
    # [CHANGE 7] mono_agreement joins the combination
    weights = np.array(
        [w_coelution, w_intensity, w_bell, w_centrality, w_mono, w_envelope],
        dtype=float,
    )
 
    scores = np.full(n_rt, np.nan)
    for c in range(n_rt):
        # windows[c] is set only for positions that cleared every gate
        if windows[c] is None:
            continue
        scores[c] = _geometric_mean(
            [
                coelution[c],
                intensity[c],
                bell[c],
                centrality[c],
                mono_agreement[c],
                envelope[c],
            ],
            weights,
        )
 
    valid = np.where(np.isfinite(scores))[0]
    if valid.size == 0:
        return _ms1_no_signal(                      # [CHANGE 9]
            n_rt,
            coelution=coelution,
            mono_agreement=mono_agreement,
            intensity=intensity,
            bell=bell,
            centrality=centrality,
            envelope=envelope,
            n_channels=n_channels,
        )
 
    apex = int(valid[np.argmax(scores[valid])])
    return {
        "apex": apex,
        "score": float(scores[apex]),
        "status": "found",
        "evidence_tier": "ms1_peak",
        "scores": scores,
        "coelution": coelution,
        "mono_agreement": mono_agreement,           # [CHANGE 9]
        "intensity": intensity,
        "bell": bell,
        "centrality": centrality,
        "envelope": envelope,
        "n_channels": n_channels,
        "best_fragments": channels_used[apex],
        "window": windows[apex],
    }
 
 
# [CHANGE 10] explicit keywords: the old `*profiles` form paired values to
# keys positionally and would silently mismatch if the key list changed
def _ms1_no_signal(
    n_rt,
    coelution=None,
    mono_agreement=None,
    intensity=None,
    bell=None,
    centrality=None,
    envelope=None,
    n_channels=None,
):
    def nan():
        return np.full(n_rt, np.nan)
 
    return {
        "apex": -1,
        "score": np.nan,
        "status": "no_signal",
        "evidence_tier": "ms1_peak",
        "scores": nan(),
        "coelution": nan() if coelution is None else coelution,
        "mono_agreement": nan() if mono_agreement is None else mono_agreement,
        "intensity": nan() if intensity is None else intensity,
        "bell": nan() if bell is None else bell,
        "centrality": nan() if centrality is None else centrality,
        "envelope": nan() if envelope is None else envelope,
        "n_channels": (
            np.zeros(n_rt, dtype=int) if n_channels is None else n_channels
        ),
        "best_fragments": np.array([], dtype=int),
        "window": None,
    }
 
 
def find_peaks_ms1(ms1, max_peaks=3, min_separation=None, min_score=0.0, **kwargs):
    """Greedy multi-apex extraction on the MS1-only score profile."""
    result = find_apex_ms1(ms1, **kwargs)
    scores = result["scores"].copy()
 
    if min_separation is None:
        min_separation = kwargs.get("half_window", 4)
 
    peaks = []
    while len(peaks) < max_peaks:
        valid = np.where(np.isfinite(scores))[0]
        if valid.size == 0:
            break
        idx = int(valid[np.argmax(scores[valid])])
        if scores[idx] < min_score:
            break
        peaks.append({"apex": idx, "score": float(scores[idx])})
        lo = max(0, idx - min_separation)
        hi = min(scores.size, idx + min_separation + 1)
        scores[lo:hi] = np.nan
 
    result["peaks"] = peaks
    return result


def find_peaks_any(
    ms2,
    ms1,
    center=None,
    scan_ms1_xics=False,
    allow_ms1_only=True,
    ms1_only_min_score=0.0,
    min_ms2_points=1,
    min_ms1_points=2,
    ms1_fwhm_tolerance=1,
    ms1_max_apex_distance=3,
    **kwargs,
):
    """
    Try the MS1+MS2 path; fall back to MS1-only when fragments are absent.
 
    Falls back on the *result*, not just on a cheap precheck, so a
    precursor whose fragments exist but never co-elute also routes to the
    MS1-only tier rather than being lost.
    """
    # Parameters prefixed with ms1_ are top-level aliases for the MS1
    # fallback and are translated to find_apex_ms1() names here.
    ms1_aliases = {
        "ms1_half_window": "half_window",
        "ms1_top_k": "top_k",
        "ms1_min_centrality": "min_centrality",
        "ms1_w_coelution": "w_coelution",
        "ms1_w_intensity": "w_intensity",
        "ms1_w_bell": "w_bell",
        "ms1_w_centrality": "w_centrality",
    }
    ms1_only_keys = {
        "min_channels", "require_mono", "mono_index", "w_envelope",
        "min_coelution", "w_mono",
    }
    alias_values = {
        dst: kwargs[src] for src, dst in ms1_aliases.items() if src in kwargs
    }
    ms2_only_alias_keys = set(ms1_aliases)
    shared = {
        k: v for k, v in kwargs.items()
        if k not in ms1_only_keys and k not in ms2_only_alias_keys
    }
    ms1_kwargs = {k: v for k, v in kwargs.items() if k in ms1_only_keys}
    ms1_kwargs.update({k: v for k, v in shared.items() if k != "ms1"})
    ms1_kwargs.update(alias_values)
    ms1_kwargs.pop("min_fragments", None)
    ms1_kwargs.pop("w_ms1_bonus", None)
 
    if _has_ms2_evidence(
        ms2,
        min_peak_height=kwargs.get("min_peak_height", 0.0),
        min_fragments=kwargs.get("min_fragments", 1),
        min_ms2_points=min_ms2_points,
    ):
        result = find_peaks(ms2, ms1=ms1, min_ms2_points=min_ms2_points, **shared)
        result["evidence_tier"] = "ms2_coelution"
        if result.get("peaks"):
            return result
 
    if not allow_ms1_only:
        result = find_peaks(ms2, ms1=ms1, min_ms2_points=min_ms2_points, **shared)
        result["evidence_tier"] = "prediction_only"
        return result

    ms1_kwargs["min_score"] = ms1_only_min_score
    ms1_kwargs["max_peaks"] = kwargs.get("max_peaks", 3)
    ms1_kwargs["min_separation"] = kwargs.get("min_separation", None)

    ms1_result = find_peaks_ms1(
        ms1, min_ms1_points=min_ms1_points, **ms1_kwargs
    )

    if scan_ms1_xics:
        ms1_result["evidence_tier"] = (
            "ms1_peak" if ms1_result.get("peaks") else "prediction_only"
        )
        return ms1_result

    if center is None:
        raise ValueError("center must be supplied when scan_ms1_xics=False.")
    all_ms1_peaks = list(ms1_result.get("peaks", []))
    overlapping_peaks = []
    annotated_ms1_peaks = []
    mono_index = int(ms1_kwargs.get("mono_index", 0))
    mono_trace = np.asarray(ms1[:, mono_index], dtype=float)
    half_window = int(ms1_kwargs.get("half_window", 4))

    for peak0 in all_ms1_peaks:
        peak = dict(peak0)
        apex = int(peak["apex"])
        lo = max(0, apex - half_window)
        hi = min(mono_trace.size, apex + half_window + 1)
        local_mono = np.ascontiguousarray(mono_trace[lo:hi], dtype=np.float64)
        peak.update(
            ms1_fwhm=0,
            ms1_fwhm_left=-1,
            ms1_fwhm_right=-1,
            center_distance=abs(float(apex) - float(center)),
            center_distance_fwhm=np.nan,
            fwhm_outside_distance=np.nan,
            overlaps_predicted_center_exact=False,
            overlaps_predicted_center=False,
        )
        if local_mono.size >= 2 and np.isfinite(local_mono).any():
            smoothed = np.asarray(smooth(local_mono), dtype=float)
            first, last, width = get_fwhm_bounds(smoothed)
            if width > 0:
                left = lo + int(first); right = lo + int(last)
                peak["ms1_fwhm"] = int(width)
                peak["ms1_fwhm_left"] = int(left)
                peak["ms1_fwhm_right"] = int(right)

                exact_overlap = bool(left <= center <= right)
                if center < left:
                    outside_distance = int(left - center)
                elif center > right:
                    outside_distance = int(center - right)
                else:
                    outside_distance = 0

                peak["overlaps_predicted_center_exact"] = exact_overlap
                peak["fwhm_outside_distance"] = int(outside_distance)
                fwhm_ok = outside_distance <= int(ms1_fwhm_tolerance)
                apex_ok = (
                    ms1_max_apex_distance is None
                    or peak["center_distance"] <= float(ms1_max_apex_distance)
                )
                peak["overlaps_predicted_center"] = bool(fwhm_ok and apex_ok)
                peak["center_distance_fwhm"] = (
                    peak["center_distance"] / max(float(width) / 2.0, 0.5)
                )
        annotated_ms1_peaks.append(peak)
        if peak["overlaps_predicted_center"]:
            overlapping_peaks.append(peak)

    ms1_result["all_ms1_peaks"] = annotated_ms1_peaks
    ms1_result["center_overlapping_ms1_peaks"] = overlapping_peaks
    ms1_result["peaks"] = overlapping_peaks
    ms1_result["evidence_tier"] = "ms1_peak" if overlapping_peaks else "prediction_only"
    return ms1_result


def jumplion_recovery_workflow():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    
    COLORS = {
    "prediction_only": "#7F7F7F",
    "ms1_peak": "#D89000",
    "ms2_coelution": "#3B82B4",
    "reported": "#222222",
    }
    
    fig, ax = plt.subplots(figsize=(3.82, 3.8))
    ax.set_xlim(1.9, 9.1)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, text, face="white", fontsize=8):
        p = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=face,
            edgecolor="#444444",
            linewidth=0.9,
        )
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=1,
                color="#555555",
            ),
        )
        if label:
            ax.text(lx, ly, label, ha="center", va="center", fontsize=7)

    # ------------------------------------------------------------
    # Predicted coordinate -> extraction
    # ------------------------------------------------------------
    box(5, 7.0, 3.5, 0.65, "Predicted RT")
    arrow(5, 6.67, 5, 6.18)

    box(5, 5.85, 3.5, 0.65, "RT search region")
    arrow(5, 5.52, 5, 5.03)

    box(5, 4.70, 3.5, 0.65, "XIC extraction")
    arrow(5, 4.37, 5, 3.88)

    box(5, 3.55, 3.5, 0.65, "Qualifying MS2 co-elution?")


    # ------------------------------------------------------------
    # MS2 branch
    # ------------------------------------------------------------
    box(
        3.6, 2.05, 2.4, 0.8,
        "MS2 co-elution\n(MS2-refined RT)",
        COLORS["ms2_coelution"],
    )
    arrow(
        4.99, 3.2,
        3.85, 2.47,
        "yes",
        3.75, 2.8,
    )

    # ------------------------------------------------------------
    # Prediction-centered branch
    # ------------------------------------------------------------
    box(
        6.75, 2.05, 3.5, 0.8,
        "Qualifying MS1 isotope peak \nnear predicted RT?",
    )
    arrow(
        5.01, 3.2,
        6.15, 2.47,
        "no",
        6.05, 2.8,
    )

    # ------------------------------------------------------------
    # Final prediction-centered tiers
    # ------------------------------------------------------------
    box(
        4.8, 0.55, 2.4, 0.8,
        "MS1 isotope peak\n(MS1-refined RT)",
        COLORS["ms1_peak"],
    )
    box(
        7.6, 0.55, 2.8, 0.8,
        "No qualifying peak\n(Predicted RT)",
        COLORS["prediction_only"],
    )
    arrow(
        6.39, 1.65,
        5.25, 0.985,
        "yes",
        5.10, 1.28,
    )
    arrow(
        6.41, 1.65,
        7.55, 0.985,
        "no",
        7.70, 1.28,
    )

    fig.tight_layout()