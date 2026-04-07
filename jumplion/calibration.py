# builtin
from typing import List, Tuple
from pathlib import Path
# external
import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
# local
from jumplion.utils import E

def split_into_rt_bins(
        sorted_rt_values, 
        max_num_bins=1, 
        min_points_per_bin=500, 
        min_rt_length_per_bin=30, 
        tolerance=E
        ):
    """
    Split data points into RT (retention time) bins for mass calibration.
    
    Parameters:
    - num_data_points: int, total number of data points (e.g., m/z values)
    - rt_values: list or array, retention time (RT) values for the data points
    - max_num_bins: int, maximum number of RT bins allowed
    - min_points_per_bin: int, minimum number of data points required per bin
    - tolerance: float, minimum allowable difference between bin centers to avoid duplicates
    
    Returns:
    - num_bins: int, actual number of RT bins created
    - rt_bin_boundaries: list, RT boundaries of each bin
    - rt_bin_centers: list, RT center of each bin
    """
    
    # Sort the RT values
    #rt_values.sort()  # sorts the list ascending by default
    n_rts = len(sorted_rt_values)
    
    # Determine the number of bins based on data size and bin constraints
    num_bins = max(1, 
                   min(max_num_bins, 
                       n_rts // min_points_per_bin, 
                       int(sorted_rt_values[-1] - sorted_rt_values[0]) // min_rt_length_per_bin)
                   )
    
    # Calculate the RT boundaries of each bin
    bin_step = (n_rts - 1) / num_bins
    rt_bin_boundaries = [sorted_rt_values[min(int(bin_step * i), n_rts - 1)] for i in range(num_bins)]
    rt_bin_boundaries.append(sorted_rt_values[n_rts - 1])  # Add the last boundary

    # Calculate the RT center of each bin
    rt_bin_centers = [0.5 * (rt_bin_boundaries[i] + rt_bin_boundaries[i + 1]) for i in range(num_bins)]
    
    # Remove bins with duplicate or very close centers
    i = 1
    while i < num_bins:
        if rt_bin_centers[i] - rt_bin_centers[i - 1] < tolerance:
            # Remove the close/duplicate center and corresponding boundary
            rt_bin_centers.pop(i)
            rt_bin_boundaries.pop(i)
            num_bins -= 1  # Decrease the bin count
        else:
            i += 1
    
    return num_bins, rt_bin_boundaries, rt_bin_centers


def build_mass_cali_model(theoretical_mzs, delta_mzs, rts, RTBinCenter, nRTBins):
    """
    Build a mass calibration model using theoretical m/z, delta m/z, and retention times (RTs).
    This model fits: delta_mz = beta0 * mz^2 + beta1 + beta2 * mz, if nRTBins = 1
    
    Parameters:
    - theoretical_mzs: list or array of theoretical m/z values
    - delta_mzs: list or array of delta m/z values (observed - theoretical)
    - rts: list or array of retention times
    
    Returns:
    - X: array of coefficients for the mass calibration model
    """
    
    mzs = theoretical_mzs   # Theoretical m/z values
    mzws = 1 / mzs          # Parts per million (ppm) error
    
    b = mzws * delta_mzs / mzs  # delta mz divided by mz^2

    TL = []
    for idx, rt in enumerate(rts):
        TL.append((idx, 0, 1))   # beta0 * mz^2
        
        # If rt is within the first bin
        if rt <= RTBinCenter[0]:
            TL.append((idx, 1, mzws[idx] * 1.0 / mzs[idx]))              # beta1 (constant term)
            TL.append((idx, 2, mzws[idx] * 1.0))                         # beta2 * mz
        # If rt is within the last bin
        elif rt >= RTBinCenter[-1]:
            TL.append((idx, 1 + (nRTBins - 1) * 2, mzws[idx] * 1.0 / mzs[idx]))
            TL.append((idx, 2 + (nRTBins - 1) * 2, mzws[idx] * 1.0))
        # If rt is within an intermediate bin
        else:
            for bin_idx in range(1, nRTBins):
                if rt < RTBinCenter[bin_idx]:
                    u = rt - RTBinCenter[bin_idx - 1]
                    v = RTBinCenter[bin_idx] - rt
                    w = u + v
                    if w > E:
                        cl = v / w
                        cr = u / w
                        TL.append((idx, 1 + (bin_idx - 1) * 2, mzws[idx] * cl / mzs[idx]))
                        TL.append((idx, 2 + (bin_idx - 1) * 2, mzws[idx] * cl))
                        TL.append((idx, 1 + bin_idx * 2, mzws[idx] * cr / mzs[idx]))
                        TL.append((idx, 2 + bin_idx * 2, mzws[idx] * cr))
                    break
        

    # Construct sparse matrix A
    A = sparse.csr_matrix((
        [t[2] for t in TL],  # The values in the sparse matrix
        ([t[0] for t in TL], [t[1] for t in TL])  # The row and column indices
    ), shape=(len(b), 1 + 2 * nRTBins))

    # Solve for X using the least squares solution
    X = linalg.spsolve(A.T @ A, A.T @ b)
    
    return X


@njit(parallel=True)
def predicted_mz_parallel(t: np.ndarray, mzs: np.ndarray, rts: np.ndarray, RTBinCenter: float, nRTBins: int) -> np.ndarray:
    if mzs.ndim != 2:
        raise ValueError("mzs must be a 2D array")
    
    if rts.ndim != 1:
        raise ValueError("rts must be a 1D array")
    
    if mzs.shape[0] != rts.shape[0]:
        raise ValueError("Number of rows in mzs must match the length of rts")
    
    pmz = np.zeros(mzs.shape)
    for i in prange(mzs.shape[0]):
        for j in range(mzs.shape[1]):
            pmz[i, j] = predicted_mz(t, mzs[i, j], rts[i], RTBinCenter, nRTBins)
    
    return pmz


@njit
def predicted_mz(t, mz, rt, RTBinCenter, nRTBins):
    # polynomial regression 
    # delta_mz = beta0 * mz^2 + beta1 + beta2 * mz, if nRTBins = 1
    delta_mz = t[0] * mz * mz 
    
    if rt <= RTBinCenter[0]:
        delta_mz += t[1] + t[2] * mz
    elif rt >= RTBinCenter[nRTBins - 1]:
        delta_mz += t[1 + (nRTBins - 1) * 2] + t[2 + (nRTBins - 1) * 2] * mz
    else:
        for i in range(1, nRTBins):
            if rt < RTBinCenter[i]:
                u = rt - RTBinCenter[i - 1]
                v = RTBinCenter[i] - rt
                w = u + v
                if w > E:
                    # calculate a weighted sum
                    delta_mz += ((t[1 + (i - 1) * 2] + t[2 + (i - 1) * 2] * mz) * v + 
                                   (t[1 + i * 2] + t[2 + i * 2] * mz) * u) / w
                break

    return mz + delta_mz


def density_scatter(x, y, ax, bins=100, **kwargs):
    """Create a density scatter plot using 2D histogram binning."""

    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    _, xedges, yedges = data.T.flatten(), x_e, y_e

    # Assign a density value to each point based on its bin
    x_idx = np.digitize(x, xedges) - 1
    y_idx = np.digitize(y, yedges) - 1
    valid = (x_idx >= 0) & (x_idx < data.shape[0]) & (y_idx >= 0) & (y_idx < data.shape[1])
    z_vals = np.zeros_like(x, dtype=float)
    z_vals[valid] = data[x_idx[valid], y_idx[valid]]

    sc = ax.scatter(x, y, c=z_vals, cmap='viridis', edgecolor='none', **kwargs)
    return sc


def mass_calibration_plot(df, ms_level, bins=100, s=30, run_name="", save_plot=True, out_dir=""):
    """
    Generate improved mass calibration visualization plots.
    
    Parameters:
    -----------
    ms_cal_file : str
        Path to the MS calibration file (CSV format)
    ms_level : str
        MS level identifier (e.g., 'ms1', 'ms2')
    RTBinCenter : array-like
        Center points for retention time bins
    nRTBins : int
        Number of retention time bins
    run_name : str, optional
        Prefix for the output filename
    """

    
    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Load data
    #df = pd.read_csv(ms_cal_file)
    rt = df['experimental_rt'].to_numpy()
    theo_mz = df['theoretical_mz_'+ms_level].to_numpy()
    
    ppm_old = (df['mass_delta_'+ms_level] * 1e6 / theo_mz).to_numpy()
    ppm_new = ((df['observed_mz_'+ms_level] - df['query_mz_'+ms_level]) * 1e6 / df['query_mz_'+ms_level]).to_numpy()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, width_ratios=[3, 3, 1])
    
    # Calculate statistics
    stats_old = {
        'mean': np.mean(ppm_old),
        'median': np.median(ppm_old),
        'std': np.std(ppm_old),
    }
    
    stats_new = {
        'mean': np.mean(ppm_new),
        'median': np.median(ppm_new),
        'std': np.std(ppm_new),
    }
    
    # Calculate y-axis limits based on data
    y_max = max(np.percentile(np.abs(ppm_old), 99.5), np.percentile(np.abs(ppm_new), 99.5)) * 1.2
    y_min = -y_max
    

    # RT vs ppm plots (before calibration)
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = density_scatter(rt, ppm_old, ax1, bins=bins, s=s)
    ax1.set_ylim([y_min, y_max])
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax1.axhline(y=stats_old['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax1.set_title('Before Mass Calibration', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Retention Time (min)', fontsize=12)
    ax1.set_ylabel('Mass Error (ppm)\n(Observed - Theoretical)', fontsize=12)
    
    # Add colorbar for density
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Density', fontsize=10)
    
    # RT vs ppm plots (after calibration)
    ax2 = fig.add_subplot(gs[1, 0])
    sc2 = density_scatter(rt, ppm_new, ax2, bins=bins, s=s)
    ax2.set_ylim([y_min, y_max])
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax2.axhline(y=stats_new['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax2.set_title('After Mass Calibration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Retention Time (min)', fontsize=12)
    ax2.set_ylabel('Mass Error (ppm)\n(Observed - Query)', fontsize=12)
    
    # Add colorbar for density
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Density', fontsize=10)
    
    # Histograms
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    sns.histplot(y=ppm_old, ax=ax3, kde=True, color='gray', edgecolor='black', alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax3.axhline(y=stats_old['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    #ax3.set_xlabel('Mass Error (ppm)', fontsize=12)
    ax3.set_xlabel('Count', fontsize=12)
    
    ax4 = fig.add_subplot(gs[1, 2], sharey=ax2)
    sns.histplot(y=ppm_new, ax=ax4, kde=True, color='gray', edgecolor='black', alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax4.axhline(y=stats_new['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    #ax4.set_xlabel('Mass Error (ppm)', fontsize=12)
    ax4.set_xlabel('Count', fontsize=12)
    
    # m/z vs ppm plots (before calibration)
    ax5 = fig.add_subplot(gs[0, 1])
    sc3 = density_scatter(theo_mz, ppm_old, ax5, bins=bins, s=s)
    ax5.set_ylim([y_min, y_max])
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax5.axhline(y=stats_old['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax5.set_title('Before Mass Calibration', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Theoretical m/z', fontsize=12)
    #ax5.set_ylabel('Mass Error (ppm)\n(Observed - Theoretical)', fontsize=12)
    
    # Add colorbar for density
    cbar3 = plt.colorbar(sc3, ax=ax5)
    cbar3.set_label('Density', fontsize=10)
    
    # m/z vs ppm plots (after calibration)
    ax6 = fig.add_subplot(gs[1, 1])
    sc4 = density_scatter(theo_mz, ppm_new, ax6, bins=bins, s=s)
    ax6.set_ylim([y_min, y_max])
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax6.axhline(y=stats_new['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax6.set_title('After Mass Calibration', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Theoretical m/z', fontsize=12)
    #ax6.set_ylabel('Mass Error (ppm)\n(Observed - Query)', fontsize=12)
    
    # Add colorbar for density
    cbar4 = plt.colorbar(sc4, ax=ax6)
    cbar4.set_label('Density', fontsize=10)
    
    # Add statistical information as text
    stats_text_old = (f"Mean: {stats_old['mean']:.2f} ppm\n"
                      f"Median: {stats_old['median']:.2f} ppm\n"
                      f"Std Dev: {stats_old['std']:.2f} ppm")
                      #f"IQR: [{stats_old['q1']:.2f}, {stats_old['q3']:.2f}] ppm")
    
    stats_text_new = (f"Mean: {stats_new['mean']:.2f} ppm\n"
                      f"Median: {stats_new['median']:.2f} ppm\n"
                      f"Std Dev: {stats_new['std']:.2f} ppm")
                      #f"IQR: [{stats_new['q1']:.2f}, {stats_new['q3']:.2f}] ppm")
    
    ax3.text(0.95, 0.02, stats_text_old, transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    
    ax4.text(0.95, 0.02, stats_text_new, transform=ax4.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    
    # Add title with MS level
    fig.suptitle(f'{ms_level.upper()}-level Mass Calibration', fontsize=16, fontweight='bold', y=0.99)
    fig.text(0.5, 0.94, f'# of data points {len(df)}', 
             horizontalalignment='center', fontsize=12, fontstyle='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    if save_plot:
        output_file = out_dir / f"{run_name}_{ms_level}_calibration_plot.pdf"
        fig.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    
def im_error_plot(df, run_name="", bins=100, s=30, save_plot=True, out_dir=""):    
    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Load data
    #df = pd.read_csv(ms_cal_file)
    theo_mz = df['theoretical_mz'].to_numpy()
    query_im = df['query_im'].to_numpy()
    delta_im = df['im_delta'].to_numpy()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[3, 3, 1])
    
    # Calculate statistics
    stats = {
        'mean': np.mean(delta_im),
        'median': np.median(delta_im),
        'std': np.std(delta_im),
    }
    
    # RT vs ppm plots (after calibration)
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = density_scatter(theo_mz, query_im, ax1, bins=bins, s=s)
    ax1.set_title('Precursor ion mobility vs m/z', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Theoretical m/z', fontsize=12)
    ax1.set_ylabel('Query IM', fontsize=12)
    
    # Add colorbar for density
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Density', fontsize=10)
    
    # Calculate y-axis limits based on data
    y_max = np.percentile(np.abs(delta_im), 99.5) * 1.2
    y_min = -y_max
    
    # RT vs ppm plots (before calibration)
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = density_scatter(query_im, delta_im, ax2, bins=bins, s=s)
    ax2.set_ylim([y_min, y_max])
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax2.axhline(y=stats['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax2.set_title('IM error vs. Query IM', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Query IM', fontsize=12)
    ax2.set_ylabel('IM Error\n(Observed - Query)', fontsize=12)
    
    # Add colorbar for density
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Density', fontsize=10)
    
    # Histograms
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)
    sns.histplot(y=delta_im, ax=ax3, kde=True, color='gray', edgecolor='black', alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, lw=1.5)
    ax3.axhline(y=stats['mean'], color='purple', linestyle='-', alpha=0.7, lw=1.5)
    ax3.set_xlabel('Count', fontsize=12)
    
    # Add statistical information as text
    stats_text = (f"Mean: {stats['mean']:.4f}\n"
                    f"Median: {stats['median']:.4f}\n"
                    f"Std Dev: {stats['std']:.5f}")
                    #f"IQR: [{stats['q1']:.4f}, {stats['q3']:.4f}]")
    
    ax3.text(0.95, 0.02, stats_text, transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    
    # Add title with MS level
    fig.suptitle('IM window setting', fontsize=16, fontweight='bold', y=0.99)
    fig.text(0.5, 0.9, f'# of data points {len(df)}', 
             horizontalalignment='center', fontsize=12, fontstyle='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_plot:
        # Save figure
        output_file = out_dir / f"{run_name}_im_error_plot.pdf"
        fig.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
############################ RT/IM spline modeling ############################
def convert_to_arrays(data):
    x = np.array([p[0] for p in data])
    y = np.array([p[1] for p in data])
    return x, y


def flip_map(data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(p[1], p[0]) for p in data]


@njit
def PAVA(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    index = np.zeros(n + 1, dtype=np.int32)
    weights = np.ones(n)
    adjusted = np.zeros(n)

    result = np.zeros(n)

    adjusted[0] = y[0]
    j = 0
    index[0] = 0
    index[1] = 1

    for i in range(1, n):
        j += 1
        adjusted[j] = y[i]
        weights[j] = 1.0
        while j > 0 and adjusted[j] < adjusted[j - 1]:
            adjusted[j - 1] = (weights[j] * adjusted[j] + weights[j - 1] * adjusted[j - 1]) / (weights[j] + weights[j - 1])
            weights[j - 1] += weights[j]
            j -= 1
        index[j + 1] = i + 1

    for k in range(j + 1):
        result[index[k]:index[k + 1]] = adjusted[k]

    return result


def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    AT = A.T
    ATA = np.dot(AT, A)
    ATb = np.dot(AT, b)
    return np.linalg.solve(ATA, ATb)


def spline(x: np.ndarray, y: np.ndarray, segments: int) -> tuple:
    # Number of data points
    n = len(x)
    
    # Apply Pool-Adjacent-Violators Algorithm to smooth the y values
    pava_y = PAVA(x, y)
    
    # Number of segments for the spline
    m = segments

    # Initialize arrays for segment boundaries and knot points
    segment_bounds = np.zeros(m + 1)
    knot_points = np.zeros(m)

    # Calculate segment boundaries
    split_interval = (n - 1) / m
    for i in range(m):
        # Distribute segment boundaries evenly across the x range
        segment_bounds[i] = x[min(int(split_interval * i), n - 1)]
    segment_bounds[m] = x[n - 1]  # Ensure the last boundary is the last x value

    # Calculate initial knot points as midpoints of segment boundaries
    for i in range(m):
        knot_points[i] = 0.5 * (segment_bounds[i] + segment_bounds[i + 1])

    # Remove knot points that are too close to each other
    valid_knots = [knot_points[0]]
    for i in range(1, m):
        if knot_points[i] - valid_knots[-1] >= 1e-8:
            valid_knots.append(knot_points[i])
    
    # Convert valid knots back to a numpy array and update m
    knot_points = np.array(valid_knots)
    m = len(knot_points)

    # Initialize coefficient array and design matrix A
    coefficients = np.zeros(2 * m)
    A = np.zeros((n, 2 * m))

    # Populate the design matrix A
    for i, x_val in enumerate(x):
        if x_val <= knot_points[0]:
            # For x values before the first knot point
            A[i, 0] = 1.0
            A[i, 1] = x_val - knot_points[0]
        else:
            # Find the appropriate knot point interval
            k = np.searchsorted(knot_points, x_val, side='right') - 1
            if k == m - 1:
                # For x values after the last knot point
                A[i, k * 2] = 1.0
                A[i, k * 2 + 1] = x_val - knot_points[k]
            else:
                # For x values between knot points
                w = knot_points[k + 1] - knot_points[k]
                u = (x_val - knot_points[k]) / w
                v = u - 1.0
                # Calculate cubic spline basis functions
                A[i, k * 2] = (1.0 + 2.0 * u) * v ** 2
                A[i, k * 2 + 1] = w * u * v ** 2
                A[i, (k + 1) * 2] = u ** 2 * (1.0 - 2.0 * v)
                A[i, (k + 1) * 2 + 1] = w * u ** 2 * v

    # Solve the least squares problem to find the coefficients
    coefficients = solve_least_squares(A, pava_y)

    return coefficients, knot_points


@njit
def calc_spline(coefficients: np.ndarray, knot_points: np.ndarray, x: float) -> float:
    m = len(knot_points)
    if m == 0:
        return 0.0

    if x <= knot_points[0]:
        return coefficients[0] + coefficients[1] * (x - knot_points[0])
    elif x >= knot_points[m - 1]:
        return coefficients[(m - 1) * 2] + coefficients[(m - 1) * 2 + 1] * (x - knot_points[m - 1])

    k = np.searchsorted(knot_points, x, side='right') - 1
    w = knot_points[k + 1] - knot_points[k]
    u = (x - knot_points[k]) / w
    v = u - 1.0
    result = coefficients[k * 2] * (1.0 + 2.0 * u) * v ** 2
    result += coefficients[k * 2 + 1] * w * u * v ** 2
    result += coefficients[(k + 1) * 2] * u ** 2 * (1.0 - 2.0 * v)
    result += coefficients[(k + 1) * 2 + 1] * w * u ** 2 * v
    return result

@njit
def calc_spline_vectorized(coefficients: np.ndarray, knot_points: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    m = len(knot_points)
    n = len(x_vals)
    results = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x = x_vals[i]
        if m == 0:
            results[i] = 0.0
            continue

        if x <= knot_points[0]:
            results[i] = coefficients[0] + coefficients[1] * (x - knot_points[0])
        elif x >= knot_points[m - 1]:
            results[i] = coefficients[(m - 1) * 2] + coefficients[(m - 1) * 2 + 1] * (x - knot_points[m - 1])
        else:
            k = np.searchsorted(knot_points, x, side='right') - 1
            w = knot_points[k + 1] - knot_points[k]
            u = (x - knot_points[k]) / w
            v = u - 1.0
            res = coefficients[k * 2] * (1.0 + 2.0 * u) * v ** 2
            res += coefficients[k * 2 + 1] * w * u * v ** 2
            res += coefficients[(k + 1) * 2] * u ** 2 * (1.0 - 2.0 * v)
            res += coefficients[(k + 1) * 2 + 1] * w * u ** 2 * v
            results[i] = res

    return results


def refine_spline(x: np.ndarray, y: np.ndarray, segments: int) -> tuple:
    
    coefficients, knot_points = spline(x, y, segments)
    if len(x) < 3:
        return coefficients, knot_points

    differences = np.abs(y - np.array([calc_spline(coefficients, knot_points, xi) for xi in x]))
    threshold = np.sort(differences)[int(0.8 * len(differences))]
    mask = differences <= threshold + 1e-8
    filtered_x, filtered_y = x[mask], y[mask]

    coefficients, knot_points = spline(filtered_x, filtered_y, segments)
    
    return coefficients, knot_points
    

def check_model_accuracy(data, coeff, points, delta_percentile=0.8):
    
    # calculate difference between experimental and predicted  
    delta = [abs(pair[1] - calc_spline(coeff, points, pair[0])) for pair in data]
    # sort delta in ascending order (from small to large)
    delta.sort()
    # 
    acc = delta[int(delta_percentile * len(delta))]
    
    return acc

    
def spline_fitting_plot(
        data: List[Tuple[float, float]], 
        coefficients: np.ndarray, 
        knot_points: np.ndarray,
        window,
        show_window,
        model,
        out_dir,
        run_name,
        figsize=(8, 8),
        save_plot=True
        ):
    
    x, y = convert_to_arrays(data)
    y_pava = PAVA(x, y)
    x_spline = np.linspace(min(x), max(x), 500)
    y_spline = np.array([calc_spline(coefficients, knot_points, xi) for xi in x_spline])

    # Create the grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[1:, :-1])  # Main scatter and spline plot
    ax_x_hist = fig.add_subplot(gs[0, :-1], sharex=ax_main)  # Histogram for x
    ax_y_hist = fig.add_subplot(gs[1:, -1], sharey=ax_main)  # Histogram for y

    # Scatter plot of original data
    ax_main.scatter(x, y, label='Original Data', color='blue', alpha=0.5)
    ax_main.plot(x, y_pava, label='PAVA Data', color='orange', linestyle='--')
    ax_main.plot(x_spline, y_spline, label='Spline Fit', color='red')

    if show_window:
        ax_main.plot(x_spline, y_spline + window, label='Margin', color='pink', linestyle='--')
        ax_main.plot(x_spline, y_spline - window, color='pink', linestyle='--')

    ax_main.scatter(knot_points, 
                    [calc_spline(coefficients, knot_points, k) for k in knot_points], 
                    label='Knot Points', color='green', marker='x')

    ax_main.set_xlabel(f'library i{model}')
    ax_main.set_ylabel(f'experiment {model}')
    ax_main.set_title(f'Model for predicting {model} from i{model}')
    ax_main.legend()
    ax_main.grid(True)

    # Histogram on the x-axis
    ax_x_hist.hist(x, bins=len(knot_points), color='gray', alpha=0.5)
    ax_x_hist.set_ylabel('Count')
    ax_x_hist.yaxis.set_visible(True)

    # Histogram on the y-axis
    ax_y_hist.hist(y, bins=len(knot_points), orientation='horizontal', color='gray', alpha=0.5)
    ax_y_hist.set_xlabel('Count')
    ax_y_hist.xaxis.set_visible(True)
    
    plt.setp(ax_x_hist.get_xticklabels(), visible=False)
    plt.setp(ax_y_hist.get_yticklabels(), visible=False)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(Path(out_dir) / f'{run_name}_{model}_fitting_plot.pdf')
        plt.close()
    else:
        plt.show()
