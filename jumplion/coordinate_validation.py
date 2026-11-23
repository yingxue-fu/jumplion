"""Generic RT/IM model validation and production fitting

This module builds a validation logic to any coordinate system
that can be modeled with a run-specific fit function and evaluated against
held-out observed coordinates.

The workflow is standard and statistically clean:

5-fold CV → estimate expected prediction error

then:

refit using all available observations → predict genuinely missing RTs


It uses a leakage-safe structure:
1. mask held-out coordinates before anchor selection and model fitting;
2. fit the model on training anchors only;
3. predict held-out coordinates;
4. apply the same post-prediction adjustment used in production;
5. summarize out-of-fold errors and window coverage.

"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

FitFunction = Callable[[np.ndarray, np.ndarray, int], Tuple[Any, Any]]
PredictFunction = Callable[[Any, Any, np.ndarray], np.ndarray]
StepSizeInput = Union[float, Mapping[str, float]]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by cross-validation and production fitting."""

    # Cross-validation
    n_splits: int = 5
    random_state: int = 42

    # Anchor-selection rule
    min_run_fraction: float = 0.5
    min_runs_floor: int = 3

    # one of: "cv", "mad", "sd"
    variability_metric: str = "mad"
    trim_pct: float = 5.0

    # Model complexity
    max_segments: int = 30
    min_points_per_bin: int = 20
    min_training_points: int = 10

    # Cross-run safeguard for predicted coordinates
    adjustment_mode: str = "clip"  # "clip", "median", or "none"
    adjustment_lower_quantile: float = 0.005
    adjustment_upper_quantile: float = 0.995
    adjustment_by_run: bool = True
    min_points_for_run_bounds: int = 50

    # Production behavior for coordinates outside the training range
    allow_extrapolation: bool = True

    # Optional metadata for labeling outputs
    coordinate_label: str = "RT"

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not (0 < self.min_run_fraction <= 1):
            raise ValueError("min_run_fraction must be in (0, 1]")
        if self.min_runs_floor < 1:
            raise ValueError("min_runs_floor must be at least 1")
        if self.variability_metric not in {"cv", "mad", "sd"}:
            raise ValueError("variability_metric must be 'cv', 'mad', or 'sd'")
        if not (0 <= self.trim_pct < 100):
            raise ValueError("trim_pct must be in [0, 100)")
        if self.max_segments < 1:
            raise ValueError("max_segments must be at least 1")
        if self.min_points_per_bin < 1:
            raise ValueError("min_points_per_bin must be at least 1")
        if self.min_training_points < 3:
            raise ValueError("min_training_points must be at least 3")
        if self.adjustment_mode not in {"clip", "median", "none"}:
            raise ValueError("adjustment_mode must be 'clip', 'median', or 'none'")
        if not (
            0 <= self.adjustment_lower_quantile
            < self.adjustment_upper_quantile
            <= 1
        ):
            raise ValueError("adjustment quantiles must satisfy 0 <= lower < upper <= 1")


@dataclass
class CoordinateProductionResult:
    """Outputs from final production-model fitting."""

    raw_coordinate_matrix: pd.DataFrame
    final_coordinate_matrix: pd.DataFrame
    coordinate_source_matrix: pd.DataFrame
    prediction_metadata: pd.DataFrame
    fit_metrics: pd.DataFrame
    models: Dict[str, Dict[str, Any]]
    min_runs_model: int


def resolve_min_runs_model(
    n_runs: int,
    config: ModelConfig,
    min_runs_model: Optional[int] = None,
) -> int:
    """Return the exact minimum-run threshold used by both cross-validation and production."""

    if n_runs < 1:
        raise ValueError("n_runs must be positive")

    if min_runs_model is None:
        threshold = max(
            config.min_runs_floor,
            int(math.ceil(config.min_run_fraction * n_runs)),
        )
        threshold = min(n_runs, threshold)
    else:
        threshold = int(min_runs_model)
        if threshold < 1 or threshold > n_runs:
            raise ValueError(
                f"min_runs_model must be between 1 and {n_runs}; got {threshold}"
            )

    return threshold


def _validate_inputs(
    mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    coordinate_column: str,
) -> pd.Series:
    if mat.index.has_duplicates:
        raise ValueError("The coordinate matrix index must contain unique precursor IDs")
    if mat.columns.has_duplicates:
        raise ValueError("The coordinate matrix columns must contain unique run names")
    if "Precursor.Id" not in lib_df_uniq_prec.columns:
        raise KeyError("lib_df_uniq_prec must contain a 'Precursor.Id' column")
    if coordinate_column not in lib_df_uniq_prec.columns:
        raise KeyError(
            f"lib_df_uniq_prec must contain a '{coordinate_column}' column"
        )

    library = lib_df_uniq_prec[["Precursor.Id", coordinate_column]].copy()
    if library["Precursor.Id"].duplicated().any():
        duplicated = library.loc[
            library["Precursor.Id"].duplicated(keep=False), "Precursor.Id"
        ].head(5)
        raise ValueError(
            "Precursor.Id must be unique in lib_df_uniq_prec. Examples of "
            f"duplicates: {duplicated.tolist()}"
        )

    library.set_index("Precursor.Id", inplace=True)
    return pd.to_numeric(library[coordinate_column], errors="coerce")



def calculate_coordinate_stats(mat: pd.DataFrame) -> pd.DataFrame:
    """Calculate row-wise RT statistics used for anchor selection."""

    numeric = mat.astype(float)
    median = numeric.median(axis=1, skipna=True)
    absolute_deviation = numeric.sub(median, axis=0).abs()

    stats = pd.DataFrame(index=mat.index)
    stats["n_observed"] = numeric.notna().sum(axis=1).astype(int)
    stats["mean"] = numeric.mean(axis=1, skipna=True)
    stats["median"] = median
    stats["sd"] = numeric.std(axis=1, skipna=True)
    stats["mad"] = absolute_deviation.median(axis=1, skipna=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        stats["cv"] = stats["sd"] / stats["mean"].abs()

    stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return stats


def select_model_precursors(
    mat: pd.DataFrame,
    min_runs_model: int,
    config: ModelConfig,
) -> Tuple[pd.Index, pd.DataFrame, float]:
    """Select stable cross-run precursors without using held-out RT cells.

    During cross-validation, call this function only after setting the held-out
    precursor-run coordinates to missing. This ensures that the held-out RT does
    not affect the observed-run count, variability filter, or spline complexity.
    """

    stats = calculate_coordinate_stats(mat)
    eligible = stats["n_observed"] >= min_runs_model

    metric_name = config.variability_metric
    metric = stats[metric_name]
    finite_eligible_metric = metric[eligible & metric.notna()]

    if finite_eligible_metric.empty:
        raise ValueError(
            "No eligible precursors have a finite variability statistic after "
            f"requiring at least {min_runs_model} observed runs"
        )

    if config.trim_pct > 0:
        threshold = float(
            finite_eligible_metric.quantile(1.0 - config.trim_pct / 100.0)
        )
        eligible &= metric.notna() & (metric <= threshold)
    else:
        threshold = float("inf")
        eligible &= metric.notna()

    selected = stats.index[eligible]
    if selected.empty:
        raise ValueError("No calibration precursors remain after filtering")

    return selected, stats, threshold


def choose_n_segments(n_points: int, config: ModelConfig) -> int:
    """Choose spline complexity from the actual run-level training size."""

    if n_points < 1:
        raise ValueError("n_points must be positive")

    return min(
        config.max_segments,
        max(
            1,
            int(2.0 * np.sqrt(float(n_points) / config.min_points_per_bin)),
        ),
    )


def prepare_run_anchors(
    mat: pd.DataFrame,
    run_name: str,
    library_coordinate: pd.Series,
    min_runs_model: int,
    config: ModelConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the training-anchor table for one run."""

    if run_name not in mat.columns:
        raise KeyError(f"Run '{run_name}' is not present in the coordinate matrix")

    selected_ids, stats, variability_threshold = select_model_precursors(
        mat=mat,
        min_runs_model=min_runs_model,
        config=config,
    )

    anchors = pd.DataFrame(
        {
            "library_coordinate": library_coordinate.reindex(selected_ids),
            "observed_coordinate": mat.loc[selected_ids, run_name],
        },
        index=selected_ids,
    )
    anchors.index.name = "precursor_id"
    anchors = anchors.dropna(subset=["library_coordinate", "observed_coordinate"])
    anchors = anchors.astype(float)
    anchors.sort_values(
        by=["library_coordinate", "observed_coordinate"],
        inplace=True,
        kind="mergesort",
    )

    if len(anchors) < config.min_training_points:
        raise ValueError(
            f"Run '{run_name}' has only {len(anchors)} usable anchors; "
            f"at least {config.min_training_points} are required"
        )

    metadata = {
        "run_name": run_name,
        "n_selected_precursors_global": int(len(selected_ids)),
        "n_run_anchors": int(len(anchors)),
        "min_runs_model": int(min_runs_model),
        "variability_metric": config.variability_metric,
        "variability_threshold": variability_threshold,
        "n_segments": choose_n_segments(len(anchors), config),
    }
    return anchors, metadata



def _fit_and_predict(
    anchors: pd.DataFrame,
    x_test: np.ndarray,
    n_segments: int,
    refine_fit_fn: FitFunction,
    predict_fn: PredictFunction,
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """Fit one spline and return test and training predictions."""

    x_train = anchors["library_coordinate"].to_numpy(dtype=np.float64)
    y_train = anchors["observed_coordinate"].to_numpy(dtype=np.float64)

    coeff, points = refine_fit_fn(x_train, y_train, int(n_segments))

    x_test_array = np.asarray(x_test, dtype=np.float64).reshape(-1)
    if x_test_array.size:
        test_prediction = np.asarray(
            predict_fn(
                coeff,
                points,
                x_test_array,
            ),
            dtype=np.float64,
        ).reshape(-1)
    else:
        test_prediction = np.empty(0, dtype=np.float64)

    train_prediction = np.asarray(
        predict_fn(coeff, points, x_train),
        dtype=np.float64,
    ).reshape(-1)

    if test_prediction.size != x_test_array.size:
        raise RuntimeError(
            "calc_spline_vectorized returned an unexpected number of test predictions"
        )
    if train_prediction.size != len(anchors):
        raise RuntimeError(
            "calc_spline_vectorized returned an unexpected number of training predictions"
        )

    return coeff, points, test_prediction, train_prediction


def _quantile_metrics(errors: np.ndarray, prefix: str = "") -> Dict[str, float]:
    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}median": float("nan"),
            f"{prefix}p80": float("nan"),
            f"{prefix}p90": float("nan"),
            f"{prefix}p95": float("nan"),
            f"{prefix}p99": float("nan"),
        }

    return {
        f"{prefix}median": float(np.quantile(finite, 0.50)),
        f"{prefix}p80": float(np.quantile(finite, 0.80)),
        f"{prefix}p90": float(np.quantile(finite, 0.90)),
        f"{prefix}p95": float(np.quantile(finite, 0.95)),
        f"{prefix}p99": float(np.quantile(finite, 0.99)),
    }


def _make_grouped_test_folds(
    precursor_ids: Sequence[Any],
    group_labels: Sequence[Any],
    n_splits: int,
    random_state: int,
) -> List[pd.Index]:
    """Create deterministic, approximately balanced grouped test folds.

    All precursor IDs with the same group label are held out together. This avoids
    leakage between charge states or related precursors when a peptidoform-level
    grouping column is supplied.
    """

    ids = np.asarray(list(precursor_ids), dtype=object)
    groups = np.asarray(list(group_labels), dtype=object)

    if ids.size != groups.size:
        raise ValueError("precursor_ids and group_labels must have equal length")
    if ids.size < 2:
        raise ValueError("At least two target coordinates are required for cross-validation")

    normalized_groups: List[Any] = []
    for precursor_id, group in zip(ids, groups):
        if pd.isna(group):
            normalized_groups.append(("__precursor__", str(precursor_id)))
        else:
            normalized_groups.append(("__group__", str(group)))

    group_to_positions: Dict[Any, List[int]] = {}
    for position, group in enumerate(normalized_groups):
        group_to_positions.setdefault(group, []).append(position)

    unique_groups = list(group_to_positions.keys())
    actual_splits = min(int(n_splits), len(unique_groups))
    if actual_splits < 2:
        raise ValueError("At least two unique groups are required for grouped cross-validation")

    rng = np.random.default_rng(random_state)
    random_order = rng.permutation(len(unique_groups))
    randomized_groups = [unique_groups[i] for i in random_order]

    # Largest groups are assigned first; randomized order breaks size ties.
    randomized_groups.sort(
        key=lambda group: len(group_to_positions[group]),
        reverse=True,
    )

    fold_groups: List[List[Any]] = [[] for _ in range(actual_splits)]
    fold_sizes = np.zeros(actual_splits, dtype=int)

    for group in randomized_groups:
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(group)
        fold_sizes[fold_index] += len(group_to_positions[group])

    folds: List[pd.Index] = []
    for groups_in_fold in fold_groups:
        positions: List[int] = []
        for group in groups_in_fold:
            positions.extend(group_to_positions[group])
        positions.sort()
        folds.append(pd.Index(ids[positions]))

    return folds


def _resolve_group_labels(
    target_ids: pd.Index,
    lib_df_uniq_prec: pd.DataFrame,
    group_col: Optional[str],
) -> pd.Series:
    if group_col is None:
        return pd.Series(target_ids.astype(str), index=target_ids)

    if group_col not in lib_df_uniq_prec.columns:
        raise KeyError(
            f"Requested group_col '{group_col}' is not present in lib_df_uniq_prec"
        )

    group_map = lib_df_uniq_prec.set_index("Precursor.Id")[group_col]
    groups = group_map.reindex(target_ids)

    # Missing group labels are treated as precursor-specific groups.
    fallback = pd.Series(
        [f"__precursor__::{precursor_id}" for precursor_id in target_ids],
        index=target_ids,
        dtype=object,
    )
    return groups.astype(object).where(groups.notna(), fallback)


def compute_observed_deviation_bounds(
    observed_mat: pd.DataFrame,
    config: ModelConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Estimate robust cross-run deviation bounds from observed RTs only."""

    observed_median = observed_mat.median(axis=1, skipna=True)
    deviations = observed_mat.sub(observed_median, axis=0)

    global_values = deviations.to_numpy(dtype=float).reshape(-1)
    global_values = global_values[np.isfinite(global_values)]

    if global_values.size == 0:
        global_lower = float("-inf")
        global_upper = float("inf")
    else:
        global_lower = float(
            np.quantile(global_values, config.adjustment_lower_quantile)
        )
        global_upper = float(
            np.quantile(global_values, config.adjustment_upper_quantile)
        )

    records: List[Dict[str, Any]] = []
    for run_name in observed_mat.columns:
        run_values = deviations[run_name].to_numpy(dtype=float)
        run_values = run_values[np.isfinite(run_values)]

        use_run_bounds = (
            config.adjustment_by_run
            and run_values.size >= config.min_points_for_run_bounds
        )

        if use_run_bounds:
            lower = float(
                np.quantile(run_values, config.adjustment_lower_quantile)
            )
            upper = float(
                np.quantile(run_values, config.adjustment_upper_quantile)
            )
            source = "run"
        else:
            lower = global_lower
            upper = global_upper
            source = "global"

        records.append(
            {
                "run_name": run_name,
                "lower_deviation": lower,
                "upper_deviation": upper,
                "bound_source": source,
                "n_observed_deviations": int(run_values.size),
            }
        )

    bounds = pd.DataFrame.from_records(records).set_index("run_name")
    return observed_median, bounds


def adjust_prediction_vector(
    precursor_ids: pd.Index,
    raw_predictions: np.ndarray,
    run_name: str,
    observed_median: pd.Series,
    bounds: pd.DataFrame,
    mode: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Adjust one vector of predictions without modifying observed coordinates."""

    raw = np.asarray(raw_predictions, dtype=float).reshape(-1)
    if raw.size != len(precursor_ids):
        raise ValueError("raw_predictions length does not match precursor_ids")
    if run_name not in bounds.index:
        raise KeyError(f"No adjustment bounds are available for run '{run_name}'")

    reference = observed_median.reindex(precursor_ids).to_numpy(dtype=float)
    lower = float(bounds.loc[run_name, "lower_deviation"])
    upper = float(bounds.loc[run_name, "upper_deviation"])

    adjusted = raw.copy()
    finite_raw = np.isfinite(raw)
    has_reference = np.isfinite(reference)
    deviation = raw - reference
    outside = (
        finite_raw
        & has_reference
        & ((deviation < lower) | (deviation > upper))
    )

    if mode == "clip":
        clipped_deviation = np.clip(deviation, lower, upper)
        adjusted[outside] = reference[outside] + clipped_deviation[outside]
    elif mode == "median":
        adjusted[outside] = reference[outside]
    elif mode == "none":
        pass
    else:
        raise ValueError("mode must be 'clip', 'median', or 'none'")

    status = np.full(raw.size, "within_bounds", dtype=object)
    status[~finite_raw] = "nonfinite_prediction"
    status[finite_raw & ~has_reference] = "no_cross_run_reference"
    if mode == "none":
        status[finite_raw & has_reference] = "adjustment_disabled"
    else:
        status[outside] = f"adjusted_{mode}"

    info = pd.DataFrame(
        {
            "precursor_id": precursor_ids,
            "run_name": run_name,
            "raw_prediction": raw,
            "observed_cross_run_median": reference,
            "raw_deviation_from_median": deviation,
            "lower_deviation_bound": lower,
            "upper_deviation_bound": upper,
            "final_prediction": adjusted,
            "was_adjusted": outside,
            "adjustment_status": status,
        }
    )
    return adjusted, info


def adjust_prediction_matrix(
    raw_coordinate_matrix: pd.DataFrame,
    observed_coordinate_matrix: pd.DataFrame,
    prediction_mask: pd.DataFrame,
    config: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adjust only predicted cells; observed RT values are guaranteed unchanged."""

    if not raw_coordinate_matrix.index.equals(observed_coordinate_matrix.index):
        raise ValueError("Raw and observed matrices must have identical row indexes")
    if not raw_coordinate_matrix.columns.equals(observed_coordinate_matrix.columns):
        raise ValueError("Raw and observed matrices must have identical columns")
    if not prediction_mask.index.equals(raw_coordinate_matrix.index):
        raise ValueError("prediction_mask has a different row index")
    if not prediction_mask.columns.equals(raw_coordinate_matrix.columns):
        raise ValueError("prediction_mask has different columns")

    observed_median, bounds = compute_observed_deviation_bounds(
        observed_mat=observed_coordinate_matrix,
        config=config,
    )

    adjusted_matrix = raw_coordinate_matrix.copy()
    adjustment_records: List[pd.DataFrame] = []

    for run_name in raw_coordinate_matrix.columns:
        ids = raw_coordinate_matrix.index[
            prediction_mask[run_name].fillna(False).to_numpy(dtype=bool)
        ]
        if len(ids) == 0:
            continue

        raw_values = raw_coordinate_matrix.loc[ids, run_name].to_numpy(dtype=float)
        adjusted_values, info = adjust_prediction_vector(
            precursor_ids=ids,
            raw_predictions=raw_values,
            run_name=run_name,
            observed_median=observed_median,
            bounds=bounds,
            mode=config.adjustment_mode,
        )
        adjusted_matrix.loc[ids, run_name] = adjusted_values
        adjustment_records.append(info)

    # Hard safety check: observed coordinates must remain exactly as supplied.
    observed_mask = observed_coordinate_matrix.notna().to_numpy(dtype=bool)
    observed_original = observed_coordinate_matrix.to_numpy(dtype=float)[observed_mask]
    observed_after = adjusted_matrix.to_numpy(dtype=float)[observed_mask]
    if not np.allclose(observed_original, observed_after, equal_nan=True):
        raise RuntimeError("Observed coordinates were modified during prediction adjustment")

    if adjustment_records:
        adjustment_info = pd.concat(adjustment_records, ignore_index=True)
    else:
        adjustment_info = pd.DataFrame(
            columns=[
                "precursor_id",
                "run_name",
                "raw_prediction",
                "observed_cross_run_median",
                "raw_deviation_from_median",
                "lower_deviation_bound",
                "upper_deviation_bound",
                "final_prediction",
                "was_adjusted",
                "adjustment_status",
            ]
        )

    return adjusted_matrix, adjustment_info, bounds.reset_index()


def cross_validate_coordinate_prediction(
    coordinate_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    coordinate_column: str,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Leakage-safe cross-validation for RT or IM prediction."""

    config = config or ModelConfig()
    coordinate_series = _validate_inputs(
        coordinate_mat, lib_df_uniq_prec, coordinate_column
    )
    min_runs = resolve_min_runs_model(
        n_runs=coordinate_mat.shape[1],
        config=config,
        min_runs_model=min_runs_model,
    )
    
    logging.info(
        "Cross-validating %s prediction for each run in %d folds",
        config.coordinate_label,
        config.n_splits
    )
    records: List[pd.DataFrame] = []
    for run_number, run_name in enumerate(coordinate_mat.columns):
        target_ids = coordinate_mat.index[
            coordinate_mat[run_name].notna() & coordinate_series.reindex(coordinate_mat.index).notna()
        ]

        if len(target_ids) < 2:
            logging.warning(
                "Skipping cross-validation for run %s because only %d observed targets are available",
                run_name,
                len(target_ids),
            )
            continue

        group_labels = _resolve_group_labels(
            target_ids=target_ids,
            lib_df_uniq_prec=lib_df_uniq_prec,
            group_col=group_col,
        )
        test_folds = _make_grouped_test_folds(
            precursor_ids=target_ids,
            group_labels=group_labels.reindex(target_ids).to_numpy(dtype=object),
            n_splits=config.n_splits,
            random_state=config.random_state + run_number,
        )

        for fold_number, test_ids in enumerate(test_folds):
            masked_mat = coordinate_mat.copy()
            observed_truth = masked_mat.loc[test_ids, run_name].astype(float).copy()
            masked_mat.loc[test_ids, run_name] = np.nan

            base_df = pd.DataFrame(
                {
                    "precursor_id": test_ids,
                    "run_name": run_name,
                    "fold": fold_number,
                    "observed_coordinate": observed_truth.reindex(test_ids).to_numpy(dtype=float),
                    "library_coordinate": coordinate_series.reindex(test_ids).to_numpy(dtype=float),
                    "group_label": group_labels.reindex(test_ids).to_numpy(dtype=object),
                    "n_observed_other_runs": masked_mat.loc[test_ids]
                    .notna()
                    .sum(axis=1)
                    .to_numpy(dtype=int),
                }
            )

            try:
                anchors, anchor_meta = prepare_run_anchors(
                    mat=masked_mat,
                    run_name=run_name,
                    library_coordinate=coordinate_series,
                    min_runs_model=min_runs,
                    config=config,
                )

                x_test = base_df["library_coordinate"].to_numpy(dtype=np.float64)
                _, _, raw_prediction, _ = _fit_and_predict(
                    anchors=anchors,
                    x_test=x_test,
                    n_segments=anchor_meta["n_segments"],
                    refine_fit_fn=fit_fn,
                    predict_fn=predict_fn,
                )

                observed_median, bounds = compute_observed_deviation_bounds(
                    observed_mat=masked_mat,
                    config=config,
                )

                final_prediction, adjustment_info = adjust_prediction_vector(
                    precursor_ids=test_ids,
                    raw_predictions=raw_prediction,
                    run_name=run_name,
                    observed_median=observed_median,
                    bounds=bounds,
                    mode=config.adjustment_mode,
                )

                train_min = float(anchors["library_coordinate"].min())
                train_max = float(anchors["library_coordinate"].max())
                is_interpolation = (x_test >= train_min) & (x_test <= train_max)

                base_df[f"predicted_{config.coordinate_label.lower()}_raw"] = raw_prediction
                base_df[f"predicted_{config.coordinate_label.lower()}_final"] = final_prediction
                base_df["raw_signed_error"] = (
                    base_df[f"predicted_{config.coordinate_label.lower()}_raw"]
                    - base_df["observed_coordinate"]
                )
                base_df["raw_absolute_error"] = base_df["raw_signed_error"].abs()
                base_df["final_signed_error"] = (
                    base_df[f"predicted_{config.coordinate_label.lower()}_final"]
                    - base_df["observed_coordinate"]
                )
                base_df["final_absolute_error"] = base_df["final_signed_error"].abs()
                base_df["is_interpolation"] = is_interpolation
                base_df["n_training_anchors"] = anchor_meta["n_run_anchors"]
                base_df["n_segments"] = anchor_meta["n_segments"]
                base_df["min_runs_model"] = min_runs
                base_df["variability_metric"] = config.variability_metric
                base_df["variability_threshold"] = anchor_meta["variability_threshold"]
                base_df["was_adjusted"] = adjustment_info["was_adjusted"].to_numpy(dtype=bool)
                base_df["adjustment_status"] = adjustment_info["adjustment_status"].to_numpy(dtype=object)
                base_df["cross_run_reference_median"] = adjustment_info[
                    "observed_cross_run_median"
                ].to_numpy(dtype=float)
                base_df["lower_deviation_bound"] = adjustment_info[
                    "lower_deviation_bound"
                ].to_numpy(dtype=float)
                base_df["upper_deviation_bound"] = adjustment_info[
                    "upper_deviation_bound"
                ].to_numpy(dtype=float)
                base_df["fit_status"] = "ok"
                base_df["fit_error"] = ""
            except Exception as exc:  # pragma: no cover
                logging.exception(
                    "%s cross-validation failed for run %s, fold %d",
                    config.coordinate_label,
                    run_name,
                    fold_number,
                )
                base_df[f"predicted_{config.coordinate_label.lower()}_raw"] = np.nan
                base_df[f"predicted_{config.coordinate_label.lower()}_final"] = np.nan
                base_df["raw_signed_error"] = np.nan
                base_df["raw_absolute_error"] = np.nan
                base_df["final_signed_error"] = np.nan
                base_df["final_absolute_error"] = np.nan
                base_df["is_interpolation"] = False
                base_df["n_training_anchors"] = 0
                base_df["n_segments"] = 0
                base_df["min_runs_model"] = min_runs
                base_df["variability_metric"] = config.variability_metric
                base_df["variability_threshold"] = np.nan
                base_df["was_adjusted"] = False
                base_df["adjustment_status"] = "fit_failed"
                base_df["cross_run_reference_median"] = np.nan
                base_df["lower_deviation_bound"] = np.nan
                base_df["upper_deviation_bound"] = np.nan
                base_df["fit_status"] = "failed"
                base_df["fit_error"] = f"{type(exc).__name__}: {exc}"

            records.append(base_df)

    if not records:
        raise RuntimeError("Cross-validation produced no evaluation records")

    result = pd.concat(records, ignore_index=True)
    result.sort_values(["run_name", "fold", "precursor_id"], inplace=True, kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    return result


def summarize_coordinate_cross_validation(
    cv_predictions: pd.DataFrame,
    config: Optional[ModelConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create per-run and pooled summaries for RT or IM validation."""

    config = config or ModelConfig()

    required = {
        "run_name",
        "observed_coordinate",
        f"predicted_{config.coordinate_label.lower()}_raw",
        f"predicted_{config.coordinate_label.lower()}_final",
        "fit_status",
        "was_adjusted",
        "is_interpolation",
        "n_training_anchors",
    }
    missing = required.difference(cv_predictions.columns)
    if missing:
        raise KeyError(f"cv_predictions is missing columns: {sorted(missing)}")

    run_records: List[Dict[str, Any]] = []

    for run_name, run_df in cv_predictions.groupby("run_name", sort=False):
        valid = run_df[
            (run_df["fit_status"] == "ok")
            & run_df["observed_coordinate"].notna()
            & run_df[f"predicted_{config.coordinate_label.lower()}_final"].notna()
        ].copy()

        record: Dict[str, Any] = {
            "run_name": run_name,
            "coordinate_label": config.coordinate_label,
            "n_targets": int(len(run_df)),
            "n_evaluated": int(len(valid)),
            "n_failed": int(len(run_df) - len(valid)),
            "failure_fraction": float(1.0 - len(valid) / len(run_df))
            if len(run_df)
            else float("nan"),
            "fraction_adjusted": float(valid["was_adjusted"].mean())
            if len(valid)
            else float("nan"),
            "fraction_extrapolated": float((~valid["is_interpolation"]).mean())
            if len(valid)
            else float("nan"),
            "median_training_anchors": float(valid["n_training_anchors"].median())
            if len(valid)
            else float("nan"),
        }

        for label, prediction_column in (
            ("raw", f"predicted_{config.coordinate_label.lower()}_raw"),
            ("final", f"predicted_{config.coordinate_label.lower()}_final"),
        ):
            signed_errors = (
                valid[prediction_column].to_numpy(dtype=float)
                - valid["observed_coordinate"].to_numpy(dtype=float)
            )
            abs_errors = np.abs(signed_errors)

            metrics = _quantile_metrics(
                abs_errors,
                prefix=f"{label}_absolute_error_",
            )
            record.update(metrics)
            record[f"{label}_median_signed_error"] = (
                float(np.median(signed_errors)) if signed_errors.size else float("nan")
            )
            record[f"{label}_mean_absolute_error"] = (
                float(np.mean(abs_errors)) if abs_errors.size else float("nan")
            )
            record[f"{label}_rmse"] = (
                float(np.sqrt(np.mean(signed_errors**2))) if signed_errors.size else float("nan")
            )

        run_records.append(record)
        logging.info(
            "Absolute prediction error of %s model for run %s at 80%%, 90%%, 95%%, 99%% quantiles: %.4f, %.4f, %.4f, %.4f.",
            config.coordinate_label, run_name[-20:],
            record["final_absolute_error_p80"],
            record["final_absolute_error_p90"],
            record["final_absolute_error_p95"],
            record["final_absolute_error_p99"]
        )
    run_metrics = pd.DataFrame.from_records(run_records)

    valid_all = cv_predictions[
        (cv_predictions["fit_status"] == "ok")
        & cv_predictions["observed_coordinate"].notna()
        & cv_predictions[f"predicted_{config.coordinate_label.lower()}_final"].notna()
    ].copy()

    pooled_signed = (
        valid_all[f"predicted_{config.coordinate_label.lower()}_final"].to_numpy(dtype=float)
        - valid_all["observed_coordinate"].to_numpy(dtype=float)
    )
    pooled_abs = np.abs(pooled_signed)

    pooled: Dict[str, Any] = {
        "n_runs": int(run_metrics.shape[0]),
        "n_targets": int(len(cv_predictions)),
        "n_evaluated": int(len(valid_all)),
        "n_failed": int(len(cv_predictions) - len(valid_all)),
        "min_runs_model": int(cv_predictions["min_runs_model"].iloc[0])
        if "min_runs_model" in cv_predictions.columns and len(cv_predictions)
        else None,
        "configuration": asdict(config),
    }
    pooled.update(_quantile_metrics(pooled_abs, prefix="pooled_final_absolute_error_"))
    pooled["pooled_final_median_signed_error"] = (
        float(np.median(pooled_signed)) if pooled_signed.size else float("nan")
    )

    if not run_metrics.empty:
        p80_column = "final_absolute_error_p80"
        pooled["median_run_final_p80_error"] = float(
            run_metrics[p80_column].median()
        )
        pooled["min_run_final_p80_error"] = float(
            run_metrics[p80_column].min()
        )
        pooled["max_run_final_p80_error"] = float(
            run_metrics[p80_column].max()
        )

    return run_metrics, pooled


def fit_coordinate_production_models(
    coordinate_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    coordinate_column: str,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
) -> CoordinateProductionResult:
    """Fit final run-specific models and predict genuinely missing coordinates."""

    config = config or ModelConfig()
    coordinate_series = _validate_inputs(
        coordinate_mat, lib_df_uniq_prec, coordinate_column
    )
    min_runs = resolve_min_runs_model(
        n_runs=coordinate_mat.shape[1],
        config=config,
        min_runs_model=min_runs_model,
    )

    raw_matrix = coordinate_mat.copy().astype(float)
    prediction_mask = pd.DataFrame(False, index=coordinate_mat.index, columns=coordinate_mat.columns)
    models: Dict[str, Dict[str, Any]] = {}
    fit_records: List[Dict[str, Any]] = []
    prediction_records: List[pd.DataFrame] = []

    label = config.coordinate_label.lower()
    
    logging.info("Fitting final production %s model for each run\n", config.coordinate_label)
    for run_name in coordinate_mat.columns:
        anchors, anchor_meta = prepare_run_anchors(
            mat=coordinate_mat,
            run_name=run_name,
            library_coordinate=coordinate_series,
            min_runs_model=min_runs,
            config=config,
        )

        missing_ids = coordinate_mat.index[
            coordinate_mat[run_name].isna() & coordinate_series.reindex(coordinate_mat.index).notna()
        ]
        x_missing = coordinate_series.reindex(missing_ids).to_numpy(dtype=np.float64)

        coeff, points, predicted_missing, fitted_training = _fit_and_predict(
            anchors=anchors,
            x_test=x_missing,
            n_segments=anchor_meta["n_segments"],
            refine_fit_fn=fit_fn,
            predict_fn=predict_fn,
        )

        train_min = float(anchors["library_coordinate"].min())
        train_max = float(anchors["library_coordinate"].max())
        is_interpolation = (x_missing >= train_min) & (x_missing <= train_max)

        accepted = np.isfinite(predicted_missing)
        if not config.allow_extrapolation:
            accepted &= is_interpolation

        accepted_ids = missing_ids[accepted]
        accepted_predictions = predicted_missing[accepted]
        raw_matrix.loc[accepted_ids, run_name] = accepted_predictions
        prediction_mask.loc[accepted_ids, run_name] = True

        train_observed = anchors["observed_coordinate"].to_numpy(dtype=float)
        train_signed_residual = fitted_training - train_observed
        train_absolute_residual = np.abs(train_signed_residual)

        fit_record: Dict[str, Any] = {
            **anchor_meta,
            "training_median_signed_residual": float(np.median(train_signed_residual)),
            "training_mean_absolute_residual": float(np.mean(train_absolute_residual)),
            "training_rmse": float(np.sqrt(np.mean(train_signed_residual**2))),
            "n_missing_with_library_coordinate": int(len(missing_ids)),
            "n_predictions_accepted": int(accepted.sum()),
            "n_extrapolated_candidates": int((~is_interpolation).sum()),
        }
        fit_record.update(
            _quantile_metrics(
                train_absolute_residual,
                prefix="training_absolute_residual_",
            )
        )
        fit_records.append(fit_record)

        model_record = {
            "coeff": coeff,
            "points": points,
            "training_library_coordinate_min": train_min,
            "training_library_coordinate_max": train_max,
            **anchor_meta,
        }
        models[str(run_name)] = model_record

        if len(missing_ids):
            prediction_status = np.full(len(missing_ids), "accepted", dtype=object)
            prediction_status[~np.isfinite(predicted_missing)] = "nonfinite_prediction"
            if not config.allow_extrapolation:
                prediction_status[np.isfinite(predicted_missing) & ~is_interpolation] = (
                    "rejected_extrapolation"
                )

            prediction_records.append(
                pd.DataFrame(
                    {
                        "precursor_id": missing_ids,
                        "run_name": run_name,
                        "library_coordinate": x_missing,
                        f"predicted_{label}_raw": predicted_missing,
                        "is_interpolation": is_interpolation,
                        "prediction_accepted": accepted,
                        "prediction_status": prediction_status,
                        "n_training_anchors": anchor_meta["n_run_anchors"],
                        "n_segments": anchor_meta["n_segments"],
                    }
                )
            )

    final_matrix, adjustment_info, adjustment_bounds = adjust_prediction_matrix(
        raw_coordinate_matrix=raw_matrix,
        observed_coordinate_matrix=coordinate_mat.astype(float),
        prediction_mask=prediction_mask,
        config=config,
    )

    source_matrix = pd.DataFrame(
        "missing",
        index=coordinate_mat.index,
        columns=coordinate_mat.columns,
        dtype=object,
    )
    source_matrix[coordinate_mat.notna()] = "observed"
    source_matrix[prediction_mask] = "predicted"

    if prediction_records:
        prediction_metadata = pd.concat(prediction_records, ignore_index=True)
        prediction_metadata = prediction_metadata.merge(
            adjustment_info[
                [
                    "precursor_id",
                    "run_name",
                    "observed_cross_run_median",
                    "raw_deviation_from_median",
                    "lower_deviation_bound",
                    "upper_deviation_bound",
                    "final_prediction",
                    "was_adjusted",
                    "adjustment_status",
                ]
            ],
            on=["precursor_id", "run_name"],
            how="left",
            validate="one_to_one",
        )
        prediction_metadata.rename(
            columns={"final_prediction": f"predicted_{label}_final"},
            inplace=True,
        )
    else:
        prediction_metadata = pd.DataFrame()

    fit_metrics = pd.DataFrame.from_records(fit_records)

    models["__metadata__"] = {
        "min_runs_model": min_runs,
        "configuration": asdict(config),
        "adjustment_bounds": adjustment_bounds.to_dict(orient="records"),
        "coordinate_column": coordinate_column,
    }

    return CoordinateProductionResult(
        raw_coordinate_matrix=raw_matrix,
        final_coordinate_matrix=final_matrix,
        coordinate_source_matrix=source_matrix,
        prediction_metadata=prediction_metadata,
        fit_metrics=fit_metrics,
        models=models,
        min_runs_model=min_runs,
    )

def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def build_coordinate_error_histogram(
    cv_predictions: pd.DataFrame,
    coordinate_label: str = "RT",
    n_bins: int = 100,
    upper_quantile: float = 0.995,
) -> pd.DataFrame:
    """Summarize pooled final OOF signed errors without saving all predictions.

    Signed error is defined as:

        predicted_coordinate - observed_coordinate

    Histogram values remain in native coordinate units. The histogram is
    symmetric around zero so prediction bias and error spread can be assessed
    directly.

    Extreme tails are excluded symmetrically according to `upper_quantile`
    of the absolute error distribution.
    """
    pred_col = f"predicted_{coordinate_label.lower()}_final"

    required = {
        "fit_status",
        "observed_coordinate",
        pred_col,
    }
    missing = required.difference(cv_predictions.columns)
    if missing:
        raise KeyError(
            f"cv_predictions is missing columns: {sorted(missing)}"
        )

    valid = cv_predictions[
        (cv_predictions["fit_status"] == "ok")
        & cv_predictions["observed_coordinate"].notna()
        & cv_predictions[pred_col].notna()
    ]

    errors = (
        valid[pred_col].to_numpy(dtype=float)
        - valid["observed_coordinate"].to_numpy(dtype=float)
    )

    errors = errors[np.isfinite(errors)]

    if errors.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_left",
                "bin_right",
                "bin_center",
                "count",
                "fraction",
            ]
        )

    # Symmetric plotting range based on absolute errors.
    abs_errors = np.abs(errors)

    limit = float(
        np.quantile(abs_errors, upper_quantile)
    )

    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(abs_errors))

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    edges = np.linspace(
        -limit,
        limit,
        int(n_bins) + 1,
    )

    counts, edges = np.histogram(
        errors,
        bins=edges,
    )

    total = int(counts.sum())

    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": (
            edges[:-1] + edges[1:]
        ) / 2.0,
        "count": counts.astype(int),
        "fraction": (
            counts / total
            if total
            else np.nan
        ),
    })

def select_representative_cv_run(
    run_metrics: pd.DataFrame,
    metric: str = "final_absolute_error_p95",
) -> str:
    """Select the run whose CV error is closest to the across-run median."""
    if run_metrics is None or run_metrics.empty:
        raise ValueError("run_metrics is empty")
    if metric not in run_metrics.columns:
        raise KeyError(f"run_metrics is missing '{metric}'")
    values = pd.to_numeric(run_metrics[metric], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        raise ValueError(f"No finite values are available for '{metric}'")
    target = float(finite.median())
    idx = (values - target).abs().idxmin()
    return str(run_metrics.loc[idx, "run_name"])


def save_coordinate_outputs(
    coordinate: str,
    out_dir: Union[str, Path],
    cv_predictions: Optional[pd.DataFrame] = None,
    run_metrics: Optional[pd.DataFrame] = None,
    pooled_summary: Optional[Dict[str, Any]] = None,
    error_histogram: Optional[pd.DataFrame] = None,
    production_result: Optional[CoordinateProductionResult] = None,
) -> None:
    """Save validation and production outputs in reproducible tabular form."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if cv_predictions is not None and run_metrics is not None and not run_metrics.empty:
        representative_run = select_representative_cv_run(run_metrics)
        cv_predictions_eg = cv_predictions[
            cv_predictions["run_name"].astype(str) == representative_run
        ].copy()
        cv_predictions_eg.to_csv(
            output / f"{coordinate}_cross_validated_predictions_representative_run.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame([{
            "representative_run": representative_run,
            "selection_metric": "final_absolute_error_p95",
            "selection_rule": "closest to median run p95 absolute error",
        }]).to_csv(
            output / f"{coordinate}_cross_validation_representative_run.tsv",
            sep="\t",
            index=False,
        )
    if run_metrics is not None:
        run_metrics.to_csv(
            output / f"{coordinate}_cross_validation_run_metrics.tsv",
            sep="\t",
            index=False,
        )
    if pooled_summary is not None:
        with open(output / f"{coordinate}_cross_validation_summary.json", "w") as handle:
            json.dump(
                pooled_summary,
                handle,
                indent=2,
                default=_json_default,
                allow_nan=True,
            )

    if error_histogram is not None:
        error_histogram.to_csv(
            output / f"{coordinate}_cross_validation_error_histogram.tsv",
            sep="\t",
            index=False,
        )

    if production_result is not None:
        # production_result.raw_coordinate_matrix.to_csv(
        #     output / "coordinate_matrix_with_raw_predictions.tsv",
        #     sep="\t",
        # )
        production_result.final_coordinate_matrix.to_csv(
            output / f"{coordinate}_mat_after_prediction.tsv",
            sep="\t",
        )
        # production_result.coordinate_source_matrix.to_csv(
        #     output / "coordinate_source.tsv",
        #     sep="\t",
        # )
        # production_result.prediction_metadata.to_csv(
        #     output / "coordinate_prediction_metadata.tsv",
        #     sep="\t",
        #     index=False,
        # )
        production_result.fit_metrics.to_csv(
            output / f"{coordinate}_production_fit_metrics.tsv",
            sep="\t",
            index=False,
        )
        with open(output / f"{coordinate}_production_models.json", "w") as handle:
            json.dump(
                production_result.models,
                handle,
                indent=2,
                default=_json_default,
                allow_nan=True,
            )


def plot_coordinate_validation_diagnostics(
    cv_predictions: pd.DataFrame,
    run_metrics: pd.DataFrame,
    out_dir: Union[str, Path],
    coordinate_label: str = "RT",
) -> Dict[str, Path]:
    """Create manuscript-friendly validation plots as separate PDF files."""

    import matplotlib.pyplot as plt
    #from matplotlib.backends.backend_pdf import PdfPages

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    pred_col = [c for c in cv_predictions.columns if c.startswith("predicted_") and c.endswith("_final")]
    if not pred_col:
        raise ValueError("No final prediction column found")
    pred_col = pred_col[0]

    valid = cv_predictions[
        (cv_predictions["fit_status"] == "ok")
        & cv_predictions["observed_coordinate"].notna()
        & cv_predictions[pred_col].notna()
    ].copy()

    paths: Dict[str, Path] = {}
    
    # Representative run: objectively selected by median run-level p95 error.
    representative_run = select_representative_cv_run(run_metrics)
    run_data = valid[valid["run_name"].astype(str) == representative_run]
    scatter_path = output / f"{coordinate_label.lower()}_cross_validation_representative_run_scatter.pdf"
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(
        run_data["observed_coordinate"],
        run_data[pred_col],
        s=8,
        alpha=0.35,
    )
    low = float(min(run_data["observed_coordinate"].min(), run_data[pred_col].min()))
    high = float(max(run_data["observed_coordinate"].max(), run_data[pred_col].max()))
    ax.plot([low, high], [low, high], linewidth=1)
    ax.set_xlabel(f"Observed {coordinate_label}")
    ax.set_ylabel(f"Out-of-fold predicted {coordinate_label}")
    run_title = representative_run if len(representative_run) < 20 else representative_run[-15:]
    ax.set_title(f"Representative run: {run_title}")
    fig.tight_layout()
    fig.savefig(scatter_path)
    plt.close(fig)
    paths["representative_run_scatter"] = scatter_path

    p95_path = output / f"{coordinate_label.lower()}_cross_validation_run_p95_errors.pdf"
    ordered = run_metrics.sort_values("final_absolute_error_p95", kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.28 * len(ordered)), 4.2))
    ax.scatter(np.arange(len(ordered)), ordered["final_absolute_error_p95"], s=22)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered["run_name"], rotation=90)
    ax.set_ylabel(f"Cross-validated p95 absolute {coordinate_label} error")
    ax.set_xlabel("Run")
    fig.tight_layout()
    fig.savefig(p95_path)
    plt.close(fig)
    paths["run_p95_errors"] = p95_path

    return paths


def run_coordinate_validation_and_production(
    coordinate_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    coordinate_column: str,
    out_dir: Union[str, Path],
    config: Optional[ModelConfig] = None,
    run_validation: bool = False,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], CoordinateProductionResult]:
    """Convenience wrapper for validation plus production fitting."""

    config = config or ModelConfig()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    cv_predictions = None
    run_metrics = None
    pooled_summary = None
    error_histogram = None
    if run_validation:
        cv_predictions = cross_validate_coordinate_prediction(
            coordinate_mat=coordinate_mat,
            lib_df_uniq_prec=lib_df_uniq_prec,
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            coordinate_column=coordinate_column,
            config=config,
            min_runs_model=min_runs_model,
            group_col=group_col,
        )
        run_metrics, pooled_summary = summarize_coordinate_cross_validation(
            cv_predictions=cv_predictions,
            config=config,
        )
        error_histogram = build_coordinate_error_histogram(
            cv_predictions=cv_predictions,
            coordinate_label=config.coordinate_label,
        )

    production_result = fit_coordinate_production_models(
        coordinate_mat=coordinate_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        coordinate_column=coordinate_column,
        config=config,
        min_runs_model=min_runs_model,
    )

    save_coordinate_outputs(
        coordinate_column.lower(),
        out_dir=output,
        cv_predictions=cv_predictions,
        run_metrics=run_metrics,
        pooled_summary=pooled_summary,
        error_histogram=error_histogram,
        production_result=production_result,
    )

    if run_validation and save_plots:
        plot_coordinate_validation_diagnostics(
            cv_predictions=cv_predictions,
            run_metrics=run_metrics,
            out_dir=output,
            coordinate_label=config.coordinate_label,
        )

    return cv_predictions, run_metrics, pooled_summary, production_result


# Convenience wrappers for RT and IM ----------------------------------------

def cross_validate_rt_prediction(
    rt_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    refine_spline_fn: FitFunction,
    calc_spline_vectorized_fn: PredictFunction,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    cfg = config or ModelConfig(coordinate_label="RT")
    return cross_validate_coordinate_prediction(
        coordinate_mat=rt_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=refine_spline_fn,
        predict_fn=calc_spline_vectorized_fn,
        coordinate_column="RT",
        config=cfg,
        min_runs_model=min_runs_model,
        group_col=group_col,
    )


def fit_production_rt_models(
    rt_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    refine_spline_fn: FitFunction,
    calc_spline_vectorized_fn: PredictFunction,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
) -> CoordinateProductionResult:
    cfg = config or ModelConfig(coordinate_label="RT")
    return fit_coordinate_production_models(
        coordinate_mat=rt_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=refine_spline_fn,
        predict_fn=calc_spline_vectorized_fn,
        coordinate_column="RT",
        config=cfg,
        min_runs_model=min_runs_model,
    )


def run_rt_prediction(
    rt_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    refine_spline_fn: FitFunction,
    calc_spline_vectorized_fn: PredictFunction,
    out_dir: Union[str, Path],
    config: Optional[ModelConfig] = None,
    run_validation: bool = False,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], CoordinateProductionResult]:
    cfg = config or ModelConfig(coordinate_label="RT")
    return run_coordinate_validation_and_production(
        coordinate_mat=rt_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=refine_spline_fn,
        predict_fn=calc_spline_vectorized_fn,
        coordinate_column="RT",
        out_dir=out_dir,
        run_validation=run_validation,
        config=cfg,
        min_runs_model=min_runs_model,
        group_col=group_col,
        save_plots=save_plots,
    )


def cross_validate_im_prediction(
    im_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    cfg = config or ModelConfig(coordinate_label="IM")
    return cross_validate_coordinate_prediction(
        coordinate_mat=im_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        coordinate_column="IM",
        config=cfg,
        min_runs_model=min_runs_model,
        group_col=group_col,
    )


def fit_production_im_models(
    im_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    config: Optional[ModelConfig] = None,
    min_runs_model: Optional[int] = None,
) -> CoordinateProductionResult:
    cfg = config or ModelConfig(coordinate_label="IM")
    return fit_coordinate_production_models(
        coordinate_mat=im_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        coordinate_column="IM",
        config=cfg,
        min_runs_model=min_runs_model,
    )


def run_im_prediction(
    im_mat: pd.DataFrame,
    lib_df_uniq_prec: pd.DataFrame,
    fit_fn: FitFunction,
    predict_fn: PredictFunction,
    out_dir: Union[str, Path],
    config: Optional[ModelConfig] = None,
    run_validation: bool = False,
    min_runs_model: Optional[int] = None,
    group_col: Optional[str] = None,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], CoordinateProductionResult]:
    cfg = config or ModelConfig(coordinate_label="IM")
    return run_coordinate_validation_and_production(
        coordinate_mat=im_mat,
        lib_df_uniq_prec=lib_df_uniq_prec,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        coordinate_column="IM",
        out_dir=out_dir,
        run_validation=run_validation,
        config=cfg,
        min_runs_model=min_runs_model,
        group_col=group_col,
        save_plots=save_plots,
    )
