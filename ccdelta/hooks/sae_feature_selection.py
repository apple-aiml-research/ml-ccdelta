# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Utilities for performing statistical analysis on SAE features.
"""

import gc
import os
import time
import typing as t
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import torch
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)


def find_sublist_indices(
    main_list: t.List[str], sub_list: t.List[str]
) -> t.Tuple[t.Optional[int], t.Optional[int]]:
    """Find the start and end indices of a sublist within a main list."""
    sub_len = len(sub_list)
    if sub_len == 0:
        return None, None
    for i in range(len(main_list) - sub_len + 1):
        if main_list[i : i + sub_len] == sub_list:
            return i, i + sub_len - 1
    return None, None


def find_sublist_indices_with_boundary_tolerance(
    main_list: t.List[str], sub_list: t.List[str], boundary_tolerance: int = 1
) -> t.Tuple[t.Optional[int], t.Optional[int]]:
    """
    Find substring match with configurable boundary token tolerance for tokenization effects.

    The tolerance represents the maximum number of boundary tokens that can mismatch
    while still considering it a valid match. The algorithm tries to find the longest
    matching subsequence, allowing up to boundary_tolerance mismatches at each end.

    Args:
        main_list: The main token list to search in
        sub_list: The token subsequence to find
        boundary_tolerance: Maximum tokens that can mismatch at each boundary (default: 1)
    """
    if len(sub_list) == 0:
        return None, None

    # Try exact match first (fastest path)
    exact_result = find_sublist_indices(main_list, sub_list)
    if exact_result != (None, None):
        return exact_result

    # Determine the minimum core that must match
    # We need at least 1 token to match in the middle
    max_possible_tolerance = (len(sub_list) - 1) // 2
    effective_tolerance = min(boundary_tolerance, max_possible_tolerance)

    if effective_tolerance <= 0:
        # Can't use tolerance on such a short list, already tried exact match
        return None, None

    # Search for the core tokens (excluding up to effective_tolerance from each end)
    core_tokens = (
        sub_list[effective_tolerance:-effective_tolerance]
        if effective_tolerance > 0
        else sub_list
    )
    core_len = len(core_tokens)

    if core_len == 0:
        return None, None

    # Find where the core appears in main_list
    for i in range(len(main_list) - core_len + 1):
        if main_list[i : i + core_len] == core_tokens:
            # Found core match. Now determine the actual range to return.
            # Start by assuming we match the full range with tolerance
            start_idx = i - effective_tolerance
            end_idx = i + core_len + effective_tolerance - 1

            # Extend backwards if additional tokens match
            while start_idx > 0 and (i - start_idx) < len(sub_list):
                sub_idx = effective_tolerance - (i - start_idx)
                if sub_idx >= 0 and main_list[start_idx - 1] == sub_list[sub_idx - 1]:
                    start_idx -= 1
                else:
                    break

            # Extend forwards if additional tokens match
            while (
                end_idx < len(main_list) - 1
                and (end_idx - i - core_len + 1) < effective_tolerance
            ):
                sub_idx = effective_tolerance + core_len + (end_idx - i - core_len + 1)
                if (
                    sub_idx < len(sub_list)
                    and main_list[end_idx + 1] == sub_list[sub_idx]
                ):
                    end_idx += 1
                else:
                    break

            # Ensure we don't go out of bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(main_list) - 1, end_idx)

            return start_idx, end_idx

    return None, None


@torch.no_grad()
def get_pooled_features(
    feature_activations: torch.Tensor,
) -> t.Dict[int, t.Dict[str, float]]:
    """
    Compute pooled feature statistics across all tokens in a sequence.
    This function is sparsity-aware and handles both dense and sparse tensors.
    """
    device = feature_activations.device
    seq_len, num_features = feature_activations.shape
    feature_stats: t.Dict[int, t.Dict[str, float]] = {}

    # --- Sparsity-Aware Path ---
    if feature_activations.is_sparse:
        # Coalesce is important to sum up duplicate indices
        feature_activations = feature_activations.coalesce()
        indices, values = feature_activations.indices(), feature_activations.values()

        if values.numel() == 0:
            return feature_stats

        feature_ids = indices[1]

        # Use bincount for efficient counting and summation
        counts = torch.bincount(feature_ids, minlength=num_features)
        sums = torch.bincount(feature_ids, weights=values, minlength=num_features)

        # Populate stats only for features that were active
        active_feature_indices = torch.nonzero(counts, as_tuple=False).squeeze(1)

        if len(active_feature_indices) == 0:
            return feature_stats

        # Batch transfer all tensors to CPU at once (handling BFloat16)
        fids_cpu = active_feature_indices.detach().cpu().numpy()
        counts_cpu = counts[active_feature_indices].float().detach().cpu().numpy()
        sums_cpu = sums[active_feature_indices].float().detach().cpu().numpy()

        for i, fid in enumerate(fids_cpu):
            count_val = counts_cpu[i]
            sum_val = sums_cpu[i]
            feature_stats[int(fid)] = {
                "activation_avg": float(sum_val / count_val),
            }

        del indices, values, feature_ids, counts, sums
        del active_feature_indices, fids_cpu, counts_cpu, sums_cpu
        gc.collect(0)

        return feature_stats

    # --- Original Dense Tensor Path ---
    active_mask = feature_activations != 0
    has_active = active_mask.any(dim=0)
    active_feature_indices = torch.nonzero(has_active, as_tuple=False).squeeze(1)

    if len(active_feature_indices) == 0:
        return feature_stats

    active_features = feature_activations[:, active_feature_indices]
    valid_mask = active_mask[:, active_feature_indices] & (active_features != 0)
    non_zero_activations = active_features[valid_mask]

    if non_zero_activations.numel() == 0:
        return feature_stats

    non_zero_indices = torch.nonzero(valid_mask, as_tuple=False)[:, 1]
    original_dtype = feature_activations.dtype

    active_counts_per_feature = torch.bincount(
        non_zero_indices, minlength=len(active_feature_indices)
    ).to(dtype=original_dtype)

    sum_per_feature = torch.zeros(
        len(active_feature_indices), device=device, dtype=original_dtype
    )
    sum_per_feature.index_add_(0, non_zero_indices, non_zero_activations)

    mean_per_feature = torch.where(
        active_counts_per_feature > 0,
        sum_per_feature / active_counts_per_feature,
        torch.tensor(0.0, device=device, dtype=original_dtype),
    )

    fids_cpu = active_feature_indices.detach().cpu().numpy()
    means_cpu = mean_per_feature.float().detach().cpu().numpy()

    for i, fid in enumerate(fids_cpu):
        if active_counts_per_feature[i] > 0:
            feature_stats[int(fid)] = {
                "activation_avg": float(means_cpu[i]),
            }

    del active_mask, has_active, active_feature_indices
    del active_features, valid_mask, non_zero_activations, non_zero_indices
    del active_counts_per_feature, sum_per_feature, mean_per_feature
    del fids_cpu, means_cpu
    gc.collect(0)

    return feature_stats


def diff_pooled_feature_stats(
    stats_a: t.Dict[int, t.Dict[str, float]],
    stats_b: t.Dict[int, t.Dict[str, float]],
) -> t.Dict[int, t.Dict[str, float]]:
    """
    Return *A - B* for two feature-stats dictionaries.
    The output contains an entry for every feature appearing in *either* input.
    """
    if not stats_a and not stats_b:
        return {}

    result: t.Dict[int, t.Dict[str, float]] = {}
    all_feature_ids = set(stats_a.keys()) | set(stats_b.keys())

    for feature_id in all_feature_ids:
        entry_a = stats_a.get(feature_id, {})
        entry_b = stats_b.get(feature_id, {})

        diff: t.Dict[str, float] = {}
        all_keys = set(entry_a.keys()) | set(entry_b.keys())

        for k in all_keys:
            diff[k] = entry_a.get(k, 0.0) - entry_b.get(k, 0.0)

        result[feature_id] = diff

    return result


def metric_decile_thresholds(
    diff_stats_list: t.List[t.Dict[int, t.Dict[str, float]]],
    metric: str,
    *,
    absolute: bool = True,
) -> t.Dict[int, float]:
    """
    Compute decile (10-percent) thresholds for a metric across *all* diffs.
    """
    if not diff_stats_list:
        raise ValueError("diff_stats_list cannot be empty.")

    values: t.List[float] = []
    for diff in diff_stats_list:
        for diff_info in diff.values():
            val = diff_info.get(metric)
            if val is not None:
                values.append(abs(val) if absolute else float(val))

    if not values:
        raise ValueError(f"Metric '{metric}' not found in any diff stats.")

    arr = np.array(values, dtype=float)
    percentiles = range(10, 100, 10)  # 10,20,...,90
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def _compute_basic_feature_stats(
    diff_stats_list: t.List[t.Dict[int, t.Dict[str, float]]],
    metric: str,
    threshold: float = 0.0,
    min_observations: int = 5,
    max_observations_percent: float = 100.0,
) -> t.Dict[str, t.Dict[int, t.Dict[str, t.Any]]]:
    """
    Compute basic feature statistics without significance testing.

    Used for magnitude-only ranking where p-values are not needed.
    Much faster than full statistical testing with Wilcoxon tests.

    Args:
        diff_stats_list: List of feature difference dictionaries
        metric: The metric key to analyze
        threshold: Absolute threshold for filtering
        min_observations: Minimum samples a feature must appear in
        max_observations_percent: Maximum percentage of samples

    Returns:
        Dictionary with 'positive', 'negative', 'mixed' keys containing features
    """
    logger.info("Computing basic statistics (no significance testing)...")

    if not diff_stats_list:
        raise ValueError("diff_stats_list cannot be empty.")

    # Collect feature data
    feature_data = defaultdict(list)
    for diff in diff_stats_list:
        for fid, stats_data in diff.items():
            val = stats_data.get(metric)
            if val is not None and abs(val) >= threshold:
                feature_data[fid].append(float(val))

    # Filter by observation frequency
    total_samples = len(diff_stats_list)
    max_observations = int(total_samples * (max_observations_percent / 100))

    feature_data_filtered = {
        fid: values
        for fid, values in feature_data.items()
        if min_observations <= len(values) <= max_observations
    }

    if len(feature_data) > len(feature_data_filtered):
        logger.info(
            f"Filtered {len(feature_data) - len(feature_data_filtered)} features "
            f"by observation criteria (min={min_observations}, "
            f"max={max_observations}/{total_samples})"
        )

    feature_data = feature_data_filtered

    logger.info(
        f"Found {len(feature_data)} features meeting criteria "
        f"(min_obs={min_observations}, max_obs={max_observations})"
    )

    # Compute basic statistics (no Wilcoxon tests)
    feature_results: t.Dict[int, t.Dict[str, t.Any]] = {}

    for fid, values in feature_data.items():
        values_arr = np.array(values)
        mean_val = np.mean(values_arr)

        feature_results[fid] = {
            "n_observations": len(values),
            "median_diff": np.median(values_arr),
            "mean_diff": mean_val,
            "std_diff": np.std(values_arr),
            "pos_count": int(np.sum(values_arr > 0)),
            "neg_count": int(np.sum(values_arr < 0)),
            "prop_pos": float(np.sum(values_arr > 0) / len(values)),
            "prop_neg": float(np.sum(values_arr < 0) / len(values)),
            "p_corrected_pos": np.nan,  # Not computed for magnitude ranking
            "p_corrected_neg": np.nan,  # Not computed for magnitude ranking
            "classification": "positive" if mean_val > 0 else "negative",
        }

    # Categorize
    final_categorized = {
        "positive": {
            fid: stats
            for fid, stats in feature_results.items()
            if stats["classification"] == "positive"
        },
        "negative": {
            fid: stats
            for fid, stats in feature_results.items()
            if stats["classification"] == "negative"
        },
        "mixed": {},
    }

    logger.info(
        f"Categorized {len(final_categorized['positive'])} positive, "
        f"{len(final_categorized['negative'])} negative features"
    )

    return final_categorized


def consistently_signed_features_with_significance(
    diff_stats_list: t.List[t.Dict[int, t.Dict[str, float]]],
    metric: str,
    *,
    threshold: float = 0.0,
    p_value_threshold: float = 0.05,
    min_observations: int = 5,
    max_observations_percent: float = 100.0,
) -> t.Dict[str, t.Dict[int, t.Dict[str, t.Any]]]:
    """
    Identifies features with statistically significant consistent changes
    and returns associated statistical information.

    Args:
        diff_stats_list: A list of feature difference dictionaries for each sample.
        metric: The specific metric key to test (e.g., 'activation_max').
        threshold: The absolute threshold to filter feature values.
        p_value_threshold: The alpha for FDR correction.
        min_observations: The minimum number of samples a feature must appear in.
        max_observations_percent: Maximum percentage of samples a feature can appear in (default: 100.0).
            Features appearing in more than this percentage are excluded (e.g., 99.0 excludes features in >=99% of samples).

    Returns:
        A dictionary with 'positive', 'negative', and 'mixed' keys containing the
        categorized features and their statistics.
    """
    logger.info("Starting vectorized statistical analysis with Wilcoxon test...")
    if not diff_stats_list:
        raise ValueError("diff_stats_list cannot be empty.")

    # Step 1: Collect all feature data and filter based on the threshold.
    logger.info(f"Collecting feature data from {len(diff_stats_list)} samples...")
    t_start = time.time()
    feature_data = defaultdict(list)
    all_fids_observed = set()
    for diff in diff_stats_list:
        for fid, stats_data in diff.items():
            all_fids_observed.add(fid)
            val = stats_data.get(metric)
            if val is not None and abs(val) >= threshold:
                feature_data[fid].append(float(val))
    logger.info(
        f"Collected data for {len(feature_data)} features in {time.time() - t_start:.2f}s"
    )

    # Filter out features that appear in too many samples (e.g., always-on features)
    total_samples = len(diff_stats_list)
    max_observations = int(total_samples * (max_observations_percent / 100))
    features_before_max_filter = len(feature_data)

    feature_data_filtered = {
        fid: values
        for fid, values in feature_data.items()
        if len(values) <= max_observations
    }

    if features_before_max_filter > len(feature_data_filtered):
        logger.info(
            f"Filtered out {features_before_max_filter - len(feature_data_filtered)} features "
            f"appearing in >{max_observations_percent}% of samples "
            f"({max_observations}/{total_samples} samples)"
        )

    feature_data = feature_data_filtered

    # Step 2: Prepare for vectorized statistical tests.
    logger.info(f"Computing basic statistics for {len(feature_data)} features...")
    t_start = time.time()
    feature_results: t.Dict[int, t.Dict[str, t.Any]] = {}

    # First pass: compute basic statistics for all features
    features_to_test = []  # List of (fid, values, n_obs, std_val) tuples

    for fid, values in tqdm(
        feature_data.items(),
        desc="Computing feature statistics",
        disable=len(feature_data) < 100,
    ):
        n_obs = len(values)

        # Calculate all metrics for this feature - convert to numpy once
        values_arr = np.array(values) if values else np.array([])
        median_val = np.median(values_arr) if values else np.nan
        mean_val = np.mean(values_arr) if values else np.nan
        std_val = np.std(values_arr) if values else np.nan
        pos_count = np.sum(values_arr > 0)
        neg_count = np.sum(values_arr < 0)

        feature_results[fid] = {
            "n_observations": n_obs,
            "median_diff": median_val,
            "mean_diff": mean_val,
            "std_diff": std_val,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "prop_pos": pos_count / n_obs if n_obs > 0 else np.nan,
            "prop_neg": neg_count / n_obs if n_obs > 0 else np.nan,
            "p_corrected_pos": np.nan,
            "p_corrected_neg": np.nan,
            "classification": "mixed",
        }

        # Check conditions for testing
        if n_obs >= min_observations and std_val > 1e-9:
            features_to_test.append((fid, values, n_obs, std_val))

    # Second pass: parallelize Wilcoxon tests using ThreadPoolExecutor
    def compute_wilcoxon_tests(args):
        """Worker function for parallel Wilcoxon test computation."""
        fid, values, n_obs, std_val = args
        try:
            # Perform one-sided Wilcoxon tests
            stat_pos, p_pos = stats.wilcoxon(
                values, alternative="greater", zero_method="wilcox"
            )
            stat_neg, p_neg = stats.wilcoxon(
                values, alternative="less", zero_method="wilcox"
            )

            # Store W+ (sum of positive ranks) for rank-biserial correlation
            # scipy returns the statistic which is W+ for alternative="greater"
            # For signed-rank test, stat_pos IS W+
            w_plus = stat_pos

            return (fid, p_pos, p_neg, w_plus, True)
        except ValueError:
            return (fid, None, None, None, False)

    fids_to_test = []
    p_values_pos = []
    p_values_neg = []
    w_plus_values = []

    logger.info(f"Computed basic statistics in {time.time() - t_start:.2f}s")

    if features_to_test:
        # Use ThreadPoolExecutor for parallel processing
        # Limit workers to 4 to reduce memory overhead
        max_workers = min(4, os.cpu_count() // 4) if os.cpu_count() is not None else 1
        t_start = time.time()
        logger.info(
            f"Running {len(features_to_test)} Wilcoxon tests in parallel with {max_workers} workers..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(compute_wilcoxon_tests, features_to_test),
                    total=len(features_to_test),
                    desc="Wilcoxon tests",
                )
            )

        logger.info(f"Completed Wilcoxon tests in {time.time() - t_start:.2f}s")

        # Collect results from parallel execution
        for fid, p_pos, p_neg, w_plus, success in results:
            if success:
                fids_to_test.append(fid)
                p_values_pos.append(p_pos)
                p_values_neg.append(p_neg)
                w_plus_values.append(w_plus)

    logger.info(f"Number of features to test: {len(fids_to_test)}")

    if not fids_to_test:
        logger.warning("No features met the testing criteria. Returning all as mixed.")
        final_categorized_fids = {"positive": {}, "negative": {}, "mixed": {}}
        for fid in all_fids_observed:
            if fid not in feature_results:
                feature_results[fid] = {
                    "n_observations": 0,
                    "median_diff": np.nan,
                    "mean_diff": np.nan,
                    "std_diff": np.nan,
                    "pos_count": 0,
                    "neg_count": 0,
                    "prop_pos": np.nan,
                    "prop_neg": np.nan,
                    "p_corrected_pos": np.nan,
                    "p_corrected_neg": np.nan,
                    "classification": "mixed",
                }
            final_categorized_fids["mixed"][fid] = feature_results[fid]
        return final_categorized_fids

    # Step 3: FDR correction for multiple comparisons.
    t_start = time.time()
    logger.info(f"Applying FDR correction to {len(fids_to_test)} p-values...")
    _, corrected_p_pos = fdrcorrection(
        np.array(p_values_pos), alpha=p_value_threshold, method="indep"
    )
    _, corrected_p_neg = fdrcorrection(
        np.array(p_values_neg), alpha=p_value_threshold, method="indep"
    )
    logger.info(f"FDR correction completed in {time.time() - t_start:.2f}s")

    # Step 4: Final filtering and structuring the output.
    final_categorized_fids: t.Dict[str, t.Dict[int, t.Dict[str, t.Any]]] = {
        "positive": {},
        "negative": {},
        "mixed": {},
    }

    # First, populate the positive, negative, and mixed categories for tested features
    for i, fid in enumerate(fids_to_test):
        is_significant_pos = corrected_p_pos[i] < p_value_threshold
        is_significant_neg = corrected_p_neg[i] < p_value_threshold

        feature_results[fid]["p_corrected_pos"] = corrected_p_pos[i]
        feature_results[fid]["p_corrected_neg"] = corrected_p_neg[i]
        feature_results[fid]["wilcoxon_w_plus"] = w_plus_values[i]

        if is_significant_pos and not is_significant_neg:
            feature_results[fid]["classification"] = "positive"
            final_categorized_fids["positive"][fid] = feature_results[fid]
        elif is_significant_neg and not is_significant_pos:
            feature_results[fid]["classification"] = "negative"
            final_categorized_fids["negative"][fid] = feature_results[fid]
        else:
            feature_results[fid]["classification"] = "mixed"
            final_categorized_fids["mixed"][fid] = feature_results[fid]

    for fid in all_fids_observed:
        if (
            fid not in final_categorized_fids["positive"]
            and fid not in final_categorized_fids["negative"]
        ):
            if fid not in feature_results:
                feature_results[fid] = {
                    "n_observations": 0,
                    "median_diff": np.nan,
                    "mean_diff": np.nan,
                    "std_diff": np.nan,
                    "pos_count": 0,
                    "neg_count": 0,
                    "prop_pos": np.nan,
                    "prop_neg": np.nan,
                    "p_corrected_pos": np.nan,
                    "p_corrected_neg": np.nan,
                    "classification": "mixed",
                }
            final_categorized_fids["mixed"][fid] = feature_results[fid]

    logger.debug(f"Positive features found: {len(final_categorized_fids['positive'])}")
    logger.debug(f"Negative features found: {len(final_categorized_fids['negative'])}")

    return final_categorized_fids


def _compute_original_sample_statistics(
    paired_data: t.List[t.Dict[str, t.Any]],
    feature_ids: t.Iterable[int],
    feature_diffing_method: str,
    target_metric: str,
) -> t.Dict[int, t.Dict[str, float]]:
    """
    Compute statistics for original positive/negative samples (minimal helper for analysis).

    Args:
        paired_data: Original paired data
        feature_ids: Feature IDs to compute statistics for
        feature_diffing_method: Method used for feature diffing
        target_metric: The target metric (for consistency)

    Returns:
        Dictionary mapping feature_id -> statistics dict with mean_positive, mean_negative, std_positive, std_negative
    """
    feature_ids_set = set(feature_ids)
    positive_samples = defaultdict(list)  # fid -> list of positive activation values
    negative_samples = defaultdict(list)  # fid -> list of negative activation values

    logger.info(f"Processing {len(paired_data)} pairs to extract original sample statistics...")

    for pair in paired_data:
        source_tokens = pair["source_tokens"]
        if "row_data" in pair and "original_target_tokens" in pair["row_data"]:
            target_tokens = pair["row_data"]["original_target_tokens"]
        else:
            target_tokens = pair["target_tokens"]
        source_features = pair["source_features"]
        target_features = pair["target_features"]

        # Get positive (target) sample statistics
        pooled_positive = get_pooled_features(target_features)

        # Get negative (source/conditioned) sample statistics based on diffing method
        if feature_diffing_method in ["source_conditioned", "source_conditioned_with_fallback"]:
            start_idx, end_idx = find_sublist_indices_with_boundary_tolerance(
                source_tokens, target_tokens, boundary_tolerance=3
            )
            if start_idx is not None and end_idx is not None:
                # Use conditioned source features (matching the diffing logic)
                if source_features.is_sparse:
                    try:
                        conditioned_features = source_features.to_sparse_csr()[start_idx : end_idx + 1].to_sparse_coo()
                    except (NotImplementedError, RuntimeError):
                        conditioned_features = source_features.to_dense()[start_idx : end_idx + 1].to_sparse()
                else:
                    conditioned_features = source_features[start_idx : end_idx + 1]
                pooled_negative = get_pooled_features(conditioned_features)
                del conditioned_features
            else:
                # Fallback to full source if conditioning fails and method allows it
                if feature_diffing_method == "source_conditioned_with_fallback":
                    pooled_negative = get_pooled_features(source_features)
                else:
                    continue  # Skip this pair for strict source conditioning
        else:
            # Use full source features for regular diffing
            pooled_negative = get_pooled_features(source_features)

        # Collect values for features we care about
        for fid in feature_ids_set:
            if fid in pooled_positive:
                # Use activation_avg as the metric (get_pooled_features returns this key)
                positive_samples[fid].append(pooled_positive[fid]["activation_avg"])
            if fid in pooled_negative:
                negative_samples[fid].append(pooled_negative[fid]["activation_avg"])

    # Compute final statistics
    result = {}
    for fid in feature_ids_set:
        pos_values = positive_samples.get(fid, [])
        neg_values = negative_samples.get(fid, [])

        result[fid] = {
            "mean_positive": np.mean(pos_values) if pos_values else float('nan'),
            "std_positive": np.std(pos_values) if pos_values else float('nan'),
            "mean_negative": np.mean(neg_values) if neg_values else float('nan'),
            "std_negative": np.std(neg_values) if neg_values else float('nan'),
        }

    logger.info(f"Computed original sample statistics for {len(result)} features")
    return result


def select_steering_features(
    paired_data: t.List[t.Dict[str, t.Any]],
    target_metric: str,
    threshold_percentile: int,
    min_observations_percent: float,
    min_observations_absolute: int,
    p_value_threshold: float,
    feature_diffing_method: str,
    num_features: int,
    ranking_method: str,
    use_all_features: bool = False,
    max_observations_percent: float = 100.0,
    balance_pos_and_neg: bool = False,
) -> t.Dict[str, t.Dict[int, t.Dict[str, t.Any]]]:
    """
    Select statistically significant features for steering, or use all features without testing.

    Args:
        paired_data: A list of dictionaries, each containing source/target tokens and features.
        target_metric: The metric to use for significance testing (e.g., 'activation_max').
        threshold_percentile: The percentile for setting the activation threshold.
        min_observations_percent: The minimum percentage of samples a feature must appear in.
        min_observations_absolute: The minimum absolute number of samples a feature must appear in.
        p_value_threshold: The alpha for FDR correction.
        feature_diffing_method: which feature diff method to use.
        num_features: Number of top features to select from the combined pool of
            positive and negative features.
        ranking_method: Method for ranking features ('significance', 'effect_size', or 'magnitude').
            - 'significance': Rank by FDR-corrected p-value (requires statistical testing)
            - 'effect_size': Rank by Cohen's d style effect size (median_diff / std_diff)
            - 'magnitude': Rank by absolute mean_diff (CAA-style, skips statistical testing)
        use_all_features: If True, skip statistical testing and use all features found in data.
        max_observations_percent: Maximum percentage of samples a feature can appear in (default: 100.0).
            Features appearing in more than this percentage are excluded (e.g., 99.0 excludes features in >=99% of samples).
        balance_pos_and_neg: If True, select equal numbers from positive and negative features
            by splitting num_features in half. Default: False.

    Returns:
        A dictionary with 'positive' and 'negative' keys containing significant features.
        Each category will contain at most num_features features.
    """

    assert feature_diffing_method in [
        "source_conditioned",
        "source_conditioned_with_fallback",
        "diff",
    ]

    logger.info(
        f"**** select_steering_features:paired_data. {paired_data[0].keys()}\n\n {paired_data[0]['row_data']}"
    )
    # Step 1: Compute conditioned differences for all pairs
    logger.info(f"Computing feature differences for {len(paired_data)} sample pairs...")
    t_start = time.time()

    def process_pair(pair):
        """Worker function to process a single pair and compute feature diff."""
        with torch.no_grad():
            source_tokens = pair["source_tokens"]

            if "row_data" in pair and "original_target_tokens" in pair["row_data"]:
                # we want to search for the actual original prompt within the target.
                # however for response collection it was surrounded by the chat template/special tokens.
                target_tokens = pair["row_data"]["original_target_tokens"]
            else:
                target_tokens = pair["target_tokens"]
            source_features = pair["source_features"]
            target_features = pair["target_features"]

            def compute_conditioned_diff_features(
                source_features, target_features, start_idx, end_idx
            ):
                # Use the overlapping portion of source that matches target
                # Slicing is most efficient on the CSR format.
                if source_features.is_sparse:
                    try:
                        # Use CSR for efficient slicing if available
                        try:
                            conditioned_features = source_features.to_sparse_csr()[
                                start_idx : end_idx + 1
                            ].to_sparse_coo()
                        except (NotImplementedError, RuntimeError):
                            # Fallback for devices like MPS that may not support CSR
                            logger.debug(
                                "to_sparse_csr not implemented, falling back to dense slicing."
                            )
                            conditioned_features = source_features.to_dense()[
                                start_idx : end_idx + 1
                            ].to_sparse()
                    except (NotImplementedError, RuntimeError):
                        # Fallback for devices like MPS that may not support CSR
                        logger.debug(
                            "to_sparse_csr not implemented, falling back to dense slicing."
                        )
                        conditioned_features = source_features.to_dense()[
                            start_idx : end_idx + 1
                        ].to_sparse()
                else:
                    conditioned_features = source_features[start_idx : end_idx + 1]
                pooled_target = get_pooled_features(target_features)
                pooled_conditioned = get_pooled_features(conditioned_features)
                diff = diff_pooled_feature_stats(pooled_target, pooled_conditioned)
                # Clean up intermediate tensors
                del conditioned_features, pooled_target, pooled_conditioned
                return diff

            def compute_diff_features(source_features, target_features):
                pooled_source = get_pooled_features(source_features)
                pooled_target = get_pooled_features(target_features)
                diff = diff_pooled_feature_stats(pooled_target, pooled_source)
                # Clean up intermediate tensors
                del pooled_source, pooled_target
                return diff

            start_idx, end_idx = find_sublist_indices_with_boundary_tolerance(
                source_tokens, target_tokens, boundary_tolerance=3
            )
            found_sublist = start_idx is not None and end_idx is not None

            if feature_diffing_method == "source_conditioned":
                # Source conditioning mode: require substring matching
                if found_sublist:
                    logger.debug(
                        f"Found boundary-tolerant substring match for pair {pair.get('id', 'unknown')} at [{start_idx}:{end_idx}]"
                    )
                    # Use the overlapping portion of source that matches target
                    diff = compute_conditioned_diff_features(
                        source_features, target_features, start_idx, end_idx
                    )
                else:
                    # Fail early when source conditioning is required but substring match not found
                    # TODO should we just skip this sample and keep learning to be more robust? should we fallback
                    # to regular diff to supoprt things like base64?
                    raise ValueError(
                        f"Source conditioning required but substring match not found for pair {pair.get('id', 'unknown')}. "
                        f"Source tokens ({len(source_tokens)}): {source_tokens[:3]}...{source_tokens[-3:]}. "
                        f"Target tokens ({len(target_tokens)}): {target_tokens[:3]}...{target_tokens[-3:]}. "
                        f"\nSource tokens all: {source_tokens}. \n\n"
                        f"Target tokens all: {target_tokens}. \n\n"
                    )
            elif (
                feature_diffing_method == "source_conditioned_with_fallback"
                and found_sublist
            ):
                logger.debug(
                    f"Found boundary-tolerant substring match for pair {pair.get('id', 'unknown')} at [{start_idx}:{end_idx}]"
                )
                diff = compute_conditioned_diff_features(
                    source_features, target_features, start_idx, end_idx
                )
            else:
                logger.debug(
                    f"Using full feature comparison for pair {pair.get('id', 'unknown')}"
                )
                diff = compute_diff_features(source_features, target_features)

            # Clean up tensors after computing diff
            del source_features, target_features
            return diff

    # Process pairs in parallel with batching for memory management
    # Main bottlenecks: token matching, dictionary operations, and CPU tensor operations
    # Limit workers to 4 to reduce memory overhead from concurrent processing
    max_workers = min(4, os.cpu_count() // 4) if os.cpu_count() is not None else 1
    max_workers = int(max_workers)
    batch_size = 100  # Process 100 pairs at a time to allow memory cleanup
    logger.info(
        f"Processing {len(paired_data)} pairs with {max_workers} workers in batches of {batch_size}..."
    )

    diff_stats_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, len(paired_data), batch_size):
            batch_end = min(batch_start + batch_size, len(paired_data))
            batch_data = paired_data[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(paired_data) + batch_size - 1) // batch_size

            batch_results = list(
                tqdm(
                    executor.map(process_pair, batch_data),
                    total=len(batch_data),
                    desc=f"Computing feature diffs (batch {batch_num}/{total_batches})",
                )
            )

            diff_stats_list.extend(batch_results)

            # Explicit cleanup after each batch to free memory
            del batch_results, batch_data

            # Log memory before and after gc
            process = psutil.Process()
            mem_before_gb = process.memory_info().rss / 1024 / 1024 / 1024
            gc.collect(generation=2)  # Force full collection including old generation
            mem_after_gb = process.memory_info().rss / 1024 / 1024 / 1024
            mem_freed_gb = mem_before_gb - mem_after_gb

            logger.info(
                f"Batch {batch_num}/{total_batches}: GC freed {mem_freed_gb:.2f} GB, "
                f"current RAM: {mem_after_gb:.2f} GB"
            )

    logger.info(
        f"Computed {len(diff_stats_list)} feature diffs in {time.time() - t_start:.2f}s"
    )
    if not diff_stats_list:
        raise ValueError("Feature selection failed: diff_stats_list is empty.")

    if use_all_features:
        logger.info("Using all features without statistical testing")
        # Collect all features from diff_stats_list and compute basic stats
        all_feature_data = defaultdict(list)
        for diff in diff_stats_list:
            for fid, stats_data in diff.items():
                val = stats_data.get(target_metric)
                if val is not None:
                    all_feature_data[fid].append(float(val))

        # Create feature stats without significance testing
        all_significant_features = {}
        for fid, values in all_feature_data.items():
            if values:  # Only include features that have data
                # Convert to numpy once for all operations
                values_arr = np.array(values)
                median_val = np.median(values_arr)
                pos_count = np.sum(values_arr > 0)
                neg_count = np.sum(values_arr < 0)

                all_significant_features[fid] = {
                    "n_observations": len(values),
                    "median_diff": median_val,
                    "mean_diff": np.mean(values_arr),
                    "std_diff": np.std(values_arr),
                    "pos_count": pos_count,
                    "neg_count": neg_count,
                    "prop_pos": pos_count / len(values),
                    "prop_neg": neg_count / len(values),
                    "p_corrected_pos": np.nan,  # No statistical testing
                    "p_corrected_neg": np.nan,  # No statistical testing
                    "classification": "positive" if median_val > 0 else "negative",
                }

        logger.info(
            f"Total number of features (no testing): {len(all_significant_features)}"
        )

        # Categorize all features and return immediately
        all_positive = {
            fid: stats
            for fid, stats in all_significant_features.items()
            if stats.get("classification") == "positive"
        }
        all_negative = {
            fid: stats
            for fid, stats in all_significant_features.items()
            if stats.get("classification") == "negative"
        }

        logger.info(
            f"Using {len(all_positive)} positive features and {len(all_negative)} negative features (all available)"
        )

        return {
            "positive": all_positive,
            "negative": all_negative,
            "top_features": all_significant_features,
        }

    # Statistical feature selection path
    # Step 2: Determine the activation threshold
    logger.info("Computing activation threshold from deciles...")
    t_start = time.time()
    try:
        deciles = metric_decile_thresholds(diff_stats_list, metric=target_metric)
        threshold = deciles.get(threshold_percentile, 0.0)
        logger.info(f"Threshold computation took {time.time() - t_start:.2f}s")
        logger.info(f"Deciles: {str(deciles)}")
        logger.info(f"Calculated threshold: {threshold}")
    except ValueError as e:
        raise ValueError(
            f"Could not compute decile thresholds: {e}. Using threshold=0."
        ) from e

    # Step 3: Find statistically significant features
    min_observations = max(
        int(len(diff_stats_list) * (min_observations_percent / 100)),
        min_observations_absolute,
    )
    logger.info(f"Minimum observations for significance: {min_observations}")

    # OPTIMIZATION: Skip statistical testing if using magnitude ranking
    # (we don't need p-values or effect sizes for magnitude-based selection)
    skip_statistical_testing = (ranking_method == "magnitude")

    t_start = time.time()
    if skip_statistical_testing:
        logger.info(
            "Skipping statistical testing for magnitude ranking "
            "(p-values and effect sizes not needed) - using fast path"
        )
        # Compute basic statistics only (no Wilcoxon tests, no FDR correction)
        sig_features = _compute_basic_feature_stats(
            diff_stats_list=diff_stats_list,
            metric=target_metric,
            threshold=threshold,
            min_observations=min_observations,
            max_observations_percent=max_observations_percent,
        )
        logger.info(f"Basic statistics computed in {time.time() - t_start:.2f}s")
    else:
        # Full statistical testing path (Wilcoxon + FDR correction)
        sig_features = consistently_signed_features_with_significance(
            diff_stats_list,
            metric=target_metric,
            threshold=threshold,
            min_observations=min_observations,
            p_value_threshold=p_value_threshold,
            max_observations_percent=max_observations_percent,
        )
        logger.info(f"Statistical testing completed in {time.time() - t_start:.2f}s")

    positive_features = sig_features.get("positive", {})
    negative_features = sig_features.get("negative", {})
    all_significant_features = {**positive_features, **negative_features}

    if not all_significant_features:
        raise ValueError("No significant features found.")

    logger.info(
        f"Total number of significant features: {len(all_significant_features)}"
    )

    # Apply feature ranking and selection on the combined set
    if balance_pos_and_neg:
        logger.info(
            f"Applying balanced feature selection: {num_features//2} positive + {num_features//2} negative features using {ranking_method} method"
        )

        # Rank positive and negative features separately
        if ranking_method == "significance":
            ranked_positive = rank_features_by_significance(positive_features)
            ranked_negative = rank_features_by_significance(negative_features)
        elif ranking_method == "effect_size":
            ranked_positive = rank_features_by_effect_size(positive_features)
            ranked_negative = rank_features_by_effect_size(negative_features)
        elif ranking_method == "magnitude":
            ranked_positive = rank_features_by_magnitude(positive_features)
            ranked_negative = rank_features_by_magnitude(negative_features)
        elif ranking_method == "median":
            ranked_positive = rank_features_by_median(positive_features)
            ranked_negative = rank_features_by_median(negative_features)
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

        # Select top N/2 from each group
        n_per_class = num_features // 2
        top_positive_features = ranked_positive[:n_per_class]
        top_negative_features = ranked_negative[:n_per_class]

        # Combine and handle odd num_features by taking extra from better-ranked group
        top_features = top_positive_features + top_negative_features
        if num_features % 2 == 1:
            # Take one more from whichever group has a better top feature remaining
            next_pos = (
                ranked_positive[n_per_class]
                if len(ranked_positive) > n_per_class
                else None
            )
            next_neg = (
                ranked_negative[n_per_class]
                if len(ranked_negative) > n_per_class
                else None
            )

            if next_pos and next_neg:
                # Compare the ranking metric values to decide which is better
                if ranking_method == "significance":
                    # Lower p-value is better
                    pos_p = _compute_significance(next_pos[1])
                    neg_p = _compute_significance(next_neg[1])
                    top_features.append(next_pos if pos_p < neg_p else next_neg)
                elif ranking_method == "effect_size":
                    # Higher effect size is better
                    pos_effect = _compute_effect_size(next_pos[1])
                    neg_effect = _compute_effect_size(next_neg[1])
                    top_features.append(
                        next_pos if pos_effect > neg_effect else next_neg
                    )
                elif ranking_method == "magnitude":
                    # Higher magnitude is better
                    pos_mag = abs(next_pos[1].get("mean_diff", 0.0))
                    neg_mag = abs(next_neg[1].get("mean_diff", 0.0))
                    top_features.append(next_pos if pos_mag > neg_mag else next_neg)
                elif ranking_method == "median":
                    # Higher median magnitude is better
                    pos_median = _compute_median_magnitude(next_pos[1])
                    neg_median = _compute_median_magnitude(next_neg[1])
                    top_features.append(next_pos if pos_median > neg_median else next_neg)
            elif next_pos:
                top_features.append(next_pos)
            elif next_neg:
                top_features.append(next_neg)

        # For logging, create full ranked list by interleaving positive and negative
        all_ranked_features = []
        max_len = max(len(ranked_positive), len(ranked_negative))
        for i in range(max_len):
            if i < len(ranked_positive):
                all_ranked_features.append(ranked_positive[i])
            if i < len(ranked_negative):
                all_ranked_features.append(ranked_negative[i])
    else:
        logger.info(
            f"Applying feature ranking and selection: top {num_features} features using {ranking_method} method"
        )

        # Get full ranked list for logging purposes
        if ranking_method == "significance":
            all_ranked_features = rank_features_by_significance(
                all_significant_features
            )
        elif ranking_method == "effect_size":
            all_ranked_features = rank_features_by_effect_size(all_significant_features)
        elif ranking_method == "magnitude":
            all_ranked_features = rank_features_by_magnitude(all_significant_features)
        elif ranking_method == "median":
            all_ranked_features = rank_features_by_median(all_significant_features)
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

        # Select top N features
        top_features = all_ranked_features[:num_features]

    # Re-categorize the selected top features into positive and negative
    top_positive = {
        fid: stats
        for fid, stats in top_features
        if stats.get("classification") == "positive"
    }
    top_negative = {
        fid: stats
        for fid, stats in top_features
        if stats.get("classification") == "negative"
    }

    all_top_features = {fid: stats for fid, stats in top_features}

    # Add original positive/negative sample statistics for analysis (minimal change)
    logger.info("Computing original positive/negative sample statistics for selected features...")
    pos_neg_stats = _compute_original_sample_statistics(paired_data, all_top_features.keys(), feature_diffing_method, target_metric)

    # Merge the original sample statistics into the final results
    for fid in all_top_features:
        if fid in pos_neg_stats:
            all_top_features[fid].update(pos_neg_stats[fid])

    # Also update the positive and negative feature dictionaries
    for fid, stats in top_positive.items():
        if fid in pos_neg_stats:
            stats.update(pos_neg_stats[fid])
    for fid, stats in top_negative.items():
        if fid in pos_neg_stats:
            stats.update(pos_neg_stats[fid])

    logger.info(
        f"Selected {len(top_positive)} positive features and {len(top_negative)} negative features"
    )

    # Log detailed statistics for selected features
    logger.info("\n" + "=" * 80)
    logger.info(f"SELECTED TOP {num_features} FEATURES (sorted by {ranking_method}):")
    logger.info("=" * 80)
    for rank, (fid, stats) in enumerate(top_features, 1):
        effect_size = abs(stats.get("median_diff", float("nan"))) / (
            stats.get("std_diff", 1.0) + 1e-8
        )
        p_val = min(
            stats.get("p_corrected_pos", float("nan")),
            stats.get("p_corrected_neg", float("nan")),
        )
        logger.info(
            f"Rank {rank:3d} | Feature {fid:6d} | Class: {stats.get('classification', 'N/A'):8s} | "
            f"Effect: {effect_size:8.4f} | p-value: {p_val:10.10f} | "
            f"median_diff: {stats.get('median_diff', float('nan')):8.4f} | "
            f"n_obs: {stats.get('n_observations', 0):4d}"
        )

    # Log the next 10 features that were not selected
    if len(all_ranked_features) > num_features:
        logger.info("\n" + "=" * 80)
        logger.info(
            f"NEXT 10 FEATURES NOT SELECTED (ranks {num_features+1} to {min(num_features+10, len(all_ranked_features))}):"
        )
        logger.info("=" * 80)
        next_10_features = all_ranked_features[num_features : num_features + 10]
        for rank, (fid, stats) in enumerate(next_10_features, num_features + 1):
            effect_size = abs(stats.get("median_diff", float("nan"))) / (
                stats.get("std_diff", 1.0) + 1e-8
            )
            p_val = min(
                stats.get("p_corrected_pos", float("nan")),
                stats.get("p_corrected_neg", float("nan")),
            )
            logger.info(
                f"Rank {rank:3d} | Feature {fid:6d} | Class: {stats.get('classification', 'N/A'):8s} | "
                f"Effect: {effect_size:8.4f} | p-value: {p_val:10.20f} | "
                f"median_diff: {stats.get('median_diff', float('nan')):8.4f} | "
                f"n_obs: {stats.get('n_observations', 0):4d}"
            )

    # Log the last 10 features
    if len(all_ranked_features) > 10:
        logger.info("\n" + "=" * 80)
        logger.info(
            f"LAST 10 FEATURES (ranks {len(all_ranked_features)-9} to {len(all_ranked_features)}):"
        )
        logger.info("=" * 80)
        last_10_features = all_ranked_features[-10:]
        for rank, (fid, stats) in enumerate(
            last_10_features, len(all_ranked_features) - 9
        ):
            effect_size = abs(stats.get("median_diff", float("nan"))) / (
                stats.get("std_diff", 1.0) + 1e-8
            )
            p_val = min(
                stats.get("p_corrected_pos", float("nan")),
                stats.get("p_corrected_neg", float("nan")),
            )
            logger.info(
                f"Rank {rank:3d} | Feature {fid:6d} | Class: {stats.get('classification', 'N/A'):8s} | "
                f"Effect: {effect_size:8.4f} | p-value: {p_val:10.10f} | "
                f"median_diff: {stats.get('median_diff', float('nan')):8.4f} | "
                f"n_obs: {stats.get('n_observations', 0):4d}"
            )
    logger.info("=" * 80 + "\n")

    return {
        "positive": top_positive,
        "negative": top_negative,
        "top_features": all_top_features,
    }


def _compute_effect_size(stats: t.Dict[str, t.Any]) -> float:
    """
    Compute Cohen's d style effect size from feature statistics.

    Args:
        stats: Dictionary containing 'median_diff' and 'std_diff' keys

    Returns:
        Effect size as abs(median_diff) / (std_diff + epsilon), or NaN if not computable
    """
    median_diff = stats.get("median_diff")
    std_diff = stats.get("std_diff")

    if median_diff is None or std_diff is None:
        return np.nan

    return abs(median_diff) / (std_diff + 1e-8)


def _compute_median_magnitude(stats: t.Dict[str, t.Any]) -> float:
    """
    Compute unstandardized median difference magnitude.

    Args:
        stats: Dictionary containing 'median_diff' key

    Returns:
        Absolute median difference, or NaN if not computable
    """
    median_diff = stats.get("median_diff")
    if median_diff is None:
        return np.nan
    return abs(median_diff)


def _compute_significance(stats: t.Dict[str, t.Any]) -> float:
    """
    Compute significance as the minimum (most significant) p-value from feature statistics.

    Args:
        stats: Dictionary containing 'p_corrected_pos' and 'p_corrected_neg' keys

    Returns:
        Minimum p-value (lower is more significant), or NaN if not computable
    """
    p_pos = stats.get("p_corrected_pos", np.nan)
    p_neg = stats.get("p_corrected_neg", np.nan)
    return float(np.nanmin([p_pos, p_neg]))


def _asc_norm(x, handle_nan_values: str = "end") -> float:
    """
    Normalize an ascending metric (smaller is better, e.g., p-value).
    NaN/None values are mapped to ±inf based on handle_nan_values policy.

    Args:
        x: The metric value (can be None, NaN, or numeric)
        handle_nan_values: Where to place NaN values ('start' or 'end')

    Returns:
        float: The normalized value for sorting
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("inf") if handle_nan_values == "end" else float("-inf")
    return float(x)


def _desc_norm(x, handle_nan_values: str = "end") -> float:
    """
    Normalize a descending metric (larger is better, e.g., effect size, observation count).
    Returns negative value for proper sorting in ascending order.
    NaN/None values are mapped to ±inf based on handle_nan_values policy.

    Args:
        x: The metric value (can be None, NaN, or numeric)
        handle_nan_values: Where to place NaN values ('start' or 'end')

    Returns:
        float: The negative normalized value for sorting
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("inf") if handle_nan_values == "end" else float("-inf")
    return -float(x)


def rank_features_by_significance(
    features_dict: t.Dict[int, t.Dict[str, t.Any]],
    top_n: t.Optional[int] = None,
    handle_nan_values: str = "end",
) -> t.List[t.Tuple[int, t.Dict[str, t.Any]]]:
    """
    Ranks features by their statistical significance (FDR-corrected p-value),
    using the more significant p-value of either the positive or negative test.

    Tie-breaking:
    - Primary: p-value (ascending - smaller is better)
    - Secondary: effect size (descending - larger is better)
    - Tertiary: number of observations (descending - more is better)
    """

    def key_extractor(stats):
        """Extract sorting key with tie-breaking and NaN handling."""
        # Get p-value (NaN-safe minimum of both tests)
        p_val = _compute_significance(stats)

        # Get effect size
        effect = _compute_effect_size(stats)

        # Get observations count (None if missing to distinguish from real zero)
        n_obs = stats.get("n_observations")

        # Return tuple: (p-value, -effect_size, -n_observations)
        return (
            _asc_norm(p_val, handle_nan_values),
            _desc_norm(effect, handle_nan_values),
            _desc_norm(n_obs, handle_nan_values),
        )

    ranked_list = sorted(features_dict.items(), key=lambda item: key_extractor(item[1]))

    if top_n is not None:
        return ranked_list[:top_n]
    return ranked_list


def rank_features_by_effect_size(
    features_dict: t.Dict[int, t.Dict[str, t.Any]],
    top_n: t.Optional[int] = None,
    handle_nan_values: str = "end",
) -> t.List[t.Tuple[int, t.Dict[str, t.Any]]]:
    """
    Ranks features by the absolute magnitude of their effect size (median_diff / std_diff).
    Larger effect sizes are ranked higher (Cohen's d style).

    Tie-breaking:
    - Primary: effect size (descending - larger is better)
    - Secondary: p-value (ascending - smaller/more significant is better)
    - Tertiary: number of observations (descending - more is better)

    Args:
        features_dict: A dictionary of features. Requires 'median_diff' and 'std_diff' in feature stats.
        top_n: Optional. If provided, returns only the top N features by effect size.
        handle_nan_values: How to treat NaN values in the ranking ('start' or 'end').

    Returns:
        A list of (feature_id, stats_dict) tuples, ranked by effect size (descending).
    """

    def key_extractor(stats):
        """Extract sorting key with tie-breaking and NaN handling."""
        # Get effect size
        effect = _compute_effect_size(stats)

        # Get p-value (NaN-safe minimum of both tests)
        p_val = _compute_significance(stats)

        # Get observations count (None if missing to distinguish from real zero)
        n_obs = stats.get("n_observations")

        # Return tuple: (-effect_size, p_value, -n_observations)
        return (
            _desc_norm(effect, handle_nan_values),
            _asc_norm(p_val, handle_nan_values),
            _desc_norm(n_obs, handle_nan_values),
        )

    ranked_list = sorted(features_dict.items(), key=lambda item: key_extractor(item[1]))

    if top_n is not None:
        return ranked_list[:top_n]
    return ranked_list


def rank_features_by_magnitude(
    features_dict: t.Dict[int, t.Dict[str, t.Any]],
    top_n: t.Optional[int] = None,
    handle_nan_values: str = "end",
) -> t.List[t.Tuple[int, t.Dict[str, t.Any]]]:
    """
    Ranks features by absolute magnitude of mean_diff (CAA-style).

    This ranking ignores statistical significance and variance, selecting
    features purely by the size of their mean difference across pairs.
    Equivalent to CAA's approach of ranking by |mean(target) - mean(source)|.

    Tie-breaking:
    - Primary: abs(mean_diff) (descending - larger magnitude is better)
    - Secondary: number of observations (descending - more data is better)

    Args:
        features_dict: Dictionary of features with statistics
        top_n: Optional number of top features to return
        handle_nan_values: How to handle NaN values ('start' or 'end')

    Returns:
        List of (feature_id, stats_dict) tuples ranked by magnitude
    """

    def key_extractor(stats):
        """Extract sorting key with tie-breaking and NaN handling."""
        mean_diff = stats.get("mean_diff", 0.0)
        n_obs = stats.get("n_observations")

        # Return tuple: (-abs(mean_diff), -n_observations)
        return (
            _desc_norm(abs(mean_diff), handle_nan_values),
            _desc_norm(n_obs, handle_nan_values),
        )

    ranked_list = sorted(features_dict.items(), key=lambda item: key_extractor(item[1]))

    if top_n is not None:
        return ranked_list[:top_n]
    return ranked_list


def rank_features_by_median(
    features_dict: t.Dict[int, t.Dict[str, t.Any]],
    top_n: t.Optional[int] = None,
    handle_nan_values: str = "end",
) -> t.List[t.Tuple[int, t.Dict[str, t.Any]]]:
    """
    Ranks features by absolute magnitude of their median difference (unstandardized).

    Tie-breaking:
    - Primary: abs(median_diff) (descending - larger is better)
    - Secondary: p-value (ascending - more significant is better)
    - Tertiary: number of observations (descending - more is better)

    Args:
        features_dict: Dictionary of features with statistics
        top_n: Optional number of top features to return
        handle_nan_values: How to handle NaN values ('start' or 'end')

    Returns:
        List of (feature_id, stats_dict) tuples ranked by median magnitude
    """

    def key_extractor(stats):
        """Extract sorting key with tie-breaking and NaN handling."""
        median_mag = _compute_median_magnitude(stats)
        p_val = _compute_significance(stats)
        n_obs = stats.get("n_observations")

        return (
            _desc_norm(median_mag, handle_nan_values),
            _asc_norm(p_val, handle_nan_values),
            _desc_norm(n_obs, handle_nan_values),
        )

    ranked_list = sorted(features_dict.items(), key=lambda item: key_extractor(item[1]))

    if top_n is not None:
        return ranked_list[:top_n]
    return ranked_list


def get_top_features(
    features_dict: t.Dict[int, t.Dict[str, t.Any]],
    num_features: int,
    ranking_method: str,
) -> t.List[t.Tuple[int, t.Dict[str, t.Any]]]:
    """
    Select feature IDs from the loaded features dictionary using ranking methods.
    This updated function ranks features from both positive and negative groups
    together to select the overall top features.

    Args:
        features_dict: A dictionary of features (e.g., combined significant features).
        num_features: Number of top features to return.
        ranking_method: The ranking method to use ('significance', 'effect_size', 'magnitude',
            or 'median').

    Returns:
        A list of (feature_id, stats_dict) tuples for the top ranked features.
    """
    # Rank features based on the specified method
    if ranking_method == "significance":
        # Note: We rank by the minimum p-value (either positive or negative) to find the most significant features overall.
        ranked_features = rank_features_by_significance(
            features_dict, top_n=num_features
        )
    elif ranking_method == "effect_size":
        ranked_features = rank_features_by_effect_size(
            features_dict, top_n=num_features
        )
    elif ranking_method == "magnitude":
        ranked_features = rank_features_by_magnitude(
            features_dict, top_n=num_features
        )
    elif ranking_method == "median":
        ranked_features = rank_features_by_median(
            features_dict, top_n=num_features
        )
    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    return ranked_features
