# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Utilities for analyzing CC-Delta sweep results from local experiment outputs.

This module provides functions for:
- Loading experiment results from local parquet files
- Parsing generation parameters from experiment IDs
- Computing safety and utility metrics
- Aggregating scores and computing Pareto frontiers
- Plotting Pareto frontier charts with Altair
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Section A: Local results loading
# ---------------------------------------------------------------------------


def load_single_run(results_dir: Path) -> pd.DataFrame:
    """Load results from a single experiment run directory.

    Reads the text generation parquet and merges evaluation scores
    (strongreject fine-tuned, strongreject rubric, fluency) when available.

    Parameters
    ----------
    results_dir : Path
        Path to a single run's results directory.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with generation and evaluation columns.
    """
    results_dir = Path(results_dir)

    # Load base generation results
    gen_path = results_dir / "generate_with_hooks" / "text_generation.parquet"
    if not gen_path.exists():
        raise FileNotFoundError(f"Generation results not found: {gen_path}")
    df = pd.read_parquet(gen_path)

    # Merge evaluation scores
    eval_merges = [
        (
            "evaluate_strongreject",
            "*.fine_tuned.scored.parquet",
            "strongreject_fine_tuned",
        ),
        ("evaluate_strongreject", "*.rubric.scored.parquet", "strongreject_rubric"),
        ("evaluate_fluency", "*.fluency.scored.parquet", "fluency"),
        ("evaluate_orbench", "*.orbench.scored.parquet", "orbench_compliance"),
    ]

    for eval_subdir, glob_pattern, target_col in eval_merges:
        eval_dir = results_dir / eval_subdir
        if not eval_dir.exists():
            continue
        parquet_files = list(eval_dir.glob(glob_pattern))
        if not parquet_files:
            continue
        eval_df = pd.read_parquet(parquet_files[0])

        # Rename source column if needed
        if target_col == "fluency" and "fluency_score" in eval_df.columns:
            eval_df = eval_df.rename(columns={"fluency_score": "fluency"})
            # Normalize fluency from 1-5 scale to 0-1
            eval_df["fluency"] = (eval_df["fluency"] - 1) / 4

        # Keep only prompt + target column for merge
        merge_cols = ["prompt"]
        if target_col in eval_df.columns:
            merge_cols.append(target_col)
        else:
            continue
        eval_df = eval_df[merge_cols].drop_duplicates(subset=["prompt"])

        df = df.merge(eval_df, on="prompt", how="left")

    # Load eleuther eval results (ifeval, mmlu) — aggregate scores per run
    eleuther_dir = results_dir / "evaluate_eleuther"
    if eleuther_dir.exists():
        import pickle

        eleuther_metrics = {
            "eleuther_ifeval.pkl": (
                "ifeval",
                lambda r: r.get("ifeval", {}).get("prompt_level_strict_acc,none"),
            ),
            "eleuther_mmlu.pkl": (
                "mmlu",
                lambda r: r.get("mmlu", {}).get("acc,none"),
            ),
        }
        for pkl_name, (col_name, extractor) in eleuther_metrics.items():
            pkl_path = eleuther_dir / pkl_name
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                val = extractor(data.get("results", {}))
                if val is not None:
                    df[col_name] = float(val)
            except Exception:
                continue

    return df


def load_sweep_results(sweep_root: Path, method_label: str) -> pd.DataFrame:
    """Load results from a sweep directory containing multiple runs.

    Parameters
    ----------
    sweep_root : Path
        Root directory of the sweep. Each subdirectory is one run,
        or the root itself is a single run.
    method_label : str
        Label to tag all rows with (e.g., "sae", "dense").

    Returns
    -------
    pd.DataFrame
        Concatenated results from all runs, with an ``intervention_type`` column.
    """
    sweep_root = Path(sweep_root)

    # Check if this is a single run (has generate_with_hooks directly)
    if (sweep_root / "generate_with_hooks").exists():
        df = load_single_run(sweep_root)
        df["intervention_type"] = method_label
        return df

    # Otherwise iterate subdirectories
    dfs = []
    for subdir in sorted(sweep_root.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            run_df = load_single_run(subdir)
            dfs.append(run_df)
        except FileNotFoundError:
            print(f"WARNING: Skipping {subdir.name} (no generation results found)")
            continue

    if not dfs:
        raise FileNotFoundError(f"No valid runs found in {sweep_root}")

    df = pd.concat(dfs, ignore_index=True)
    df["intervention_type"] = method_label
    return df


# ---------------------------------------------------------------------------
# Section B: Generation param parsing
# ---------------------------------------------------------------------------


def parse_experiment_id(experiment_id: str) -> dict:
    """Parse an experiment ID back into a dictionary of parameters.

    Parameters
    ----------
    experiment_id : str
        String in format ``key1:value1__key2:value2__key3:value3``

    Returns
    -------
    dict
        Dictionary mapping parameter names to values (as strings).
    """
    if not experiment_id:
        return {}

    params = {}
    pairs = experiment_id.split("__")
    for pair in pairs:
        if ":" not in pair:
            continue  # Skip malformed pairs
        key, value = pair.split(":", 1)  # Split on first colon only
        params[key] = value

    return params


def expand_generation_params(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the ``generation_param`` column into separate columns.

    Parses experiment ID strings and creates individual columns for each
    hyperparameter, coercing numeric values where possible.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``generation_param`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for each parsed parameter.
    """
    df = df.copy()
    if "generation_param" not in df.columns:
        return df

    parsed = df["generation_param"].apply(
        lambda x: parse_experiment_id(str(x)) if pd.notna(x) else {}
    )
    param_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Coerce numeric columns
    for col in param_df.columns:
        param_df[col] = pd.to_numeric(param_df[col], errors="coerce").fillna(
            param_df[col]
        )

    # Avoid overwriting existing columns
    new_cols = [c for c in param_df.columns if c not in df.columns]
    df = pd.concat([df, param_df[new_cols]], axis=1)

    return df


# ---------------------------------------------------------------------------
# Section C: Metrics
# ---------------------------------------------------------------------------


def _sanitize_strongreject_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure strongreject columns are in [0, 1].

    Values outside this range are replaced with ``pd.NA`` and a summary
    warning is printed.
    """
    cols = ["strongreject_fine_tuned", "strongreject_rubric"]
    for col in cols:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found; skipping.")
            continue

        invalid_mask = (df[col] < 0) | (df[col] > 1)
        n_invalid = invalid_mask.sum()

        if n_invalid > 0:
            print(
                f"INFO: {n_invalid} invalid '{col}' values outside [0,1] set to pd.NA."
            )
            df.loc[invalid_mask, col] = pd.NA

    return df


def add_safety_score(df: pd.DataFrame) -> pd.Series:
    """Compute safety score from strongreject columns.

    Sanitizes strongreject columns and computes::

        safety = 1 - mean(strongreject_fine_tuned, strongreject_rubric)

    Returns
    -------
    pd.Series
        Series of safety scores.
    """
    print("Calculating safety score")
    df = _sanitize_strongreject_scores(df)

    if not {"strongreject_fine_tuned", "strongreject_rubric"}.issubset(df.columns):
        raise ValueError("Missing required columns for safety computation.")

    mean_strongreject = df[["strongreject_fine_tuned", "strongreject_rubric"]].mean(
        axis=1, skipna=True
    )
    safety = 1 - mean_strongreject
    print("Done calculating safety score")
    return safety


def add_utility(
    df: pd.DataFrame,
    metrics: list[str],
    weights: list[float] | None = None,
    normalize_weights: bool = True,
    new_col: str = "utility",
) -> pd.DataFrame:
    """Add a utility column as the weighted mean of metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing metric columns.
    metrics : list[str]
        List of metric column names to average.
    weights : list[float], optional
        Optional list of weights (same length as metrics). Defaults to equal weights.
    normalize_weights : bool
        Whether to normalize weights to sum to 1 (default: True).
    new_col : str
        Name of the output column to store the utility score.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new ``{new_col}`` column added.
    """
    df = df.copy()

    if weights is None:
        weights = [1.0] * len(metrics)
    elif len(weights) != len(metrics):
        raise ValueError("Length of weights must match metrics")

    weights = list(map(float, weights))
    if normalize_weights:
        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights must be non-zero")
        weights = [w / total for w in weights]

    # Weighted mean
    df[new_col] = sum(w * df[m] for w, m in zip(weights, metrics)) / sum(weights)

    return df


# ---------------------------------------------------------------------------
# Section D: Aggregation and Pareto
# ---------------------------------------------------------------------------


def group_mean_scores(
    df: pd.DataFrame,
    group_cols: str | list[str],
    score_cols: list[str],
    count_col_name: str = "sample_count",
    include_minmax: bool = False,
) -> pd.DataFrame:
    """Group by columns and compute mean of score columns.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    group_cols : str or list[str]
        Column name(s) to group by.
    score_cols : list[str]
        List of score column names to average.
    count_col_name : str
        Name of the new column for group sizes.
    include_minmax : bool
        If True, also compute min and max for each score column.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with one row per group.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if include_minmax:
        agg_dict = {col: ["mean", "min", "max"] for col in score_cols}
        agg_result = df.groupby(group_cols).agg(agg_dict)
        agg_result.columns = [f"{col}_{stat}" for col, stat in agg_result.columns]
        agg_result = agg_result.reset_index()
    else:
        means = df.groupby(group_cols)[score_cols].mean()
        agg_result = means.reset_index()

    counts = df.groupby(group_cols).size().rename(count_col_name)
    result = pd.concat([agg_result.set_index(group_cols), counts], axis=1).reset_index()
    return result


def save_altair(
    chart: alt.Chart,
    path: str | Path,
    fmt: str | None = None,
    scale_factor: int = 2,
    background: str | None = "white",
) -> Path:
    """Save an Altair chart as vector (pdf/svg) or high-res png.

    Uses the ``vl-convert`` backend. Requires ``pip install vl-convert-python``.

    Parameters
    ----------
    chart : alt.Chart
        The chart to save.
    path : str or Path
        Output file path.
    fmt : str, optional
        Format: "pdf", "svg", or "png". Inferred from path if None.
    scale_factor : int
        Scale factor for PNG output (default: 2).
    background : str, optional
        Background color (default: "white"). Set to None for transparent.

    Returns
    -------
    Path
        The output file path.
    """
    out = Path(path)
    if fmt is None:
        fmt = (out.suffix.lstrip(".") or "pdf").lower()
    else:
        out = out.with_suffix(f".{fmt.lower()}")
    out.parent.mkdir(parents=True, exist_ok=True)

    c = chart if background is None else chart.configure(background=background)

    if fmt in {"pdf", "svg"}:
        c.save(out.as_posix(), format=fmt, method="vl-convert")
    elif fmt == "png":
        c.save(
            out.as_posix(), format="png", method="vl-convert", scale_factor=scale_factor
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'pdf', 'svg', or 'png'.")

    return out


def plot_multi_metric(
    df: pd.DataFrame,
    score_cols: List[str],
    x_col: str,
    color_col: Optional[str] = None,
    reference_df: Optional[pd.DataFrame] = None,
    width: int = 180,
    height: int = 150,
    title: Optional[str] = None,
    score_labels: Optional[dict] = None,
) -> alt.HConcatChart:
    """Plot side-by-side line charts showing each metric vs a hyperparameter.

    Creates one line chart per metric, horizontally concatenated. Each chart
    shows how a score changes across values of ``x_col``, optionally colored
    by ``color_col``. A dashed reference line is drawn when ``reference_df``
    is provided.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated experiment data (one row per configuration).
    score_cols : list[str]
        Metric columns to plot (one chart per metric).
    x_col : str
        Column for the x-axis (e.g., "steering_multiplier", "strength").
    color_col : str, optional
        Column for color encoding (e.g., "intervention_type").
    reference_df : pd.DataFrame, optional
        Baseline reference data. Mean of each score column is shown as a
        dashed horizontal rule.
    width, height : int
        Dimensions for each individual chart.
    title : str, optional
        Overall chart title.
    score_labels : dict, optional
        Mapping from column name to display label
        (e.g., ``{"safety": "Safety Score"}``).

    Returns
    -------
    alt.HConcatChart
        Horizontally concatenated line charts.
    """
    if score_labels is None:
        score_labels = {}

    charts = []
    for i, score in enumerate(score_cols):
        if score not in df.columns:
            continue

        y_title = score_labels.get(score, score.replace("_", " ").title())
        x_title = x_col.replace("_", " ").title()

        encoding: dict = {
            "x": alt.X(f"{x_col}:Q", title=x_title),
            "y": alt.Y(f"{score}:Q", title=y_title, scale=alt.Scale(zero=False)),
            "tooltip": [
                alt.Tooltip(f"{x_col}:Q"),
                alt.Tooltip(f"{score}:Q", format=".3f"),
            ],
        }

        if color_col and color_col in df.columns:
            show_legend = i == 0
            encoding["color"] = alt.Color(
                f"{color_col}:N",
                title=color_col.replace("_", " ").title() if show_legend else None,
                legend=alt.Legend() if show_legend else None,
            )
            encoding["tooltip"].insert(0, alt.Tooltip(f"{color_col}:N"))

        line = (
            alt.Chart(df)
            .mark_line(point=True, strokeWidth=2)
            .encode(**encoding)
            .properties(width=width, height=height)
        )

        layers = [line]

        # Add reference line from baseline
        if reference_df is not None and score in reference_df.columns:
            ref_val = reference_df[score].mean()
            if pd.notna(ref_val):
                ref_line = (
                    alt.Chart(pd.DataFrame({score: [ref_val]}))
                    .mark_rule(
                        color="red", strokeDash=[4, 4], opacity=0.7, strokeWidth=1
                    )
                    .encode(y=alt.Y(f"{score}:Q"))
                )
                layers.append(ref_line)

        charts.append(alt.layer(*layers))

    if not charts:
        raise ValueError("No valid score columns found in data.")

    combined = alt.hconcat(*charts).resolve_scale(y="independent")
    if title:
        combined = combined.properties(
            title=alt.TitleParams(text=title, fontSize=14, anchor="middle")
        )
    return combined


# ---------------------------------------------------------------------------
# Section E: Optimal configuration search (paper methodology)
# ---------------------------------------------------------------------------


def _validate_scores_in_unit_interval(
    df: pd.DataFrame, cols: list[str], name: str, tol: float = 1e-9
) -> None:
    """Raise ValueError if any score column has values outside [0, 1]."""
    bad = {}
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            if not s.empty and ((s < -tol) | (s > 1 + tol)).any():
                bad[c] = int(((s < -tol) | (s > 1 + tol)).sum())
    if bad:
        details = ", ".join(f"{k} (violations={v})" for k, v in bad.items())
        raise ValueError(
            f"Expected scores in [0,1] for {name}; out-of-range: {details}"
        )


def _find_best_config_for_threshold(
    configs_with_baselines: pd.DataFrame,
    threshold_change: float,
    target_score: str,
    constraint_scores: List[str],
    tie_breakers: List[str],
    model_col: str = "model",
) -> list:
    """Find best configuration for each model at a specific threshold.

    A configuration is valid if every constraint score is within
    ``threshold_change`` of the baseline value. Among valid configs,
    the one maximizing ``target_score`` (with ``tie_breakers``) is selected.

    Parameters
    ----------
    configs_with_baselines : pd.DataFrame
        Configurations merged with baseline scores (has ``_baseline`` columns).
    threshold_change : float
        How much below baseline constraints can drop.
    target_score : str
        Score to maximize.
    constraint_scores : list[str]
        Scores that must satisfy constraints.
    tie_breakers : list[str]
        Scores to use for tie-breaking.
    model_col : str
        Column name for model grouping (default: "model").

    Returns
    -------
    list[pd.Series]
        Best configuration for each model.
    """
    results = []

    groups = (
        configs_with_baselines.groupby(model_col, sort=False)
        if model_col in configs_with_baselines.columns
        else [("all", configs_with_baselines)]
    )

    for _model_name, model_configs in groups:
        model_configs = model_configs.copy()

        # Compute constraint thresholds for this model
        for constraint_score in constraint_scores:
            baseline_value = model_configs[f"{constraint_score}_baseline"].iloc[0]
            threshold_value = max(baseline_value - threshold_change, 0.0)
            model_configs.loc[:, f"{constraint_score}_threshold"] = threshold_value

        # Filter to valid configurations (those meeting constraints)
        valid_configs = model_configs
        for constraint_score in constraint_scores:
            threshold_col = f"{constraint_score}_threshold"
            valid_configs = valid_configs.loc[
                valid_configs[constraint_score] >= valid_configs[threshold_col]
            ]

        if valid_configs.empty:
            continue

        # Pick best: maximize target_score, then tie_breakers
        sort_columns = [target_score] + [
            col for col in tie_breakers if col in valid_configs.columns
        ]
        sorted_configs = valid_configs.sort_values(
            sort_columns, ascending=False, kind="mergesort"
        )
        best_config = sorted_configs.iloc[0].copy()

        # Ensure baseline columns are present
        requested_scores = [target_score] + constraint_scores
        for score in requested_scores:
            base_col = f"{score}_baseline"
            if base_col not in best_config.index and base_col in model_configs.columns:
                best_config[base_col] = model_configs[base_col].iloc[0]

        results.append(best_config)

    return results


def find_optimal_configs(
    agg_df: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    threshold_changes: Optional[List[float]] = None,
    target_score: str = "safety",
    constraint_scores: Optional[List[str]] = None,
    tie_breakers: Optional[List[str]] = None,
    model_col: str = "model",
    verbose: bool = True,
) -> pd.DataFrame:
    """Find optimal configurations that maximize safety under utility constraints.

    For each threshold level, finds the configuration that maximizes
    ``target_score`` while keeping all ``constraint_scores`` within
    ``threshold_change`` of their baseline values.

    This implements the paper's methodology for finding best configurations:
    maximize safety subject to utility metrics not dropping more than X below
    baseline.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated data with one row per configuration.
    baseline_scores : pd.DataFrame
        Baseline scores (one row per model, or a single row).
    threshold_changes : list[float], optional
        Absolute threshold values for constraint relaxation.
        Default: ``[0.05, 0.10, 0.25, 1.0]``
    target_score : str
        Score to maximize (default: "safety").
    constraint_scores : list[str], optional
        Scores that must stay above ``baseline - threshold_change``.
        Default: uses all available from ``["fluency", "ifeval", "mmlu"]``.
    tie_breakers : list[str], optional
        Scores for tie-breaking (default: same as constraint_scores).
    model_col : str
        Column for model grouping (default: "model").
    verbose : bool
        Print diagnostic information (default: True).

    Returns
    -------
    pd.DataFrame
        One row per (model, threshold_change) with the optimal configuration.
    """
    if threshold_changes is None:
        threshold_changes = [0.05, 0.10, 0.25, 1.0]

    if constraint_scores is None:
        # Auto-detect available constraint scores
        candidate_constraints = ["fluency", "ifeval", "mmlu"]
        constraint_scores = [c for c in candidate_constraints if c in agg_df.columns]
        if not constraint_scores:
            raise ValueError(
                "No constraint scores found. Need at least one of: "
                + ", ".join(candidate_constraints)
            )

    if tie_breakers is None:
        tie_breakers = constraint_scores

    # Validate threshold_changes
    for tc in threshold_changes:
        if not (0.0 <= float(tc) <= 1.0):
            raise ValueError(f"All threshold_changes must be in [0,1], got {tc}")

    # Validate required columns
    requested_scores = [target_score] + list(constraint_scores)

    # If no model column, add a dummy one
    has_model = model_col in agg_df.columns
    if not has_model:
        agg_df = agg_df.copy()
        agg_df[model_col] = "default"
    if model_col not in baseline_scores.columns:
        baseline_scores = baseline_scores.copy()
        baseline_scores[model_col] = "default"

    _validate_scores_in_unit_interval(agg_df, requested_scores, "aggregated data")
    _validate_scores_in_unit_interval(
        baseline_scores, requested_scores, "baseline_scores"
    )

    # Merge baseline scores with aggregated data
    need_base = [model_col] + list(dict.fromkeys(requested_scores))
    need_base = [c for c in need_base if c in baseline_scores.columns]
    merged = agg_df.merge(
        baseline_scores[need_base],
        on=model_col,
        how="inner",
        suffixes=("", "_baseline"),
    )

    if merged.empty:
        raise ValueError(
            "No data after merging with baseline scores (no matching models?)"
        )

    if verbose:
        n_models = merged[model_col].nunique()
        n_configs = len(merged)
        print(f"Searching {n_configs} configurations across {n_models} model(s)")
        print(f"Maximizing '{target_score}' under constraints: {constraint_scores}")

    # Search for optimal configs at each threshold level
    all_results = []

    for threshold_change in threshold_changes:
        best_configs = _find_best_config_for_threshold(
            merged,
            threshold_change,
            target_score,
            constraint_scores,
            tie_breakers,
            model_col=model_col,
        )

        for config in best_configs:
            config["threshold_change"] = threshold_change

        all_results.extend(best_configs)

    if not all_results:
        if verbose:
            print("No valid configurations found at any threshold level")
        return pd.DataFrame()

    # Combine results
    final = pd.DataFrame(all_results)

    # Add *_diff columns (absolute difference from baseline)
    for sc in requested_scores:
        base_col = f"{sc}_baseline"
        if sc in final.columns and base_col in final.columns:
            final.loc[:, f"{sc}_diff"] = final[sc] - final[base_col]

    # Sort for stable presentation
    sort_cols = [c for c in [model_col, "threshold_change"] if c in final.columns]
    if sort_cols:
        final = final.sort_values(sort_cols).reset_index(drop=True)

    # Remove dummy model column if we added it
    if not has_model and model_col in final.columns:
        final = final.drop(columns=[model_col])

    if verbose:
        print(f"Found {len(final)} optimal configuration(s)")

    return final
