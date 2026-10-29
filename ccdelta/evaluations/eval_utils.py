# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_aggregate_stats(
    results: pd.DataFrame, output_column: str, input_filename: str
) -> dict:
    """
    Compute aggregate statistics for evaluation results.

    Args:
        results: DataFrame containing the evaluation results
        output_column: Name of the column containing the scores
        input_filename: Name of the input file

    Returns:
        Dictionary containing aggregate statistics
    """
    valid_scores = results[output_column].dropna()
    if len(valid_scores) > 0:
        return {
            "input_file": input_filename,
            "total_rows": len(results),
            "successfully_scored": len(valid_scores),
            "mean_score": float(valid_scores.mean()),
            "median_score": float(valid_scores.median()),
            "std_score": float(valid_scores.std()),
            "min_score": float(valid_scores.min()),
            "max_score": float(valid_scores.max()),
            "score_column": output_column,
            "score_distribution": {
                "q25": float(valid_scores.quantile(0.25)),
                "q50": float(valid_scores.quantile(0.50)),
                "q75": float(valid_scores.quantile(0.75)),
                "q90": float(valid_scores.quantile(0.90)),
                "q95": float(valid_scores.quantile(0.95)),
                "q99": float(valid_scores.quantile(0.99)),
            },
        }
    else:
        return {
            "input_file": input_filename,
            "total_rows": len(results),
            "successfully_scored": 0,
            "error": "No valid scores generated",
            "score_column": output_column,
        }


def save_results(
    results: pd.DataFrame,
    output_path: Path,
    input_path: str,
    aggregate_stats: dict,
    evaluator: str,
) -> None:
    """
    Save raw evaluation results to parquet file and aggregate scores to JSON.

    Args:
        results: DataFrame containing the evaluation results
        output_path: Directory path where results will be saved
        input_path: Path to the input file (used to generate output filename)
        aggregate_stats: Dictionary containing aggregate statistics to save as JSON
        evaluator: Name of the evaluator type (for unique filenames)
    """
    # Extract filename from input path (handle both local paths and conductor URLs)
    if "/" in input_path:
        input_filename = input_path.split("/")[-1]
    else:
        input_filename = input_path

    # Get the stem (filename without extension) and create output filename with evaluator type
    input_stem = Path(input_filename).stem
    out_filename = f"{input_stem}.{evaluator}.scored.parquet"
    out_file = output_path / out_filename

    logger.info(f"Saving eval results to {out_file}")
    results.to_parquet(out_file, index=False)

    # Save JSON summary with evaluator type in filename
    json_filename = f"{input_stem}.{evaluator}.scored.summary.json"
    json_file = output_path / json_filename

    with open(json_file, "w") as f:
        json.dump(aggregate_stats, f, indent=2)

    logger.info(f"Saved aggregate scores to {json_file}")
