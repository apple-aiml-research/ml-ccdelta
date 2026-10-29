# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import logging
import typing as t
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from strong_reject.evaluate import evaluate_dataset, evaluate
import datasets
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ccdelta.datasets.tabular_dataset import load_tabular_data
from ccdelta.evaluations.eval_utils import compute_aggregate_stats, save_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _evaluate_single_sample(args: tuple) -> dict | None:
    """Helper function to evaluate a single sample - needed for multiprocessing."""
    i, prompt, response, evaluator_name = args
    try:
        eval_results = evaluate(
            forbidden_prompt=prompt, response=response, evaluators=[evaluator_name]
        )

        if eval_results and len(eval_results) > 0:
            result_dict = eval_results[0]
            extracted_result = {
                "score": result_dict.get("score", None),
                "refusal": result_dict.get("refusal", None),
            }
            return extracted_result
        else:
            return None
    except Exception as e:
        logger.debug(f"Sample {i} failed evaluation (likely degenerate text): {e}")
        return None


def _evaluate_rubric_individually(
    df_clean: pd.DataFrame,
    prompt_column: str,
    response_column: str,
    evaluator_name: str,
    max_workers: int = 4,
) -> list[dict | None]:
    """
    Evaluate rubric samples individually using multiprocessing for speed.

    Args:
        df_clean: Clean DataFrame with samples to evaluate
        prompt_column: Name of the prompt column
        response_column: Name of the response column
        evaluator_name: StrongReject evaluator name
        max_workers: Number of parallel workers (default: 4 to avoid rate limits)

    Returns:
        List of dictionaries with evaluation results, None for completely failed samples
    """
    # Prepare arguments for parallel processing
    sample_args = []
    for i, (_, row) in enumerate(df_clean.iterrows()):
        sample_args.append(
            (i, row[prompt_column], row[response_column], evaluator_name)
        )

    results: list[dict | None] = []
    for _ in range(len(sample_args)):
        results.append(None)

    # Use ThreadPoolExecutor for I/O bound API calls (better than ProcessPoolExecutor for API calls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_evaluate_single_sample, args): i
            for i, args in enumerate(sample_args)
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_index),
            total=len(sample_args),
            desc="Evaluating rubric samples",
        ):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.debug(f"Sample {index} failed evaluation: {e}")
                results[index] = None

    return results


def evaluate_strongreject(
    data_path: str,
    prompt_column: str = "prompt",
    response_column: str = "steered_text",
    evaluator: str = "fine_tuned",
    batch_size: int = 1,
    num_sentences: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate text generation results using StrongReject evaluator.

    Args:
        data_path: Path to the data file containing text generation results
        prompt_column: Name of the column containing prompts (default: "prompt")
        response_column: Name of the column containing responses to evaluate (default: "steered_text")
        evaluator: StrongReject evaluator type: "fine_tuned" or "rubric" (default: "fine_tuned")
        batch_size: Batch size for evaluation (default: 1)
        num_sentences: Limit number of sentences to evaluate (default: None for all)

    Returns:
        DataFrame with original data plus StrongReject scores
    """
    logger.info(f"Loading data from {data_path}")
    # load_tabular_data now handles both strings and lists of strings
    df = load_tabular_data(data_path)

    # Validate required columns exist
    if prompt_column not in df.columns:
        raise ValueError(
            f"Prompt column '{prompt_column}' not found in data. Available columns: {list(df.columns)}"
        )
    if response_column not in df.columns:
        raise ValueError(
            f"Response column '{response_column}' not found in data. Available columns: {list(df.columns)}"
        )

    # Filter out rows with missing data
    logger.info(f"Original data has {len(df)} rows")
    df_clean = df.dropna(subset=[prompt_column, response_column]).copy()
    logger.info(f"After removing missing data: {len(df_clean)} rows")

    # Limit number of sentences if specified
    if num_sentences is not None and num_sentences > 0:
        df_clean = df_clean.head(num_sentences)
        logger.info(f"Limited to {num_sentences} sentences: {len(df_clean)} rows")

    if len(df_clean) == 0:
        logger.warning("No valid data to evaluate")
        df["strongreject_score"] = float("nan")
        return df

    # Determine evaluator name
    if evaluator == "fine_tuned":
        evaluator_name = "strongreject_finetuned"
    elif evaluator == "rubric":
        evaluator_name = "strongreject_rubric"
    else:
        raise ValueError(
            f"Unknown evaluator type: {evaluator}. Use 'fine_tuned' or 'rubric'"
        )

    logger.info(f"Using StrongReject evaluator: {evaluator_name}")

    # Add scores column first
    score_column = f"strongreject_{evaluator}"
    df[score_column] = float("nan")
    clean_indices = df_clean.index.tolist()

    # Use different evaluation strategies based on evaluator type
    if evaluator == "rubric":
        # Use individual evaluation for rubric to handle parsing failures gracefully
        logger.info("Running StrongReject rubric evaluation (individual samples)...")

        MAX_WORKERS = 16
        eval_results = _evaluate_rubric_individually(
            df_clean,
            prompt_column,
            response_column,
            evaluator_name,
            max_workers=MAX_WORKERS,
        )

        # Process results - initialize refusal column
        refusal_column = f"strongreject_{evaluator}_refusal"
        df[refusal_column] = float("nan")

        successful_scores = 0
        successful_refusals = 0

        for idx, result in zip(clean_indices, eval_results):
            if result is not None:
                # Extract score (may be None if parsing failed)
                score = result.get("score")
                if score is not None and not pd.isna(score):
                    df.loc[idx, score_column] = score
                    successful_scores += 1

                # Extract refusal (may be None if parsing failed)
                refusal = result.get("refusal")
                if refusal is not None and not pd.isna(refusal):
                    df.loc[idx, refusal_column] = refusal
                    successful_refusals += 1
            # If result is None, leave both columns as NaN (already initialized)

        logger.info(f"Rubric evaluation completed:")
        logger.info(f"  - Successful scores: {successful_scores}/{len(clean_indices)}")
        logger.info(
            f"  - Successful refusals: {successful_refusals}/{len(clean_indices)}"
        )
        logger.info(f"Added refusal scores to column: {refusal_column}")

    else:
        # Use batch evaluation for fine_tuned evaluator (more efficient)
        # Create dataset only when needed for batch processing
        eval_data = {
            "forbidden_prompt": df_clean[prompt_column].tolist(),
            "response": df_clean[response_column].tolist(),
        }
        hf_dataset = datasets.Dataset.from_dict(eval_data)

        try:
            logger.info("Running StrongReject evaluation...")
            eval_dataset = evaluate_dataset(
                hf_dataset, [evaluator_name], batch_size=batch_size
            )
            eval_df = eval_dataset.to_pandas()

            # Check if expected 'score' column exists
            if "score" not in eval_df.columns:
                logger.warning(
                    f"StrongReject evaluation completed but 'score' column not found. Available columns: {list(eval_df.columns)}"
                )
            else:
                scores = eval_df["score"].tolist()
                # Map scores back to original indices
                for idx, score in zip(clean_indices, scores):
                    df.loc[idx, score_column] = score

        except Exception as e:
            logger.error(f"Error during StrongReject evaluation: {e}")
            # For batch evaluation errors, assign NaN scores to all samples
            logger.warning(
                "Assigning NaN scores to all samples due to evaluation failure"
            )
            for idx in clean_indices:
                df.loc[idx, score_column] = float("nan")

    # Log summary statistics
    valid_scores = df[score_column].dropna()
    total_samples = len(clean_indices)
    if len(valid_scores) > 0:
        logger.info(f"StrongReject evaluation completed:")
        logger.info(f"  - Total samples: {total_samples}")
        logger.info(f"  - Successfully scored: {len(valid_scores)}")
        logger.info(
            f"  - Failed/degenerate samples: {total_samples - len(valid_scores)}"
        )
        logger.info(f"  - Mean score: {valid_scores.mean():.4f}")
        logger.info(
            f"  - Score range: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]"
        )
    else:
        logger.warning(
            "No valid scores were generated - all samples may contain degenerate text"
        )

    return df


def run_strongreject_eval(
    data_path: str,
    prompt_column: str = "prompt",
    response_column: str = "steered_text",
    evaluator: str = "fine_tuned",
    batch_size: int = 1,
    num_sentences: int | None = None,
    results_dir: str | None = None,
) -> None:
    """
    Main function to run StrongReject evaluation and save results.

    Args:
        data_path: Path to the data file containing text generation results
        prompt_column: Name of the column containing prompts (default: "prompt")
        response_column: Name of the column containing responses to evaluate (default: "steered_text")
        evaluator: StrongReject evaluator type: "fine_tuned" or "rubric" (default: "fine_tuned")
        batch_size: Batch size for evaluation (default: 1)
        num_sentences: Limit number of sentences to evaluate (default: None for all)
        results_dir: Directory to save results (default: "./results")
    """
    # Run evaluation
    results = evaluate_strongreject(
        data_path=data_path,
        prompt_column=prompt_column,
        response_column=response_column,
        evaluator=evaluator,
        batch_size=batch_size,
        num_sentences=num_sentences,
    )

    # Set up output directory
    if results_dir is not None:
        output_path = Path(results_dir) / Path(__file__).stem
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("./results") / Path(__file__).stem
        output_path.mkdir(parents=True, exist_ok=True)

    # Generate output column name and compute aggregate statistics
    output_column = f"strongreject_{evaluator}"

    # Extract filename from input path for statistics
    if "/" in data_path:
        input_filename = data_path.split("/")[-1]
    else:
        input_filename = data_path

    # Compute aggregate statistics
    aggregate_stats = compute_aggregate_stats(results, output_column, input_filename)

    # Save results with JSON summary
    save_results(results, output_path, data_path, aggregate_stats, evaluator)

    logger.info(f"StrongReject evaluation completed. Results saved to {output_path}")


@hydra.main(config_path="../configs", config_name="text_generation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for standalone execution."""
    # Extract parameters from config and call explicit parameter function
    run_strongreject_eval(
        data_path=cfg.data_path,
        prompt_column=cfg.get("prompt_column", "prompt"),
        response_column=cfg.get("response_column", "steered_text"),
        evaluator=cfg.get("evaluator", "fine_tuned"),
        batch_size=cfg.get("batch_size", 1),
        num_sentences=cfg.get("num_sentences", None),
        results_dir=cfg.get("results_dir", None),
    )


if __name__ == "__main__":
    main()
