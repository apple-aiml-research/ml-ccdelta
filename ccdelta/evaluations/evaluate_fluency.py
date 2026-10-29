# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import click
import logging
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel

from ccdelta.datasets.tabular_dataset import load_tabular_data
from ccdelta.evaluations.eval_utils import compute_aggregate_stats, save_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FluencyScore(BaseModel):
    rationale: str
    score: str


# from https://arxiv.org/abs/2407.07087
COPYBENCH_EVAL_TEMPLATE = """
You are a fair judge assistant tasked with providing clear, objective rationale based on specific criteria, ensuring each
assessment reflects the absolute standards set for performance.

###Task Description:
A response text to evaluate, and a score rubric representing evaluation criteria are given.
The response may be a partial one so it is okay if it ends abruptly you are only looking at the text that is present.
1. Write a detailed rationale that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a rationale, write a score that is an integer between 1 and 5. You should refer to the score rubric.

###Response to evaluate:
<start_of_response>
{response}
<end_of_response>

###Score Rubrics:
Evaluate whether the given text is well-written and grammatical.
Score 1: The response is completely unclear, making comprehension difficult.
Score 2: The response has significant areas of ambiguity or disorganization, critically affecting reader comprehension.
Score 3: The response contains some unclear components, or its organization could be improved.
Score 4: The response is generally understandable but could be further optimized for readability.
Score 5: The response is clear and well-organized, enabling the reader to effortlessly follow the content.
"""


# Template mapping
EVAL_TEMPLATES = {
    "copybench": COPYBENCH_EVAL_TEMPLATE,
}


def get_fluency(text, client, template="copybench", model="gpt-4o"):
    if template not in EVAL_TEMPLATES:
        raise ValueError(
            f"Unknown template '{template}'. Available: {list(EVAL_TEMPLATES.keys())}"
        )

    score_prompt = EVAL_TEMPLATES[template].format(response=text)
    client_resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": score_prompt}],
        text_format=FluencyScore,
    )

    score = client_resp.output_parsed
    return score


def _evaluate_single_sample(args: tuple) -> dict | None:
    """Helper function to evaluate a single sample - needed for multiprocessing."""
    i, response, client, template, model = args
    try:
        eval_result: FluencyScore = get_fluency(
            text=response, client=client, template=template, model=model
        )
        if eval_result is not None:
            extracted_result = {
                "score": float(eval_result.score),
                "rationale": eval_result.rationale,
            }
            logger.info("Fluency response extracted result: %s", str(extracted_result))
            return extracted_result
        else:
            return None
    except Exception as e:
        logger.debug(f"Sample {i} failed evaluation: {e}")
        return None


def _evaluate_samples(
    df_clean: pd.DataFrame,
    response_column: str,
    client: OpenAI,
    max_workers: int,
    template: str = "copybench",
    model: str = "gpt-4o",
) -> list[dict | None]:
    """
    Evaluate samples using multiprocessing for speed.

    Args:
        df_clean: Clean DataFrame with samples to evaluate
        response_column: Name of the response column
        max_workers: Number of parallel workers (default: 4 to avoid rate limits)

    Returns:
        List of dictionaries with evaluation results, None for completely failed samples
    """
    # Prepare arguments for parallel processing
    sample_args = []
    for i, (_, row) in enumerate(df_clean.iterrows()):
        sample_args.append((i, row[response_column], client, template, model))

    results: list[dict | None] = []
    for _ in range(len(sample_args)):
        results.append(None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_evaluate_single_sample, args): i
            for i, args in enumerate(sample_args)
        }

        for future in tqdm(
            as_completed(future_to_index),
            total=len(sample_args),
            desc="Evaluating samples",
        ):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.debug(f"Sample {index} failed evaluation: {e}")
                results[index] = None

    return results


def run_fluency_eval(
    data_path: str,
    response_column: str = "steered_text",
    num_sentences: int | None = None,
    max_workers: int = 4,
    results_dir: str | None = None,
    template: str = "copybench",
    model: str = "gpt-4o",
):
    client = OpenAI()

    score_column = "fluency_score"
    rationale_column = "fluency_rationale"

    logger.info(f"Loading data from {data_path}")
    df = load_tabular_data(data_path)

    if response_column not in df.columns:
        raise ValueError(
            f"Response column '{response_column}' not found in data. Available columns: {list(df.columns)}"
        )

    # Filter out rows with missing data
    logger.info(f"Original data has {len(df)} rows")
    df_clean = df.dropna(subset=[response_column]).copy()
    logger.info(f"After removing missing data: {len(df_clean)} rows")

    if len(df_clean) == 0:
        logger.warning("No valid data to evaluate")
        df[score_column] = float("nan")
        return df

    # Limit number of sentences if specified
    if num_sentences is not None and num_sentences > 0:
        df_clean = df_clean.head(num_sentences)
        logger.info(f"Limited to {num_sentences} sentences: {len(df_clean)} rows")

    df[score_column] = float("nan")
    clean_indices = df_clean.index.tolist()

    results = _evaluate_samples(
        df_clean,
        response_column,
        client=client,
        max_workers=max_workers,
        template=template,
        model=model,
    )

    for idx, result in zip(clean_indices, results):
        if result is not None:
            # Extract score (may be None if parsing failed)
            score = result.get("score")
            rationale = result.get("rationale")
            if score is not None and not pd.isna(score):
                df.loc[idx, score_column] = score
                df.loc[idx, rationale_column] = rationale

    # Set up output directory
    if results_dir is not None:
        output_path = Path(results_dir) / Path(__file__).stem
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("./results") / Path(__file__).stem
        output_path.mkdir(parents=True, exist_ok=True)

    # Extract filename from input path for statistics
    if "/" in data_path:
        input_filename = data_path.split("/")[-1]
    else:
        input_filename = data_path

    # Compute aggregate statistics
    aggregate_stats = compute_aggregate_stats(df, score_column, input_filename)

    # Save results with JSON summary
    save_results(df, output_path, data_path, aggregate_stats, "fluency")

    logger.info(f"Fluency evaluation completed. Results saved to {output_path}")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, readable=True))
@click.option("--response-column", default="steered_text")
@click.option("--num-sentences", type=int, help="sentences to evaluate (default: all)")
@click.option("--max-workers", type=int, default=4)
@click.option("--results-dir", help="Directory to save results (default: ./results)")
@click.option("--template", type=click.Choice(["copybench"]), default="copybench")
@click.option("--model", default="gpt-4o")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    data_path: str,
    response_column: str,
    num_sentences: int | None,
    max_workers: int,
    results_dir: str | None,
    template: str,
    model: str,
    verbose: bool,
) -> None:
    """Evaluate fluency of text in a data file.

    DATA_PATH: Path to the input data file (CSV, JSON, or Parquet)
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Starting fluency evaluation on {data_path}")

    run_fluency_eval(
        data_path=data_path,
        response_column=response_column,
        num_sentences=num_sentences,
        max_workers=max_workers,
        results_dir=results_dir,
        template=template,
        model=model,
    )


if __name__ == "__main__":
    main()
