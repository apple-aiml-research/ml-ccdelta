# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

# Loads a model and a dataset and extracts intermediate responses
import functools
import logging
import os
import typing as t
from pathlib import Path
import json

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import pipeline, set_seed

from ccdelta.models import get_model
from ccdelta.models.model_with_hooks import ModelWithHooks
from ccdelta.executors import get_executor
from ccdelta.datasets.tabular_dataset import create_tabular_dataloader
from ccdelta.evaluations import evaluate_strongreject
from ccdelta.utils.utils import log_memory_usage, cleanup_cuda_memory


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


# Default sampling kwargs, can be overridden in config
DEFAULT_SAMPLING_KWARGS = dict(
    do_sample=True, temperature=1.0, top_p=0.3, repetition_penalty=1.2
)

# Default kwargs for deterministic generation (do_sample=False)
DEFAULT_DETERMINISTIC_KWARGS = dict(
    do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=None
)


def prepare_tabular_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame for parquet serialization compatibility.

    Parquet supports numpy types, NaN values, and lists natively, so we only need
    to convert dictionaries to JSON strings since parquet can't handle dicts.
    Modifies DataFrame in-place for memory efficiency.

    Args:
        df: DataFrame with potentially mixed types from TabularDataset

    Returns:
        DataFrame with parquet-compatible types (same object, modified in-place)
    """
    for col in df.columns:
        # Convert dictionaries to JSON strings (parquet can't handle dicts)
        # Leave everything else as-is since parquet supports numpy types, lists, etc.
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    return df


def prepare_tabular_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame for JSON serialization compatibility.

    JSON supports dictionaries natively but may have issues with numpy types and NaN.
    Convert numpy types to Python native types and handle NaN values.
    Modifies DataFrame in-place for memory efficiency.

    Args:
        df: DataFrame with potentially mixed types from TabularDataset

    Returns:
        DataFrame with JSON-compatible types (same object, modified in-place)
    """
    import numpy as np

    for col in df.columns:
        # Convert numpy types to Python native types for JSON compatibility
        df[col] = df[col].apply(
            lambda x: (
                x.item() if isinstance(x, (np.integer, np.floating, np.bool_)) else x
            )
        )

        # Convert numpy arrays to lists for JSON compatibility
        df[col] = df[col].apply(
            lambda x: (x.tolist() if isinstance(x, np.ndarray) else x)
        )

        # Handle NaN values (JSON doesn't support NaN)
        # Skip lists and arrays to avoid "ambiguous truth value" error
        def safe_nan_check(x):
            if isinstance(x, (list, np.ndarray)):
                return x  # Lists and arrays are never NaN
            return None if pd.isna(x) else x

        df[col] = df[col].apply(safe_nan_check)

    return df


def print_generated_sentences(output: t.List[t.Dict[str, str]]) -> None:
    for o in output:
        print("-" * 80)
        print(o[0]["generated_text"])


def get_chat_templated_generate_fn(generator, tokenizer):
    """
    Returns a wrapped generate function that applies a chat template to prompts.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        # If the tokenizer doesn't have a chat template, return the original generator
        logger.warning("Tokenizer does not have a chat_template. Using raw prompts.")
        return generator

    @functools.wraps(generator)
    def templated_generator(prompts: list[str], **kwargs):
        # 1. Apply the chat template to each prompt in the batch
        templated_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        # 2. Call the original generator with the formatted prompts
        return generator(templated_prompts, **kwargs)

    return templated_generator


def generate(cfg: DictConfig) -> None:
    """Generates text using a pretrained language model with optional interventions.

    This function generates text using a pretrained language model specified in the
    configuration (`cfg`). It allows for applying interventions during the generation
    process, controlled by parameters defined in `cfg.intervention_params`. The generated
    text is then logged and optionally saved to a CSV file.

    Args:
        cfg (DictConfig): A Hydra configuration object containing all necessary
            parameters for text generation and intervention. See the
            `configs/text_generation.yaml` example for details.

    Raises:
        ValueError: If `cfg.prompt` ends with ".txt" but the file does not exist.

    Returns:
        None
    """
    # Handle optional verbose setting
    verbose = getattr(cfg, "verbose", cfg.get("text_generation", {}).get("verbose", 1))
    if verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)

    if cfg.results_dir is not None:
        output_path = Path(cfg.results_dir) / Path(__file__).stem
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    logger.info("Output path=%s", str(output_path))

    # Use cfg.model_params first, then fall back to cfg.model if it doesn't exist
    model_config = getattr(cfg, "model_params", getattr(cfg, "model", None))
    if model_config is None:
        raise ValueError("Neither 'model_params' nor 'model' found in configuration")
    # MPS workaround: Load on CPU first, then move to device to avoid memory errors
    final_device = cfg.device
    load_device = "cpu" if final_device == "mps" else final_device
    model, tokenizer = get_model(
        cache_dir=cfg.cache_dir,
        device=load_device,
        model_task="text-generation",
        **model_config,
    )
    if final_device != load_device:
        model.to(final_device)

    # Create hooked model
    model_hooks = ModelWithHooks(
        module=model,
    )

    # Log memory usage after model loading
    log_memory_usage("After model loading")

    results = []

    text_gen_cfg = cfg.get("text_generation", cfg)

    # Get sampling kwargs from config, with appropriate defaults
    sampling_kwargs_cfg = text_gen_cfg.get("sampling_kwargs", DEFAULT_SAMPLING_KWARGS)
    if OmegaConf.is_config(sampling_kwargs_cfg):
        sampling_kwargs = OmegaConf.to_container(sampling_kwargs_cfg, resolve=True)
    else:
        sampling_kwargs = sampling_kwargs_cfg

    # Ensure sampling_kwargs is a dict, and use appropriate defaults based on do_sample setting
    if not isinstance(sampling_kwargs, dict):
        sampling_kwargs = DEFAULT_SAMPLING_KWARGS

    # If do_sample is False, use deterministic defaults to avoid warnings
    if not sampling_kwargs.get("do_sample", True):
        # Start with deterministic defaults and overlay any explicit config
        final_kwargs = DEFAULT_DETERMINISTIC_KWARGS.copy()
        final_kwargs.update(sampling_kwargs)
        sampling_kwargs = final_kwargs
        logger.info(f"Using deterministic sampling kwargs: {sampling_kwargs}")
    else:
        logger.info(f"Using sampling kwargs: {sampling_kwargs}")

    # Set up prompt loader
    set_seed(cfg.seed)
    batch_size = text_gen_cfg.get("batch_size", 1)

    # Set up text generation pipeline with batching enabled
    generator = pipeline(
        "text-generation",
        model=model_hooks.module,
        tokenizer=tokenizer,
        batch_size=batch_size,  # Enable native pipeline batching
    )

    # Manually set the device for the pipeline if using MPS.
    # Note that with mps it is better to move the model onto the device
    # after loading onto CPU.
    if final_device == "mps":
        generator.device = torch.device("mps")

    generate_fn = functools.partial(
        generator,
        max_new_tokens=text_gen_cfg.get("new_seq_len", 50),
        return_full_text=False,  # Return only the generated text, not the prompt
        **sampling_kwargs,
    )

    prompt = text_gen_cfg.get("prompt", "Once upon a time")

    # Handle both single file paths and lists of file paths (including OmegaConf ListConfig)
    if OmegaConf.is_list(prompt) or isinstance(prompt, list):
        prompt = (
            OmegaConf.to_container(prompt) if OmegaConf.is_config(prompt) else prompt
        )

    if text_gen_cfg.get("tabular_mode", False) or (
        isinstance(prompt, list)
        and prompt
        and prompt[0].endswith((".parquet", ".json", ".csv"))
    ) or (
        isinstance(prompt, str)
        and prompt.endswith((".parquet", ".json", ".csv"))
    ):
        # For tabular data files - handles both single files and lists
        prompt_column = text_gen_cfg.get("prompt_column", "text")
        output_column = text_gen_cfg.get("output_column", "steered_text")
        query = text_gen_cfg.get("query", None)
        use_chat_template = text_gen_cfg.get("use_chat_template", True)

        max_rows = text_gen_cfg.get("num_sentences", None)
        if text_gen_cfg.get("fast", False):
            max_rows = 100

        prompt_loader = create_tabular_dataloader(
            prompt,  # Pass prompt as-is (supports both str and list[str])
            prompt_column=prompt_column,
            output_column=output_column,
            batch_size=batch_size,
            max_rows=max_rows,
            query=query,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
    else:
        raise ValueError(
            f"Unsupported prompt format: {prompt!r}. "
            "Provide a path to a .parquet, .json, or .csv file."
        )

    executor_name = cfg.intervention_params.get("executor", "with_hooks")
    executor = get_executor(executor_name)

    intervention_config = cfg.intervention_params

    # Convert OmegaConf objects to plain Python objects
    # This is needed for complex nested structures like feature_masks
    if OmegaConf.is_config(intervention_config):
        intervention_config = OmegaConf.to_object(intervention_config)

    results = executor(
        model_hooks=model_hooks,
        generate_fn=generate_fn,
        prompt_loader=prompt_loader,
        config=intervention_config,
        module_names=model_config.module_names,
    )

    # Log memory usage after text generation
    log_memory_usage("After text generation")

    # Save compilation error stats to output directory if available
    compilation_stats = getattr(model_hooks, "_compilation_stats", None)
    if compilation_stats and output_path:
        stats_path = output_path / "compilation_error_stats.json"
        with open(stats_path, "w") as f:
            json.dump(compilation_stats, f, indent=2)
        logger.info(f"Compilation error stats saved to {stats_path}")

    # Display sample results
    if results:
        print("\n")
        logger.info("Sample generated results:")
        logger.info("=" * 50)
        for i, result in enumerate(results[:10]):  # Show first 3 results
            param_info = result.get("generation_param", "N/A")
            prompt_text = result.get("prompt", "N/A")
            output_column = getattr(
                prompt_loader.dataset, "output_column", "steered_text"
            )
            generation = result.get(output_column, "N/A")
            logger.info(
                f"Result {i+1}: {param_info} | {prompt_text[:30]}... | {generation[:50]}..."
            )

    saved_file_path = None
    # Save enhanced results as DataFrame
    if output_path is not None and results:
        df = pd.DataFrame(results)

        # Check if we have tabular data to determine output format
        data_type = getattr(prompt_loader.dataset, "data_type", "text")

        if data_type == "tabular":
            # Match output format to input format for tabular data
            # Use the first file for format detection if prompt is a list
            first_file = prompt[0] if isinstance(prompt, list) else prompt
            if first_file.endswith(".parquet"):
                # Input was parquet -> output to parquet
                prepare_tabular_for_parquet(df)
                saved_file_path = output_path / "text_generation.parquet"
                df.to_parquet(saved_file_path)
            elif first_file.endswith(".json"):
                # Input was JSON -> output to JSON
                prepare_tabular_for_json(df)
                saved_file_path = output_path / "text_generation.json"
                df.to_json(saved_file_path, orient="records", indent=2)
            else:
                # Default to parquet for CSV or other tabular formats
                prepare_tabular_for_parquet(df)
                saved_file_path = output_path / "text_generation.parquet"
                df.to_parquet(saved_file_path)
        else:
            saved_file_path = output_path / "text_generation.csv"
            df.to_csv(saved_file_path)

        if saved_file_path:
            logger.info(f"Results saved to {saved_file_path}")

    # Unload the generator model from memory before we start evals.
    logger.info("Unloading models from memory before running evaluations...")
    # Delete all references to the model and tokenizer
    del generator
    del model_hooks
    del model
    del tokenizer
    del prompt_loader
    del results
    del executor

    # Clean up CUDA memory
    cleanup_cuda_memory()
    logger.info("Model cleanup completed")

    # Log memory usage after cleanup
    log_memory_usage("After model cleanup")

    # Run integrated scoring if configured and results were saved
    if saved_file_path:
        _score_generated_text(cfg, saved_file_path, output_path)


def _score_generated_text(
    cfg: DictConfig, saved_file_path: Path, output_path: Path
) -> None:
    """Run integrated scoring evaluations on generated text results.

    Args:
        cfg: Configuration containing text_generation.scoring settings
        saved_file_path: Path to the saved text generation results file
        output_path: Base output directory for results
    """
    text_gen_cfg = cfg.get("text_generation", cfg)
    scoring_cfg = text_gen_cfg.get("scoring", {})

    if not scoring_cfg:
        logger.info("No scoring configuration found, skipping integrated scoring")
        return

    logger.info("Running integrated scoring evaluations...")

    # Run StrongReject evaluations - handle both single and multiple configurations
    for config_key, config_value in scoring_cfg.items():
        if config_key.startswith("strongreject"):
            logger.info(f"Running integrated StrongReject evaluation: {config_key}")
            try:
                # Extract parameters from config and call with explicit parameters
                evaluate_strongreject.run_strongreject_eval(
                    data_path=str(saved_file_path),
                    prompt_column=config_value.get("prompt_column", "prompt"),
                    response_column=config_value.get("response_column", "steered_text"),
                    evaluator=config_value.get("evaluator", "fine_tuned"),
                    batch_size=config_value.get("batch_size", 1),
                    num_sentences=None,  # Process all rows for integrated scoring
                    results_dir=str(output_path.parent),
                )

                logger.info(
                    f"Integrated StrongReject evaluation ({config_key}) completed successfully"
                )

            except Exception as e:
                logger.error(
                    f"Integrated StrongReject evaluation ({config_key}) failed: {e}"
                )
                # Don't fail the entire generation process due to scoring errors

    # Clean up evaluation models and memory
    logger.info("Cleaning up evaluation models...")
    cleanup_cuda_memory()
    logger.info("Evaluation cleanup completed")

    # Log memory usage after evaluation cleanup
    log_memory_usage("After evaluation cleanup")


@hydra.main(config_path="../configs", config_name="text_generation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
