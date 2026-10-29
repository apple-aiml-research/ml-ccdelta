# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from ccdelta.datasets.tabular_dataset import load_tabular_data
from ccdelta.models import get_model
from ccdelta.models.model_with_hooks import ModelWithHooks
from ccdelta.utils.run_pair import run_pair_attack, HookedModelPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate(cfg: DictConfig) -> pd.DataFrame:
    """Run the PAIR attack evaluation.

    Loads the model with hooks and iterates over forbidden prompts,
    running the PAIR attack on each.

    Args:
        cfg: Hydra config with model_params, intervention_params, and pair settings.

    Returns:
        DataFrame with per-prompt PAIR attack results.
    """
    # Load model and tokenizer
    module, tokenizer = get_model(
        model_path=cfg.model_params.model_path,
        cache_dir=cfg.cache_dir,
        dtype=cfg.dtype,
        device=cfg.device,
        model_task="text-generation",
    )

    # Create hooked model
    model_with_hooks = ModelWithHooks(module=module)

    if cfg.intervention_params.state_path is not None:
        model_with_hooks.load_hooks_from_folder(
            folder=Path(cfg.intervention_params.state_path),
            module_names=cfg.model_params.module_names,
            hook_type=cfg.intervention_params.name,
            **cfg.intervention_params.hook_params,
        )
    model_with_hooks.register_hooks()
    logger.info("Hooks attached for PAIR evaluation:")
    for key in model_with_hooks.hooks:
        logger.info(model_with_hooks.hooks[key])

    model = model_with_hooks.module

    # Wrap hooked model as a pipeline
    victim_pipeline = HookedModelPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.get("new_seq_len", 5000),
        device=cfg.device,
    )

    # Load forbidden prompts
    df = load_tabular_data(cfg.data_path)
    prompt_column = cfg.get("prompt_column", "forbidden_prompt")

    if prompt_column not in df.columns:
        raise ValueError(
            f"Prompt column '{prompt_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Limit number of prompts if specified
    num_sentences = cfg.get("num_sentences", None)
    if num_sentences is not None and num_sentences > 0:
        df = df.head(num_sentences)
        logger.info(f"Limited to {num_sentences} prompts")

    # PAIR attack config
    target_str = cfg.get("target_str", "Sure, I'm happy to help")
    attack_models = list(
        cfg.get("attack_models", ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo"])
    )
    judge_models = cfg.get("judge_models", None)
    evaluator = cfg.get("evaluator", "strongreject_rubric")
    max_iter = cfg.get("max_iter", 20)
    max_conversation_length = cfg.get("max_conversation_length", 3)
    max_tokens = cfg.get("max_tokens", 200)

    # Retrieve generation_param for output compatibility with generate_with_hooks
    generation_param = cfg.get("experiment_id", None)
    if generation_param is None:
        # Fallback: try intervention_params.experiment_id
        generation_param = getattr(
            cfg.get("intervention_params", {}), "experiment_id", "pair"
        )

    # Optionally generate baseline (jailbreak="none") rows
    generate_nojailbreak = cfg.get("generate_nojailbreak", True)
    results = []

    if generate_nojailbreak:
        logger.info("Generating baseline (jailbreak='none') responses...")
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Baseline generation"
        ):
            prompt = row[prompt_column]

            messages = [{"role": "user", "content": prompt}]
            output = victim_pipeline(messages)
            generated_text = output[0]["generated_text"]

            # Apply chat template for the prompt column (matches PAIR row format)
            chat_templated_prompt = prompt
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    chat_templated_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass

            results.append(
                {
                    "prompt": chat_templated_prompt,
                    "row_index": idx,
                    "generation_param": generation_param,
                    "steered_text": generated_text,
                    "forbidden_prompt": prompt,
                    "dataset": row.get("dataset", "unknown"),
                    "source": row.get("source", "unknown"),
                    "category": row.get("category", "unknown"),
                    "split": row.get("split", "unknown"),
                    "jailbroken_prompt": prompt,
                    "jailbreak": "none",
                    "jailbreak_found": False,
                    "best_score": float("nan"),
                    "num_iterations": 0,
                }
            )
        logger.info(f"Generated {len(results)} baseline rows")

    # Run PAIR attack for each prompt
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="PAIR attack prompts"):
        prompt = row[prompt_column]
        dataset_name = row.get("dataset", "unknown")
        category = row.get("category", "unknown")

        logger.info(
            f"Running PAIR attack on prompt {idx + 1}/{len(df)}: {prompt[:80]}..."
        )

        attack_result = run_pair_attack(
            forbidden_prompt=prompt,
            victim_pipeline=victim_pipeline,
            target_str=target_str,
            attack_models=attack_models,
            judge_models=judge_models,
            evaluator=evaluator,
            max_iter=max_iter,
            max_conversation_length=max_conversation_length,
            max_tokens=max_tokens,
        )

        jailbroken_prompt = attack_result["best_prompt"]

        # Apply chat template to jailbroken_prompt (same as TabularDataset.__getitem__)
        chat_templated_prompt = jailbroken_prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                chat_templated_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": jailbroken_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        results.append(
            {
                "prompt": chat_templated_prompt,
                "row_index": idx,
                "generation_param": generation_param,
                "steered_text": attack_result["best_response"],
                "forbidden_prompt": prompt,
                "dataset": dataset_name,
                "source": row.get("source", "unknown"),
                "category": category,
                "split": row.get("split", "unknown"),
                "jailbroken_prompt": jailbroken_prompt,
                "jailbreak": "pair",
                # PAIR-specific columns (extra)
                "jailbreak_found": attack_result["jailbreak_found"],
                "best_score": attack_result["best_score"],
                "num_iterations": attack_result["num_iterations"],
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def run_pair_eval(cfg: DictConfig) -> None:
    """Entry point for PAIR evaluation. Runs the attack and saves results.

    Args:
        cfg: Hydra config with pair evaluation settings.
    """
    results = evaluate(cfg)

    # Set up output directory
    results_dir = cfg.get("results_dir", None)
    if results_dir is not None:
        output_path = Path(results_dir) / "generate_with_hooks"
    else:
        output_path = Path("./results") / "generate_with_hooks"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results to parquet
    parquet_file = output_path / "text_generation.parquet"
    results.to_parquet(parquet_file, index=False)
    logger.info(f"Saved PAIR results to {parquet_file}")

    # Compute and save aggregate stats (PAIR rows only)
    pair_results = results[results["jailbreak"] == "pair"]
    total_prompts = len(pair_results)
    attack_success_rate = (
        pair_results["jailbreak_found"].sum() / total_prompts
        if total_prompts > 0
        else 0.0
    )
    mean_best_score = pair_results["best_score"].mean() if total_prompts > 0 else 0.0

    summary = {
        "attack_success_rate": float(attack_success_rate),
        "mean_best_score": float(mean_best_score),
        "total_prompts": total_prompts,
        "config": {
            "target_str": cfg.get("target_str", "Sure, I'm happy to help"),
            "attack_models": list(cfg.get("attack_models", [])),
            "max_iter": cfg.get("max_iter", 20),
            "max_conversation_length": cfg.get("max_conversation_length", 3),
            "max_tokens": cfg.get("max_tokens", 200),
        },
    }

    json_file = output_path / "pair_results.summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved PAIR summary to {json_file}")

    # Log summary
    logger.info("PAIR evaluation completed:")
    logger.info(f"  Total prompts: {total_prompts}")
    logger.info(f"  Attack success rate: {attack_success_rate:.4f}")
    logger.info(f"  Mean best score: {mean_best_score:.2f}")


@hydra.main(config_path="../configs", config_name="text_generation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for standalone execution."""
    run_pair_eval(cfg)


if __name__ == "__main__":
    main()
