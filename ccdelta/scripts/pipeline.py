# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
This script defines a pipeline for training and evaluating generative models with interventions.
It handles loading configurations, managing responses, and learning interventions on specified modules.

The pipeline leverages Hydra for configuration management and provides both "atonce" and "incr" modes for intervention learning.
"atonce" mode learns interventions on all modules simultaneously, while "incr" mode allows for incremental learning, leveraging previously learned interventions when training on subsequent modules.

"""

import multiprocessing
import sys
import tempfile
import pickle
import os
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

from ccdelta.evaluations import (
    evaluate_eleuther,
    evaluate_PAIR,
    evaluate_perplexity,
    evaluate_strongreject,
    evaluate_fluency,
    evaluate_orbench,
)

from ccdelta.scripts import generate_with_hooks
from ccdelta.scripts.learn_intervention import (
    InterventionsManager,
    learn_intervention,
    ResponsesManager,
)

from ccdelta.utils.logging import get_logger
from ccdelta.utils.utils import log_memory_usage, cleanup_cuda_memory

# Get logger with auto-configured environment variable support
logger = get_logger(__name__)


def run_in_subprocess(target_func, cfg_path):
    """
    Executes a target function in a separate process to ensure memory isolation.
    """
    # We must use the 'spawn' start method for CUDA compatibility
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=target_func, args=(cfg_path,))
    process.start()
    process.join()  # Wait for the subprocess to complete

    if process.exitcode != 0:
        raise RuntimeError(
            f"Subprocess for {target_func.__name__} failed with exit code {process.exitcode}"
        )


def _run_text_generation(config_path):
    log_memory_usage("_run_text_generation start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    # If interventions were learned, pass the path to the evaluation
    if "interventions" in cfg and cfg.interventions.intervention_params.get(
        "learn_intervention", True
    ):
        intervention_state_path = InterventionsManager.get_output_path(
            cfg.interventions
        )
        cfg.text_generation.intervention_params.state_path = intervention_state_path
    if cfg.text_generation.get("fast", False):
        cfg.text_generation.new_seq_len = 10
        cfg.text_generation.num_sentences = 3

    generate_with_hooks.generate(cfg.text_generation)


def _run_mmlu(config_path):
    log_memory_usage("_run_mmlu start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    # If interventions were learned, pass the path to the evaluation
    if "interventions" in cfg and cfg.interventions.intervention_params.get(
        "learn_intervention", True
    ):
        intervention_state_path = InterventionsManager.get_output_path(
            cfg.interventions
        )
        cfg.mmlu.intervention_params.state_path = intervention_state_path

    if cfg.mmlu.get("fast", False):
        cfg.mmlu.limit = 10
        cfg.mmlu.bootstrap_iters = 2
    evaluate_eleuther.run_eleuther_eval(cfg.mmlu, task="mmlu")


def _run_ifeval(config_path):
    log_memory_usage("_run_ifeval start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    # If interventions were learned, pass the path to the evaluation
    if "interventions" in cfg and cfg.interventions.intervention_params.get(
        "learn_intervention", True
    ):
        intervention_state_path = InterventionsManager.get_output_path(
            cfg.interventions
        )
        cfg.mmlu.intervention_params.state_path = intervention_state_path

    if cfg.ifeval.get("fast", False):
        cfg.ifeval.limit = 10
        cfg.ifeval.bootstrap_iters = 2
    evaluate_eleuther.run_eleuther_eval(cfg.ifeval, task="ifeval")

def _run_model_perplexity(config_path):
    log_memory_usage("_run_model_perplexity start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    if cfg.model_perplexity.get("fast", False):
        cfg.model_perplexity.perplexity_model_path = "EleutherAI/pythia-70m"
    evaluate_perplexity.evaluate(cfg.model_perplexity)


def _run_strongreject_eval(config_key, config_path):
    log_memory_usage("_run_strongreject_eval start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    strongreject_cfg = cfg[config_key]
    if strongreject_cfg.get("fast", False):
        strongreject_cfg.batch_size = 2

    # Extract parameters from config and call with explicit parameters
    evaluate_strongreject.run_strongreject_eval(
        data_path=strongreject_cfg.data_path,
        prompt_column=strongreject_cfg.get("prompt_column", "prompt"),
        response_column=strongreject_cfg.get("response_column", "steered_text"),
        evaluator=strongreject_cfg.get("evaluator", "fine_tuned"),
        batch_size=strongreject_cfg.get("batch_size", 1),
        num_sentences=strongreject_cfg.get("num_sentences", None),
        results_dir=strongreject_cfg.get("results_dir", None),
    )


def _run_fluency_eval(config_path):
    log_memory_usage("_run_fluency_eval start")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    fluency_config = cfg.fluency
    if fluency_config.get("fast", False):
        fluency_config.num_sentences = 10

    evaluate_fluency.run_fluency_eval(
        data_path=fluency_config.data_path,
        response_column=fluency_config.get("response_column", "steered_text"),
        max_workers=fluency_config.get("max_workers", 1),
        num_sentences=fluency_config.get("num_sentences", None),
        results_dir=fluency_config.get("results_dir", None),
    )


def _run_orbench_eval(config_path):
    log_memory_usage("_run_orbench_eval start")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    orbench_config = cfg.orbench
    if orbench_config.get("fast", False):
        orbench_config.num_sentences = 10

    evaluate_orbench.run_orbench_eval(
        data_path=orbench_config.data_path,
        prompt_column=orbench_config.get("prompt_column", "prompt"),
        response_column=orbench_config.get("response_column", "steered_text"),
        judge_model=orbench_config.get("judge_model", "gpt-4-turbo-2024-04-09"),
        max_workers=orbench_config.get("max_workers", 16),
        num_sentences=orbench_config.get("num_sentences", None),
        results_dir=orbench_config.get("results_dir", None),
    )


def _run_pair_generation(config_path):
    log_memory_usage("_run_pair_generation start")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    pair_cfg = cfg.pair

    # If interventions were learned, pass the path to the evaluation
    if "interventions" in cfg and cfg.interventions.intervention_params.get(
        "learn_intervention", True
    ):
        intervention_state_path = InterventionsManager.get_output_path(
            cfg.interventions
        )
        pair_cfg.intervention_params.state_path = intervention_state_path

    if pair_cfg.get("fast", False):
        pair_cfg.num_sentences = 2
        pair_cfg.max_iter = 3

    evaluate_PAIR.run_pair_eval(pair_cfg)


EVAL_REGISTRY = {
    "text-generation": _run_text_generation,
    "mmlu": _run_mmlu,
    "ifeval": _run_ifeval,
    "model_perplexity": _run_model_perplexity,
    "strongreject_finetuned": partial(_run_strongreject_eval, "strongreject_finetuned"),
    "strongreject_rubric": partial(_run_strongreject_eval, "strongreject_rubric"),
    "fluency": _run_fluency_eval,
    "orbench": _run_orbench_eval,
    "pair": _run_pair_generation,
    "pair_prompt_defense": _run_pair_generation,
}

EVALS_REQUIRING_OPENAI = {"strongreject_rubric", "fluency", "orbench", "pair", "pair_prompt_defense"}


def _run_intervention_learning_and_responses(config_path):
    log_memory_usage("_run_intervention_learning_and_responses start")
    """Wrapper to run the entire intervention learning phase in a subprocess."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)

    if cfg.get("compute_interventions", True):
        # Learn intervention first, which includes computing responses.
        learn_intervention(cfg)
        # Clean up intervention learning models before evaluations
        cleanup_cuda_memory()
        log_memory_usage("After intervention learning")

    elif cfg.get("compute_responses", False):
        # Only compute responses
        responses_manager = ResponsesManager(cfg.responses)
        responses_manager.compute_responses()
        # Clean up response computation models
        del responses_manager
        cleanup_cuda_memory()
        log_memory_usage("After response computation")


@hydra.main(config_path="../configs", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to run the intervention pipeline."""
    logger.info(OmegaConf.to_yaml(cfg))

    log_memory_usage("main start")

    # Default to True for production, tests will override to False
    use_subprocess = cfg.get("use_subprocess", True)

    # Create a temporary file to pass the config.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(cfg, tmp)
        config_path = tmp.name

    def _run_func(use_subprocess, target_func, config_path):
        if use_subprocess:
            logger.info("Spawning subprocess for target func")
            run_in_subprocess(target_func, config_path)
        else:
            logger.info("Running target func in main process")
            target_func(config_path)

    try:
        # Run the entire intervention learning phase
        if cfg.get("compute_interventions", True) or cfg.get(
            "compute_responses", False
        ):
            _run_func(
                use_subprocess=use_subprocess,
                target_func=_run_intervention_learning_and_responses,
                config_path=config_path,
            )

        for eval_task in cfg.evaluation:
            if eval_task in EVALS_REQUIRING_OPENAI and not os.environ.get("OPENAI_API_KEY"):
                logger.warning(
                    f"Skipping '{eval_task}' evaluation: OPENAI_API_KEY not set"
                )
                continue
            cleanup_cuda_memory()
            target_func = EVAL_REGISTRY[eval_task]
            _run_func(
                use_subprocess=use_subprocess,
                target_func=target_func,
                config_path=config_path,
            )

        logger.info("Pipeline complete")
    finally:
        os.remove(config_path)


if __name__ == "__main__":
    main()
