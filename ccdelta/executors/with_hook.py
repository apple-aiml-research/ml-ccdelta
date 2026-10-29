# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Hooked executor for intervention parameter evaluation.
"""

import json
import logging
from pathlib import Path
from typing import List
import hashlib

from ccdelta.hooks import get_hook

# Import type aliases from types module
from .types import ExecutorResult, GenerateFunction, ModelWithHooks, PromptLoader
from .executor_utils import generate_with_prompt_loader

logger = logging.getLogger(__name__)


def hooks_executor(
    model_hooks: ModelWithHooks,
    generate_fn: GenerateFunction,
    prompt_loader: PromptLoader,
    config: dict,
    module_names: List[str],
) -> List[ExecutorResult]:
    """
    Args:
        model_hooks: ModelWithHooks instance for hook management
        generate_fn: Text generation function
        prompt_loader: Iterator/loader for prompts
        config: Configuration dictionary containing hook_params
        module_names: List of module names to apply hooks to

    Returns:
        List of results in format: [config_id, prompt, generation]
    """
    results = []
    hook_params = config.get("hook_params", {})

    # state_path is present, load hooks from folder and run once.
    logger.info("Starting dict sweep with a single configuration from state_path")
    model_hooks.remove_hooks()

    model_hooks.load_hooks_from_folder(
        folder=Path(config["state_path"]),
        module_names=module_names,
        hook_type=config["name"],
        **hook_params,
    )
    model_hooks.register_hooks()

    # Generate with current configuration
    base_config_id = config.get("experiment_id", "config")

    # hash current config_id
    config_hash = str(hashlib.sha256(base_config_id.encode()).hexdigest())[0:8]
    desc = f"Generation {config_hash}: "

    # Generate with current configuration using common helper
    config_results = generate_with_prompt_loader(
        prompt_loader=prompt_loader,
        generate_fn=generate_fn,
        desc=desc,
        param_value=base_config_id,
    )
    results.extend(config_results)

    # Collect compilation error stats before hooks are removed
    compilation_stats = {}
    for name, hook in model_hooks.hooks.items():
        if hasattr(hook, "get_compilation_error_stats"):
            stats = hook.get_compilation_error_stats()
            if stats.get("compiled"):
                compilation_stats[name] = stats
    if compilation_stats:
        state_path = config.get("state_path")
        if state_path:
            stats_path = Path(state_path).parent / "compilation_error_stats.json"
            with open(stats_path, "w") as f:
                json.dump(compilation_stats, f, indent=2)
            logger.info(f"Compilation error stats saved to {stats_path}")
        # Also stash on model_hooks so the caller can access it
        model_hooks._compilation_stats = compilation_stats

    model_hooks.remove_hooks()

    logger.info("Completed hooked execution with %d total generations", len(results))
    return results
