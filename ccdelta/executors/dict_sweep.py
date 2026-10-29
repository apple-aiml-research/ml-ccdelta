# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Dictionary parameter sweep executor for intervention parameter evaluation.

This module provides the dict_sweep_executor function that handles complex
parameter structures that can include multiple configurations. For example,
feature_masks can be a list of lists where each inner list represents a
different feature mask configuration to test.
"""

import logging
from pathlib import Path
from typing import List
import hashlib

from ccdelta.hooks import get_hook

# Import type aliases from types module
from .types import ExecutorResult, GenerateFunction, ModelWithHooks, PromptLoader
from .executor_utils import generate_with_prompt_loader

logger = logging.getLogger(__name__)


def dict_sweep_executor(
    model_hooks: ModelWithHooks,
    generate_fn: GenerateFunction,
    prompt_loader: PromptLoader,
    config: dict,
    module_names: List[str],
) -> List[ExecutorResult]:
    """
    Executor for complex dict/object parameters (e.g., feature_masks).

    This executor handles complex parameter structures that can include multiple
    configurations. For example, feature_masks can be a list of lists where each
    inner list represents a different feature mask configuration to test.

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

    if config.get("state_path") is not None:
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
        model_hooks.remove_hooks()

    else:
        # state_path is not present, sweep over feature_masks.
        feature_masks = hook_params.get("feature_masks", [])

        # Determine if we have multiple feature mask configurations
        if (
            feature_masks
            and isinstance(feature_masks, list)
            and len(feature_masks) > 0
            and isinstance(feature_masks[0], list)
        ):
            # Multiple configurations - sweep through each one
            configurations = feature_masks
            logger.info(
                f"Starting dict sweep with {len(configurations)} feature mask configurations"
            )
        else:
            # Single configuration
            configurations = [feature_masks] if feature_masks else [None]
            logger.info("Starting dict sweep with single configuration")

        for config_idx, feature_mask_config in enumerate(configurations):
            model_hooks.remove_hooks()

            # Create hooks directly with dict parameters
            current_hook_params = dict(hook_params)
            if feature_mask_config is not None:
                current_hook_params["feature_masks"] = feature_mask_config

            hooks = []
            for module_name in module_names:
                hook = get_hook(
                    config["name"],
                    module_name=module_name,
                    **current_hook_params,
                )
                hooks.append(hook)
            model_hooks.set_hooks({h.module_name: h for h in hooks})

            model_hooks.register_hooks()

            # Generate with current configuration
            base_config_id = config.get("experiment_id", "config")
            current_config_id = (
                f"{base_config_id}_{config_idx}"
                if len(configurations) > 1
                else base_config_id
            )

            # hash current config_id
            config_hash = str(hashlib.sha256(current_config_id.encode()).hexdigest())[
                0:8
            ]
            desc = f"Generation {config_hash}: "

            # Generate with current configuration using common helper
            config_results = generate_with_prompt_loader(
                prompt_loader=prompt_loader,
                generate_fn=generate_fn,
                desc=desc,
                param_value=current_config_id,
            )
            results.extend(config_results)

            model_hooks.remove_hooks()

    logger.info("Completed dict sweep with %d total generations", len(results))
    return results
