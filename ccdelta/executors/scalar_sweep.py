# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Scalar parameter sweep executor for intervention parameter evaluation.

This module provides the scalar_sweep_executor function that handles traditional
parameter sweeping approach used by OT methods, where a single scalar parameter
is varied across a range of values using np.linspace.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from ccdelta.hooks import get_hook

# Import type aliases from types module
from .types import ExecutorResult, GenerateFunction, ModelWithHooks, PromptLoader
from .executor_utils import generate_with_prompt_loader

# Type aliases for better understanding of DataLoader collation behavior
CollatedBatch = Dict[str, Union[List[Any], Dict[str, List[Any]], torch.Tensor]]
RowData = Dict[str, Any]

logger = logging.getLogger(__name__)


def scalar_sweep_executor(
    model_hooks: ModelWithHooks,
    generate_fn: GenerateFunction,
    prompt_loader: PromptLoader,
    config: dict,
    module_names: List[str],
) -> List[ExecutorResult]:
    """
    Executor for scalar parameter sweeping (e.g., strength values).

    This executor handles the traditional parameter sweeping approach used by
    OT methods, where a single scalar parameter is varied across a range of
    values using np.linspace.

    Args:
        model_hooks: ModelWithHooks instance for hook management
        generate_fn: Text generation function
        prompt_loader: Iterator/loader for prompts
        config: Configuration dictionary containing executor_params
        module_names: List of module names to apply hooks to

    Returns:
        List of results in format: [param_value, prompt, generation]
    """
    results = []

    # Get sweep parameters from config
    executor_params = config.get("executor_params", {})
    param_name = executor_params.get("param_name", "strength")
    min_value = executor_params.get("min_value", 0)
    max_value = executor_params.get("max_value", 1)
    num_samples = executor_params.get("num_samples", 11)

    logger.info(
        f"Starting scalar sweep of '{param_name}' from {min_value} to {max_value} with {num_samples} samples"
    )

    # Generate parameter values using np.linspace (same as current OT method)
    param_values = np.linspace(min_value, max_value, num_samples)

    for param_value in param_values:
        model_hooks.remove_hooks()

        if config.get("state_path") is not None:
            # Load trained hooks from folder (OT methods)
            hook_params = dict(config.get("hook_params", {}))
            hook_params[param_name] = param_value

            model_hooks.load_hooks_from_folder(
                folder=Path(config["state_path"]),
                module_names=module_names,
                hook_type=config["name"],
                **hook_params,
            )
        else:
            # Create hooks directly (SAE methods can also use scalar sweeping)
            hooks = []
            for module_name in module_names:
                hook_params = dict(config.get("hook_params", {}))
                hook_params[param_name] = param_value

                hook = get_hook(config["name"], module_name=module_name, **hook_params)
                hooks.append(hook)

            model_hooks.set_hooks({h.module_name: h for h in hooks})

        model_hooks.register_hooks()

        # Generate with current parameter value using common helper
        sweep_param_results = generate_with_prompt_loader(
            prompt_loader=prompt_loader,
            generate_fn=generate_fn,
            desc=f"Generation {param_value:0.2f}",
            param_value=param_value,
        )
        results.extend(sweep_param_results)

        model_hooks.remove_hooks()

    logger.info(f"Completed scalar sweep with {len(results)} total generations")
    return results
