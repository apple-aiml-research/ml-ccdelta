# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
No-hooks executor for baseline evaluation without any hook registration.
"""

import logging
from typing import Any

from ccdelta.models.model_with_hooks import ModelWithHooks
from .types import ExecutorFunction, ExecutorResult, GenerateFunction, PromptLoader
from .executor_utils import generate_with_prompt_loader

logger = logging.getLogger(__name__)


def no_hooks_executor(
    model_hooks: ModelWithHooks,
    generate_fn: GenerateFunction,
    prompt_loader: PromptLoader,
    config: dict[str, Any],
    module_names: list[str],
) -> list[ExecutorResult]:
    """
    Execute text generation without registering any hooks.

    Args:
        model_hooks: ModelWithHooks instance (hooks will not be used)
        generate_fn: Text generation function
        prompt_loader: Iterator yielding batches of prompts
        config: Configuration dictionary (ignored for no-hooks)
        module_names: List of module names (ignored for no-hooks)

    Returns:
        List of results with param_info="no_hooks" for all generations
    """
    logger.info("Running no-hooks baseline generation (no hooks registered)")

    # Ensure no hooks are registered
    model_hooks.remove_hooks()

    # Use experiment_id if provided, otherwise fall back to "no_hooks"
    param_value = config.get("experiment_id", "no_hooks")

    # Use common generation helper
    results = generate_with_prompt_loader(
        prompt_loader=prompt_loader,
        generate_fn=generate_fn,
        desc=f"Generation no_hooks: {param_value}",
        param_value=param_value,
    )

    logger.info(f"No-hooks baseline generation completed: {len(results)} results")
    return results
