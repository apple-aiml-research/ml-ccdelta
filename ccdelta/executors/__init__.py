# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Executor framework for intervention parameter sweeping and evaluation.

This module provides a generic executor system that handles different types of
execution strategies for intervention methods.

The executor system is designed to work with the existing ModelWithHooks
infrastructure without requiring changes to the hook loading mechanisms.
"""

from .no_hooks import no_hooks_executor
from .with_hook import hooks_executor
from .types import ExecutorFunction

# Executor registry - maps executor names to executor functions
EXECUTOR_REGISTRY: dict[str, ExecutorFunction] = {
    "no_hooks": no_hooks_executor,
    "with_hooks": hooks_executor,
}


def get_executor(name: str) -> ExecutorFunction:
    """
    Get executor function by name.

    Args:
        name: Name of the executor (e.g., 'no_hooks', 'hooks')

    Returns:
        Executor function matching the ExecutorFunction protocol

    Raises:
        ValueError: If executor name is not found in registry
    """
    if name not in EXECUTOR_REGISTRY:
        raise ValueError(
            f"Unknown executor: {name}. Available executors: {list(EXECUTOR_REGISTRY.keys())}"
        )
    return EXECUTOR_REGISTRY[name]


def list_available_executors() -> list[str]:
    """
    List all available executor names.

    Returns:
        List of executor names
    """
    return list(EXECUTOR_REGISTRY.keys())
