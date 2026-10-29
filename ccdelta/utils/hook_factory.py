# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Centralized hook creation utilities with SAE config resolution.

This module provides utilities for creating hooks with proper SAE configuration
resolution, supporting both single-SAE (backward compatible) and multi-layer SAE
configurations while leaving non-SAE hooks completely unaffected.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf

from ccdelta.hooks import get_hook
from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)

# SAE hook types that need config resolution
SAE_HOOK_TYPES = {
    "sae_feature_space",
    "sae_responses",
    "paired_sae_responses",
    "paired_sae_error_analysis",
}


def _get_nested_sae_config(sae_configs: DictConfig, module_name: str) -> Optional[Dict]:
    """
    Navigate nested config structure created by Hydra CLI overrides.

    When using CLI overrides like `+model.sae_configs.model.layers.20.sae_id=value`,
    Hydra creates a nested structure instead of a key with dots. This function
    navigates that nested structure to find the SAE config.

    Args:
        sae_configs: The sae_configs DictConfig
        module_name: Module name like "model.layers.20"

    Returns:
        SAE config dict if found via nested path, None otherwise.
    """
    try:
        # Navigate the nested structure: model.layers.20 -> ["model", "layers", "20"]
        parts = module_name.split(".")
        current = sae_configs
        for part in parts:
            if OmegaConf.is_config(current) and part in current:
                current = current[part]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        # Check if we found a valid SAE config (should have sae_id or release)
        if current is not None:
            current_dict = (
                OmegaConf.to_object(current)
                if OmegaConf.is_config(current)
                else dict(current) if isinstance(current, dict) else None
            )
            if current_dict and ("sae_id" in current_dict or "release" in current_dict):
                return current_dict
    except Exception:
        pass
    return None


def is_sae_hook(hook_name: str) -> bool:
    return hook_name in SAE_HOOK_TYPES


def resolve_sae_config_for_module(
    module_name: str, model_config: DictConfig
) -> Optional[Dict[str, Any]]:
    """
    Resolve SAE config for a specific module name.

    Supports both configuration formats:
    - New format: model_config.sae_configs[module_name] (per-module mapping)
    - Old format: model_config.sae (shared across all modules)

    Args:
        module_name: Module name like "model.layers.16"
        model_config: Model configuration from configs/model/*.yaml

    Returns:
        SAE config dict for the module, or None if no SAE configured.
        None is normal for non-SAE models and should not cause errors.

    Examples:
        # New format (multi-layer)
        >>> config = OmegaConf.create({
        ...     'sae_configs': {
        ...         'model.layers.16': {'sae_id': 'l16r_32x'},
        ...         'model.layers.20': {'sae_id': 'l20r_32x'}
        ...     }
        ... })
        >>> resolve_sae_config_for_module('model.layers.16', config)
        {'sae_id': 'l16r_32x'}

        # Old format (single SAE)
        >>> config = OmegaConf.create({'sae': {'sae_id': 'l16r_32x'}})
        >>> resolve_sae_config_for_module('model.layers.16', config)
        {'sae_id': 'l16r_32x'}

        # No SAE config (non-SAE model)
        >>> config = OmegaConf.create({'module_names': ['model.layers.16']})
        >>> resolve_sae_config_for_module('model.layers.16', config)
        None
    """
    # New format: per-module mapping (takes precedence)
    if hasattr(model_config, "sae_configs") and model_config.sae_configs:
        # Get base config from quoted key (e.g., sae_configs["model.layers.20"])
        base_config = model_config.sae_configs.get(module_name)

        # Check for nested override from CLI (e.g., sae_configs.model.layers.20)
        # Hydra CLI overrides create nested structure instead of modifying quoted key
        nested_override = _get_nested_sae_config(model_config.sae_configs, module_name)

        if nested_override:
            if base_config is not None:
                # Merge: start with base, override with nested values
                base_dict = (
                    OmegaConf.to_object(base_config)
                    if OmegaConf.is_config(base_config)
                    else dict(base_config)
                )
                base_dict.update(nested_override)
                logger.debug(
                    f"Using merged SAE config for {module_name} "
                    f"(base + CLI overrides: {list(nested_override.keys())})"
                )
                return base_dict
            else:
                # Only nested override exists (no base config)
                logger.debug(f"Using CLI override SAE config for {module_name}")
                return nested_override

        if base_config is not None:
            logger.debug(f"Using per-module SAE config for {module_name}")
            # Convert OmegaConf to dict if needed
            return (
                OmegaConf.to_object(base_config)
                if OmegaConf.is_config(base_config)
                else base_config
            )

    # Old format: shared SAE (backward compatibility)
    if hasattr(model_config, "sae") and model_config.sae:
        logger.debug(f"Using shared SAE config for {module_name} (backward compat)")
        sae_config = model_config.sae
        return (
            OmegaConf.to_object(sae_config)
            if OmegaConf.is_config(sae_config)
            else sae_config
        )

    # No SAE config - this is normal for non-SAE models
    logger.debug(
        f"No SAE config found for {module_name} (non-SAE model or missing config)"
    )
    return None


def _create_hook_with_sae_config(
    hook_name: str,
    module_name: str,
    base_hook_params: Dict[str, Any],
    **extra_kwargs,
):
    """
    Create a hook with proper SAE config resolution.
    """
    # Sanity check: This function should only be called for SAE hooks
    if not is_sae_hook(hook_name):
        logger.warning(
            f"create_hook_with_sae_config() called for non-SAE hook '{hook_name}'. "
            f"Non-SAE hooks should use hooks.get_hook() directly."
        )

    # Extract model_config from hook_params if present
    hook_params = dict(base_hook_params)
    model_config = hook_params.pop("model_config", None)  # Remove after extraction

    if model_config is not None:
        logger.debug(f"Extracted model_config from hook_params for {module_name}")

    # Resolve SAE config for this module (if model_config available)
    sae_config = None
    if model_config is not None:
        sae_config = resolve_sae_config_for_module(module_name, model_config)

    # Prepare hook params with module-specific SAE config
    if sae_config is not None:
        hook_params["sae_config"] = sae_config
        sae_id = sae_config.get("sae_id", sae_config.get("release", "custom"))
        logger.info(f"Creating {hook_name} hook for {module_name} with SAE: {sae_id}")
    else:
        # For SAE hooks, missing SAE config might be OK during loading (config in state dict)
        if is_sae_hook(hook_name) and model_config is None:
            logger.debug(
                f"Creating SAE hook '{hook_name}' for {module_name} without model_config in hook_params"
            )
        logger.info(
            f"Creating {hook_name} hook for {module_name} (SAE config will be loaded from state dict if present)"
        )

    # Create hook with resolved config
    return get_hook(hook_name, module_name=module_name, **hook_params, **extra_kwargs)


def create_or_load_hook(
    hook_name: str,
    module_name: str,
    base_hook_params: Dict[str, Any],
    state_path: Optional[Path] = None,
    **extra_kwargs,
):
    """
    Universal hook creation/loading function with automatic SAE/non-SAE handling.

    This function provides a single entry point for all hook creation and loading,
    automatically handling the differences between SAE and non-SAE hooks.

    - For SAE hooks: Uses hook_factory with SAE config resolution
    - For non-SAE hooks: Uses hooks.get_hook() directly (backward compat)
    - Supports both creation (state_path=None) and loading (state_path provided)

    Args:
        hook_name: Hook type (e.g., "sae_feature_space", "mean_ot")
        module_name: Module to hook (e.g., "model.layers.16")
        base_hook_params: Base hook parameters (may contain model_config for SAE hooks)
        state_path: Optional path to .statedict file. If provided, loads existing hook.
                   If None, creates new hook.
        **extra_kwargs: Additional arguments passed to hook creation

    Returns:
        Hook instance (newly created or loaded from disk)

    Examples:
        >>> # Create new SAE hook
        >>> hook = create_or_load_hook(
        ...     hook_name="sae_feature_space",
        ...     module_name="model.layers.16",
        ...     base_hook_params={'device': 'cuda', 'model_config': cfg.model}
        ... )
        >>>
        >>> # Load existing SAE hook
        >>> hook = create_or_load_hook(
        ...     hook_name="sae_feature_space",
        ...     module_name="model.layers.16",
        ...     base_hook_params={'device': 'cuda', 'model_config': cfg.model},
        ...     state_path=Path("interventions/model.layers.16.statedict")
        ... )
        >>>
        >>> # Create/load non-SAE hook (works the same way)
        >>> hook = create_or_load_hook(
        ...     hook_name="mean_ot",
        ...     module_name="model.layers.16",
        ...     base_hook_params={'device': 'cuda'},
        ...     state_path=some_path  # optional
        ... )
    """
    if is_sae_hook(hook_name):
        # SAE hooks: Use hook factory with config resolution
        hook = _create_hook_with_sae_config(
            hook_name=hook_name,
            module_name=module_name,
            base_hook_params=base_hook_params,
            **extra_kwargs,
        )
        if state_path is not None:
            hook.from_state_path(state_path)
            logger.info(f"Loaded hook state from {state_path}")
        return hook
    else:
        # Non-SAE hooks: Use hooks.get_hook() directly (backward compat)
        hook = get_hook(
            hook_name,
            module_name=module_name,
            **base_hook_params,
            **extra_kwargs,
        )
        if state_path is not None:
            hook.from_state_path(state_path)
        return hook
