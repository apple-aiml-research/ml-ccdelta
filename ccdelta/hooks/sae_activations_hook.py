# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Minimal SAE Activations Hook for computing SAE features without saving."""

import typing as t
from typing import Optional, Any

import torch

from ccdelta.utils.logging import get_logger
from ccdelta.utils.sae import load_sae_model_for_hook

logger = get_logger(__name__)



class SAEActivationsHook:
    """Minimal hook that just computes SAE activations without saving.

    This hook loads an SAE model and computes activations, designed to be used
    as an inner hook within PairedResponsesHook or other composite hooks.
    """

    def __init__(
        self,
        module_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        sae_config: dict[str, Any] = None,
        return_sparse: bool = False,
        **kwargs,
    ):
        """Initialize SAEActivationsHook.

        Args:
            module_name: Name of the module this hook is attached to
            device: Device to load SAE model on
            dtype: Data type for SAE model
            sae_config: sae config
            return_sparse: If True, return features as a sparse tensor.
            **kwargs: Additional arguments (ignored)
        """
        self.module_name = module_name
        self.device = device
        self.dtype = dtype
        self.return_sparse = return_sparse
        self.cache_dir = kwargs.get("cache_dir")

        # State for batch processing
        self.batch: t.Optional[dict] = None

        # Initialize SAE Model
        self.sae_config = sae_config
        self.sae_model = None
        if self.sae_config is not None:
            self.sae_model = load_sae_model_for_hook(
                config=self.sae_config,
                device=self.device,
                dtype=self.dtype,
                cache_dir=self.cache_dir,
            )
        logger.info(f"Initialized SAEActivationsHook for {module_name}")

    def update(self, batch_idx: int, batch: dict):
        """Update hook state with batch information."""
        self.batch = batch

    def __call__(self, module, input, output):
        """Compute SAE activations from module output."""
        logger.debug(
            f"Called with output type: {type(output)}, shape: {output.shape if hasattr(output, 'shape') else 'N/A'}"
        )
        if self.sae_model is None:
            logger.warning("SAE model not loaded, falling back to raw activations")
            return output

        # Ensure output is on the same device as SAE model
        output = output.to(
            device=self.sae_model.W_enc.device, dtype=self.sae_model.W_enc.dtype
        )
        original_shape = output.shape
        # Apply SAE encoding to all positions
        sae_features = self.sae_model.encode(output)

        # Conditionally convert to a sparse tensor
        if self.return_sparse:
            sae_features = sae_features.to_sparse()
            logger.debug("Converted SAE features to sparse format")

        sae_features = sae_features.to(dtype=output.dtype)

        logger.debug(f"Returning SAE features with shape: {sae_features.shape}")
        return sae_features

    def __str__(self):
        return f"SAEActivationsHook(module_name={self.module_name}, sae_config={self.sae_config})"
