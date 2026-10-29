# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Context-Conditioned CAA Hook for dense activation steering with spatial conditioning."""

import logging
import typing as t
from typing import Any, List, Optional

import torch

from .intervention_hook import InterventionHook
from .sae_feature_selection import find_sublist_indices_with_boundary_tolerance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CAAContextConditionedHook(InterventionHook):
    """
    Context-Conditioned CAA (Contrastive Activation Addition) hook.

    This hook combines the spatial conditioning approach from SAE feature selection
    with the dense steering vector approach from CAA. It applies substring matching
    to isolate overlapping regions between source and target sequences before
    computing the mean difference vector.

    Key differences from standard CAA:
    - Uses paired data (source/target) instead of labeled batches
    - Applies spatial conditioning via substring matching
    - Conditions on overlapping token regions before computing differences

    Key differences from SAE:
    - Works on dense activations instead of SAE feature space
    - No statistical feature selection (uses all dimensions)
    - Computes single steering vector instead of per-feature masks

    This hook only supports source_conditioned mode (strict substring matching).
    For baseline CAA without conditioning, use the standard CAAHook instead.

    Args:
        module_name: Name of the module to hook
        device: Device to place tensors on
        intervention_position: When to intervene ("all", "last", "prompt")
        dtype: Data type for tensors
        strength: Multiplier for steering vector
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(
        self,
        module_name: str,
        device: Optional[str] = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        feature_diffing_method: str = "source_conditioned",
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
            **kwargs,
        )
        self.strength = float(strength)
        self.feature_diffing_method = feature_diffing_method

        # Validate feature_diffing_method
        valid_methods = ["source_conditioned"]
        if feature_diffing_method not in valid_methods:
            raise ValueError(
                f"Invalid feature_diffing_method '{feature_diffing_method}'. "
                f"Valid options: {valid_methods}"
            )

        self.register_buffer("vector", torch.empty(0))

    def __str__(self):
        txt = (
            f"CAAContextConditioned("
            f"module_name={self.module_name}, "
            f"strength={self.strength}, "
            f"method={self.feature_diffing_method})"
        )
        return txt

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
    ):
        """Load steering vector from state dict."""
        self.vector = state_dict["vector"].to(self.device).to(self.dtype)
        self._post_load()

    def fit(
        self,
        responses: t.Union[torch.Tensor, List[dict]],
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Learn steering vector from paired data with spatial conditioning.

        Args:
            responses: List of dicts with structure:
                {
                    "source_features": torch.Tensor [source_seq, hidden_dim],
                    "target_features": torch.Tensor [target_seq, hidden_dim],
                    "source_tokens": List[str],
                    "target_tokens": List[str],
                    "row_data": dict (optional) with "original_target_tokens"
                }
            labels: Dummy tensor (ignored for paired learning)
            **kwargs: Additional parameters
        """
        # Validate input is list of dicts (paired data format)
        if not (
            isinstance(responses, list)
            and len(responses) > 0
            and isinstance(responses[0], dict)
        ):
            logger.error(f"Expected a list of dicts, but got {type(responses)}")
            raise ValueError(
                "CAAContextConditionedHook requires paired data (list of dicts). "
                "Did you use the correct response collection hook?"
            )

        logger.info(
            f"Learning steering vector from {len(responses)} pairs "
            f"using method='{self.feature_diffing_method}'"
        )

        # Process each pair with conditioning
        diffs = []
        skipped = 0
        for i, pair in enumerate(responses):
            try:
                diff = self._process_pair(pair)
                if diff is not None:
                    diffs.append(diff)
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Failed to process pair {i}: {e}")
                skipped += 1

        if not diffs:
            raise ValueError(
                f"No valid pairs could be processed. "
                f"Skipped {skipped}/{len(responses)} pairs."
            )

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped}/{len(responses)} pairs during processing"
            )

        # Aggregate into steering vector
        diffs_tensor = torch.stack(diffs)  # [num_pairs, hidden_dim]
        self.vector = diffs_tensor.mean(dim=0).to(self.dtype).to(self.device)

        # Check for NaN/Inf values and replace with zeros if found
        if torch.isnan(self.vector).any() or torch.isinf(self.vector).any():
            logger.warning(
                "NaN or Inf detected in steering vector! Replacing with zeros. "
                "This may indicate issues with the input features."
            )
            self.vector = torch.zeros_like(self.vector)

        # Compute statistics
        if len(diffs) > 1:
            std = diffs_tensor.std(dim=0).mean().item()
        else:
            std = 0.0  # Only one sample, no variance
        norm = self.vector.norm().item()

        logger.info(
            f"Learned steering vector: shape={self.vector.shape}, "
            f"norm={norm:.4f}, mean_std={std:.4f}, "
            f"from {len(diffs)} pairs"
        )

        self._post_load()

    def _process_pair(self, pair: dict) -> Optional[torch.Tensor]:
        """
        Process a single pair with spatial conditioning.

        Args:
            pair: Dictionary containing source/target features and tokens

        Returns:
            Difference vector [hidden_dim] or None if pair should be skipped
        """
        source_tokens = pair["source_tokens"]
        source_features = pair["source_features"]  # [source_seq, hidden_dim]
        target_features = pair["target_features"]  # [target_seq, hidden_dim]

        # Get target tokens for matching
        # Prefer original_target_tokens from row_data for more accurate matching
        if "row_data" in pair and "original_target_tokens" in pair["row_data"]:
            target_tokens = pair["row_data"]["original_target_tokens"]
        else:
            target_tokens = pair["target_tokens"]

        # Find substring match with tolerance for tokenization differences
        start_idx, end_idx = find_sublist_indices_with_boundary_tolerance(
            source_tokens, target_tokens, boundary_tolerance=3
        )
        found_sublist = start_idx is not None and end_idx is not None

        # Apply conditioning based on method
        if self.feature_diffing_method == "source_conditioned":
            # Strict mode: require substring match
            if not found_sublist:
                raise ValueError(
                    f"Source conditioning required but substring match "
                    f"not found for pair {pair.get('id', 'unknown')}. "
                    f"Source tokens ({len(source_tokens)}): "
                    f"{source_tokens[:3]}...{source_tokens[-3:]}. "
                    f"Target tokens ({len(target_tokens)}): "
                    f"{target_tokens[:3]}...{target_tokens[-3:]}."
                )
            conditioned_source = source_features[start_idx : end_idx + 1]
            logger.debug(
                "Pair %s: Conditioned source from [%d:%d] (length %d)",
                pair.get("id", "unknown"),
                start_idx,
                end_idx,
                end_idx - start_idx + 1,
            )
        else:
            raise ValueError(
                f"Unknown feature_diffing_method: {self.feature_diffing_method}"
            )

        # First check if input features already have NaN before pooling
        if torch.isnan(conditioned_source).any():
            logger.warning(
                f"NaN detected in RAW conditioned_source BEFORE pooling for pair {pair.get('id', 'unknown')}. "
                f"Shape: {conditioned_source.shape}, dtype: {conditioned_source.dtype}, "
                f"device: {conditioned_source.device}, "
                f"NaN count: {torch.isnan(conditioned_source).sum().item()}/{conditioned_source.numel()}"
            )
        if torch.isnan(target_features).any():
            logger.warning(
                f"NaN detected in RAW target_features BEFORE pooling for pair {pair.get('id', 'unknown')}. "
                f"Shape: {target_features.shape}, dtype: {target_features.dtype}, "
                f"device: {target_features.device}, "
                f"NaN count: {torch.isnan(target_features).sum().item()}/{target_features.numel()}"
            )

        # Mean pool across sequence dimension
        source_mean = conditioned_source.mean(dim=0)  # [hidden_dim]
        target_mean = target_features.mean(dim=0)  # [hidden_dim]

        # Compute difference: target - source (pointing toward desired behavior)
        diff = target_mean - source_mean  # [hidden_dim]

        return diff

    def forward(self, module, input_, output) -> Any:
        """
        Apply steering vector additively (same as standard CAA).

        Args:
            module: The hooked module
            input: Module input
            output: Module output (activations to steer)

        Returns:
            Steered output: output + strength * vector
        """
        if output.ndim == 4:
            raise DeprecationWarning(
                "InterventionHook already takes care of formatting the input to dims"
            )

        return output + self.strength * self.vector
