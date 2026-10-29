# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""SAE Reconstruction Error Hook for computing error statistics during data collection."""

import typing as t
from typing import Any, Optional

import torch

from ccdelta.models.model_with_hooks import collect_control_ids
from ccdelta.utils.logging import get_logger
from ccdelta.utils.sae import load_sae_model_for_hook

logger = get_logger(__name__)


class SAEReconstructionErrorHook:
    """Inner hook for SAE reconstruction error and activation analysis.

    This hook computes the encode-decode roundtrip through an SAE and returns:
    - 'sae_features': encoded features [B, S, D_sae]
    - 'error_stats': dict with per-sequence error statistics
    - 'activation_stats': dict with feature activation statistics

    Designed to be used as an inner hook within PairedResponsesHook for
    analyzing SAE reconstruction quality on paired data.
    """

    def __init__(
        self,
        module_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        sae_config: dict[str, Any] | None = None,
        return_sparse: bool = False,
        tokenizer: t.Any = None,
        skip_special_token_filtering: bool = False,
        l0_threshold: float = 0.0,
        **kwargs,
    ):
        """Initialize SAEReconstructionErrorHook.

        Args:
            module_name: Name of the module this hook is attached to
            device: Device to load SAE model on
            dtype: Data type for SAE model
            sae_config: SAE configuration dict
            return_sparse: If True, return features as a sparse tensor.
            tokenizer: Tokenizer for identifying special tokens.
            skip_special_token_filtering: If True, include special tokens in stats.
                If False (default), exclude BOS/EOS/PAD from statistics.
            l0_threshold: Threshold for counting active features (default 0.0).
                Use small positive value (e.g., 1e-6) if SAE outputs tiny nonzeros.
            **kwargs: Additional arguments (ignored)
        """
        self.module_name = module_name
        self.device = device
        self.dtype = dtype
        self.return_sparse = return_sparse
        self.cache_dir = kwargs.get("cache_dir")
        self.tokenizer = tokenizer
        self.skip_special_token_filtering = skip_special_token_filtering
        self.l0_threshold = l0_threshold

        # Build set of control token IDs for separate statistics
        # Control tokens include: special tokens, whitespace, system prompt tokens
        self.control_token_ids: set = set()
        if tokenizer is not None and not skip_special_token_filtering:
            # Use collect_control_ids for comprehensive control token detection
            # This includes special tokens, whitespace, and system prompt tokens
            try:
                control_ids = collect_control_ids(
                    tokenizer, include_whitespace=True, include_system_prompts=True
                )
                self.control_token_ids = set(control_ids)
                logger.info(
                    f"Collected {len(self.control_token_ids)} control token IDs for separate stats"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to collect control IDs: {e}, falling back to all_special_ids"
                )
                # Fallback: collect all special token IDs
                if hasattr(tokenizer, "all_special_ids"):
                    self.control_token_ids = set(tokenizer.all_special_ids)
                else:
                    # Fallback: collect individual special tokens
                    for attr in [
                        "bos_token_id",
                        "eos_token_id",
                        "pad_token_id",
                        "unk_token_id",
                    ]:
                        token_id = getattr(tokenizer, attr, None)
                        if token_id is not None:
                            self.control_token_ids.add(token_id)
                logger.info(
                    f"Will use {len(self.control_token_ids)} special token IDs for separate stats"
                )

        # State for batch processing
        self.batch: t.Optional[dict] = None
        self.batch_idx: t.Optional[int] = None
        self._batch_count: int = 0  # Track batches for logging

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
        logger.info(f"Initialized SAEReconstructionErrorHook for {module_name}")

    def update(self, batch_idx: int, batch: dict):
        """Update hook state with batch information."""
        self.batch = batch
        self.batch_idx = batch_idx

    def get_sae_metadata(self) -> dict:
        """Extract all available metadata from SAE model with graceful fallbacks.

        Returns:
            Dictionary containing SAE configuration and metadata.
        """
        metadata = {}

        if self.sae_model is None:
            return metadata

        # From sae_lens loading (if available - these are attached in sae.py)
        for attr in [
            "sae_lens_release",
            "sae_lens_sae_id",
            "expected_l0",
            "expected_var_explained",
            "neuronpedia_id",
            "repo_id",
            "model_name_from_directory",
        ]:
            if hasattr(self.sae_model, attr):
                value = getattr(self.sae_model, attr)
                if value is not None:
                    metadata[attr] = value

        # From SAE config
        try:
            cfg = self.sae_model.cfg
            metadata["d_in"] = cfg.d_in
            metadata["d_sae"] = cfg.d_sae
            metadata["dtype"] = str(cfg.dtype)
            metadata["architecture"] = cfg.architecture()
            metadata["normalize_activations"] = cfg.normalize_activations
            metadata["apply_b_dec_to_input"] = cfg.apply_b_dec_to_input
        except Exception as e:
            logger.warning(f"Could not extract SAE config: {e}")

        # From SAE metadata object (if available)
        try:
            if hasattr(self.sae_model, "cfg") and hasattr(
                self.sae_model.cfg, "metadata"
            ):
                meta = self.sae_model.cfg.metadata
                if meta is not None:
                    # Export ALL metadata attributes
                    if hasattr(meta, "to_dict"):
                        metadata["sae_metadata"] = meta.to_dict()
                    elif hasattr(meta, "__dict__"):
                        metadata["sae_metadata"] = dict(meta.__dict__)
        except Exception as e:
            logger.warning(f"Could not extract SAE metadata: {e}")

        # Add our own config for reference
        if self.sae_config:
            metadata["sae_config_used"] = self.sae_config

        return metadata

    def __call__(self, module, input, output) -> dict:
        """Compute SAE activations and reconstruction error statistics.

        Returns:
            Dictionary containing:
            - 'sae_features': Encoded SAE features [B, S, D_sae]
            - 'error_stats': Per-sequence error statistics (all tokens)
            - 'error_stats_content': Per-sequence error statistics (content tokens only)
            - 'error_stats_control': Per-sequence error statistics (control tokens only)
            - 'activation_stats': Per-sequence activation statistics (all tokens)
            - 'activation_stats_content': Per-sequence activation statistics (content tokens only)
            - 'activation_stats_control': Per-sequence activation statistics (control tokens only)
        """
        logger.debug(
            f"Called with output type: {type(output)}, "
            f"shape: {output.shape if hasattr(output, 'shape') else 'N/A'}"
        )

        if self.sae_model is None:
            logger.warning("SAE model not loaded, returning empty result")
            return {
                "sae_features": output,
                "error_stats": {},
                "error_stats_content": {},
                "error_stats_control": {},
                "activation_stats": {},
                "activation_stats_content": {},
                "activation_stats_control": {},
            }

        # Ensure output is on the same device as SAE model
        output = output.to(
            device=self.sae_model.W_enc.device, dtype=self.sae_model.W_enc.dtype
        )

        # Build masks for all/control/content tokens
        masks = self._build_valid_token_mask(output.device, output.shape[:2])

        # 1. Encode through SAE
        sae_features = self.sae_model.encode(output)

        # 2. Decode (reconstruct)
        reconstruction = self.sae_model.decode(sae_features)

        # 3. Compute error
        error = output - reconstruction

        # 4. Compute per-sequence ERROR statistics for each category
        # All tokens (for backwards compatibility)
        error_stats = self._compute_error_stats(error, output, masks["all"])
        # Content tokens only (for clean analysis)
        error_stats_content = self._compute_error_stats(error, output, masks["content"])
        # Control tokens only (for separate reporting)
        error_stats_control = self._compute_error_stats(error, output, masks["control"])

        # 5. Compute per-sequence ACTIVATION statistics for each category
        activation_stats = self._compute_activation_stats(sae_features, masks["all"])
        activation_stats_content = self._compute_activation_stats(
            sae_features, masks["content"]
        )
        activation_stats_control = self._compute_activation_stats(
            sae_features, masks["control"]
        )

        # Log detailed stats for first batch to verify control/content separation
        self._batch_count += 1
        if self._batch_count == 1:
            num_all = masks["all"].sum().item() if masks["all"] is not None else 0
            num_control = (
                masks["control"].sum().item() if masks["control"] is not None else 0
            )
            num_content = (
                masks["content"].sum().item() if masks["content"] is not None else 0
            )

            logger.info(f"=== FIRST BATCH TOKEN SEPARATION STATS ===")
            logger.info(f"  Total valid tokens: {num_all}")
            logger.info(
                f"  Control tokens: {num_control} ({100*num_control/max(num_all,1):.1f}%)"
            )
            logger.info(
                f"  Content tokens: {num_content} ({100*num_content/max(num_all,1):.1f}%)"
            )

            # Log FVE comparison to verify separation is meaningful
            fve_all = error_stats.get("fve")
            fve_content = error_stats_content.get("fve")
            fve_control = error_stats_control.get("fve")
            if fve_all is not None:
                logger.info(f"  FVE (all tokens):     {fve_all.mean().item():.4f}")
            if fve_content is not None:
                logger.info(f"  FVE (content only):   {fve_content.mean().item():.4f}")
            if fve_control is not None and num_control > 0:
                logger.info(f"  FVE (control only):   {fve_control.mean().item():.4f}")
            logger.info(f"===========================================")

        # Optionally convert features to sparse
        if self.return_sparse:
            sae_features = sae_features.to_sparse()
            logger.debug("Converted SAE features to sparse format")

        sae_features = sae_features.to(dtype=output.dtype)

        logger.debug(f"Returning SAE features with shape: {sae_features.shape}")

        return {
            "sae_features": sae_features,
            "error_stats": error_stats,
            "error_stats_content": error_stats_content,
            "error_stats_control": error_stats_control,
            "activation_stats": activation_stats,
            "activation_stats_content": activation_stats_content,
            "activation_stats_control": activation_stats_control,
        }

    def _build_valid_token_mask(
        self, device: torch.device, shape: tuple
    ) -> dict[str, Optional[torch.Tensor]]:
        """Build masks for different token categories (excludes padding).

        Args:
            device: Device to create mask on.
            shape: Tuple of (batch_size, seq_len).

        Returns:
            Dictionary with three masks:
            - 'all': Boolean tensor [B, S] for all valid tokens (excludes padding)
            - 'control': Boolean tensor [B, S] for control tokens only
            - 'content': Boolean tensor [B, S] for content tokens only (non-padding, non-control)
            Each mask has True = include in stats, False = exclude.
        """
        B, S = shape
        masks = {
            "all": None,
            "control": None,
            "content": None,
        }

        # Start with attention mask (excludes padding)
        all_valid_mask = None
        if self.batch is not None and "attention_mask" in self.batch:
            attention_mask = self.batch["attention_mask"]
            if isinstance(attention_mask, torch.Tensor):
                all_valid_mask = attention_mask.to(device=device, dtype=torch.bool)
            else:
                all_valid_mask = torch.tensor(
                    attention_mask, device=device, dtype=torch.bool
                )

            # Verify shape
            if all_valid_mask.shape != (B, S):
                logger.warning(
                    f"Attention mask shape {all_valid_mask.shape} doesn't match expected {(B, S)}, ignoring"
                )
                all_valid_mask = None

        # Default to all True if no attention mask
        if all_valid_mask is None:
            all_valid_mask = torch.ones((B, S), dtype=torch.bool, device=device)

        masks["all"] = all_valid_mask

        # Build control token mask if we have control token IDs and input_ids
        if (
            self.control_token_ids
            and self.batch is not None
            and "input_ids" in self.batch
        ):
            input_ids = self.batch["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.to(device=device)
            else:
                input_ids = torch.tensor(input_ids, device=device)

            # Verify shape
            if input_ids.shape == (B, S):
                # Create mask for control tokens using vectorized isin
                control_ids_tensor = torch.tensor(
                    list(self.control_token_ids), dtype=input_ids.dtype, device=device
                )
                control_mask = torch.isin(input_ids, control_ids_tensor)

                # Control tokens: valid AND control
                masks["control"] = all_valid_mask & control_mask
                # Content tokens: valid AND NOT control
                masks["content"] = all_valid_mask & (~control_mask)

                num_control = masks["control"].sum().item()
                num_content = masks["content"].sum().item()
                logger.debug(
                    f"Token split: {num_control} control, {num_content} content"
                )
            else:
                logger.warning(
                    f"Input IDs shape {input_ids.shape} doesn't match expected {(B, S)}, "
                    "cannot separate control/content tokens"
                )
                # Fall back to all tokens as content
                masks["content"] = all_valid_mask
                masks["control"] = torch.zeros((B, S), dtype=torch.bool, device=device)
        else:
            # No control token filtering - treat all valid as content
            masks["content"] = all_valid_mask
            masks["control"] = torch.zeros((B, S), dtype=torch.bool, device=device)

        return masks

    def _padded_to_filtered_indices(
        self,
        padded_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert indices from padded space to filtered space.

        For a mask [1, 1, 0, 0, 1], valid positions are [0, 1, 4] which map
        to filtered indices [0, 1, 2]. This method converts padded indices
        to their corresponding filtered indices.

        Args:
            padded_indices: [B, K] indices into the padded sequence.
            attention_mask: [B, S] mask where True=valid, False=padding.

        Returns:
            [B, K] indices into the filtered sequence. Invalid positions are -1.
        """
        B, S = attention_mask.shape
        device = attention_mask.device

        # Build per-batch mapping: mapping[b, padded_idx] = filtered_idx or -1
        mapping = torch.full((B, S), -1, dtype=torch.long, device=device)
        for b in range(B):
            valid_mask = attention_mask[b].bool()
            num_valid = valid_mask.sum().item()
            mapping[b, valid_mask] = torch.arange(num_valid, device=device)

        # Clamp indices to valid range for gather (out-of-bounds will be handled by -1 in mapping)
        safe_indices = padded_indices.clamp(0, S - 1)

        # Remap using gather
        return mapping.gather(dim=-1, index=safe_indices)

    def _compute_error_stats(
        self,
        error: torch.Tensor,
        original: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> dict:
        """Compute per-sequence error statistics, respecting attention mask.

        Args:
            error: [B, S, H] reconstruction error tensor.
            original: [B, S, H] original activation tensor.
            attention_mask: Optional [B, S] mask where True=valid, False=padding.

        Returns:
            Dictionary with error statistics. Note that top_10_*_token_indices
            are in FILTERED space (matching positions in the filtered token list),
            not padded space. Returns NaN values for batches with no valid tokens.
        """
        B, S, _H = error.shape
        device = error.device
        d_in = self.sae_model.cfg.d_in

        # Check for empty mask (no valid tokens)
        if attention_mask is not None:
            has_valid_tokens = attention_mask.any()
            if not has_valid_tokens:
                # Return NaN/zero-filled stats for completely empty mask
                nan_tensor = torch.full((B,), float("nan"), device=device)
                nan_k = torch.full((B, min(10, S)), float("nan"), device=device)
                idx_k = torch.full((B, min(10, S)), -1, dtype=torch.long, device=device)
                zero_tensor = torch.zeros((B,), device=device)
                D = original.shape[-1]
                return {
                    "error_norm_mean": nan_tensor,
                    "error_norm_max": nan_tensor,
                    "error_norm_min": nan_tensor,
                    "original_norm_mean": nan_tensor,
                    "original_norm_max": nan_tensor,
                    "relative_error_mean": nan_tensor,
                    "fvu": nan_tensor,
                    "fve": nan_tensor,
                    "mse_mean": nan_tensor,
                    "top_10_highest_error_token_indices": idx_k,
                    "top_10_highest_error_values": nan_k,
                    "top_10_lowest_error_token_indices": idx_k,
                    "top_10_lowest_error_values": nan_k,
                    # Sum fields for FVU_mean - zeros so they don't affect global aggregation
                    "sum_x": torch.zeros((B, D), dtype=torch.float64, device=device),
                    "sum_x_sq": zero_tensor,
                    "sum_error_sq": zero_tensor,
                    "token_count": zero_tensor,
                }

        # Compute squared norms in fp32 for numerical stability (avoids sqrt then square)
        # This is both more stable and faster than norm().square()
        error_sq_per_token = (error.float() ** 2).sum(dim=-1)  # [B, S] - ||e_t||²
        original_sq_per_token = (original.float() ** 2).sum(dim=-1)  # [B, S] - ||x_t||²

        # Derive L2 norms from squared norms for legacy stats
        error_norm_per_token = error_sq_per_token.sqrt()  # [B, S] - ||e_t||
        original_norm_per_token = original_sq_per_token.sqrt()  # [B, S] - ||x_t||

        if attention_mask is not None:
            # Explicit dtype casting for robustness
            mask = attention_mask.to(dtype=error_sq_per_token.dtype)
            mask_bool = attention_mask.bool()
            # Per-batch valid counts - check if any batch has no valid tokens
            valid_counts_raw = mask.sum(dim=-1)  # [B]
            has_valid_per_batch = valid_counts_raw > 0
            valid_counts = valid_counts_raw.clamp(
                min=1.0
            )  # [B] - clamp to avoid division by zero

            # L2 norm stats (mean, max, min)
            error_norm_mean = (error_norm_per_token * mask).sum(dim=-1) / valid_counts
            original_norm_mean = (original_norm_per_token * mask).sum(
                dim=-1
            ) / valid_counts

            # For max/min, mask invalid positions
            error_for_max = error_norm_per_token.clone()
            error_for_max[~mask_bool] = float("-inf")
            error_norm_max = error_for_max.max(dim=-1).values

            error_for_min = error_norm_per_token.clone()
            error_for_min[~mask_bool] = float("inf")
            error_norm_min = error_for_min.min(dim=-1).values

            original_for_max = original_norm_per_token.clone()
            original_for_max[~mask_bool] = float("-inf")
            original_norm_max = original_for_max.max(dim=-1).values

            # Relative error (mean of per-token ratios)
            relative_error = error_norm_per_token / (original_norm_per_token + 1e-8)
            relative_error_mean = (relative_error * mask).sum(dim=-1) / valid_counts

            # Squared norm stats for FVU/FVE/MSE
            error_sq_mean = (error_sq_per_token * mask).sum(dim=-1) / valid_counts
            original_sq_mean = (original_sq_per_token * mask).sum(dim=-1) / valid_counts

            # Set NaN for batches with no valid tokens
            if not has_valid_per_batch.all():
                nan_mask = ~has_valid_per_batch
                error_norm_mean[nan_mask] = float("nan")
                error_norm_max[nan_mask] = float("nan")
                error_norm_min[nan_mask] = float("nan")
                original_norm_mean[nan_mask] = float("nan")
                original_norm_max[nan_mask] = float("nan")
                relative_error_mean[nan_mask] = float("nan")
                error_sq_mean[nan_mask] = float("nan")
                original_sq_mean[nan_mask] = float("nan")
        else:
            error_norm_mean = error_norm_per_token.mean(dim=-1)
            error_norm_max = error_norm_per_token.max(dim=-1).values
            error_norm_min = error_norm_per_token.min(dim=-1).values
            original_norm_mean = original_norm_per_token.mean(dim=-1)
            original_norm_max = original_norm_per_token.max(dim=-1).values
            relative_error_mean = (
                error_norm_per_token / (original_norm_per_token + 1e-8)
            ).mean(dim=-1)

            error_sq_mean = error_sq_per_token.mean(dim=-1)
            original_sq_mean = original_sq_per_token.mean(dim=-1)

        error_stats = {
            "error_norm_mean": error_norm_mean,  # [B]
            "error_norm_max": error_norm_max,  # [B]
            "error_norm_min": error_norm_min,  # [B]
            "original_norm_mean": original_norm_mean,  # [B]
            "original_norm_max": original_norm_max,  # [B]
            "relative_error_mean": relative_error_mean,  # [B]
        }

        # Add raw sums for global FVU_mean computation (Gao-style) in notebook.
        # NOTE: Aggregation must be TOKEN-WEIGHTED, not sample-weighted:
        #   μ = Σsum_x / Σtoken_count  (global mean across ALL tokens)
        #   E[||x||²] = Σsum_x_sq / Σtoken_count
        #   variance = E[||x||²] - ||μ||²  (clamp to ≥0 for numerical stability)
        #   FVU_mean = (Σsum_error_sq / Σtoken_count) / variance
        # Use float64 for sums to avoid numerical drift during aggregation.
        if attention_mask is not None:
            # Vectorized sum over valid tokens: mask and sum
            # original: [B, S, D], mask: [B, S] -> masked_original: [B, S, D] with zeros at invalid
            masked_original = original.double() * mask_bool.unsqueeze(-1)
            sum_x_per_sample = masked_original.sum(dim=1)  # [B, D] float64

            sum_x_sq_per_sample = (original_sq_per_token.double() * mask).sum(
                dim=-1
            )  # [B] float64
            sum_error_sq_per_sample = (error_sq_per_token.double() * mask).sum(
                dim=-1
            )  # [B] float64
            token_count_per_sample = valid_counts_raw.to(dtype=torch.int64)  # [B] int64
        else:
            sum_x_per_sample = original.sum(dim=1).double()  # [B, D] float64
            sum_x_sq_per_sample = original_sq_per_token.sum(
                dim=-1
            ).double()  # [B] float64
            sum_error_sq_per_sample = error_sq_per_token.sum(
                dim=-1
            ).double()  # [B] float64
            token_count_per_sample = torch.full(
                (B,), S, dtype=torch.int64, device=device
            )

        error_stats["sum_x"] = sum_x_per_sample  # [B, D] float64
        error_stats["sum_x_sq"] = sum_x_sq_per_sample  # [B] float64
        error_stats["sum_error_sq"] = sum_error_sq_per_sample  # [B] float64
        error_stats["token_count"] = token_count_per_sample  # [B] int64

        # === Energy-normalized FVU/FVE (per-sample) ===
        # These metrics use E[||x||²] in the denominator (total activation energy).
        #
        # FVU_energy = E[||e||²] / E[||x||²]
        # FVE_energy = 1 - FVU_energy
        #
        # NOTE: This is NOT Gao/GemmaScope-style FVU, which uses variance in the
        # denominator: FVU_mean = E[||e||²] / E[||x - μ||²]. The two are equivalent
        # when activations are zero-mean (||μ||² ≈ 0). For Gao-style FVU, use the
        # sum_x/sum_x_sq/sum_error_sq fields to compute global variance-normalized
        # FVU in post-processing (see notebook).
        #
        # The notebook should display these as "FVU_energy" / "FVE_energy" to
        # distinguish from "FVU_mean" / "FVE_mean" (Gao-style).

        # FVU_energy = E[||e||²] / E[||x||²]
        fvu = error_sq_mean / (original_sq_mean + 1e-8)
        error_stats["fvu"] = fvu  # [B] energy-normalized

        # FVE_energy = 1 - FVU_energy
        fve = 1.0 - fvu
        error_stats["fve"] = fve  # [B] energy-normalized, bounded

        # MSE = E[||e||²] / d_in - actual mean squared error per dimension
        mse_mean = error_sq_mean / float(d_in)
        error_stats["mse_mean"] = mse_mean  # [B] - comparable across models

        # Compute top-10 highest/lowest error tokens (from valid tokens only)
        if attention_mask is not None:
            # For topk, mask invalid with -inf (highest) and +inf (lowest)
            error_for_topk_high = error_norm_per_token.clone()
            error_for_topk_high[~mask_bool] = float("-inf")
            error_for_topk_low = error_norm_per_token.clone()
            error_for_topk_low[~mask_bool] = float("inf")
        else:
            error_for_topk_high = error_norm_per_token
            error_for_topk_low = error_norm_per_token

        k = min(10, S)  # Handle sequences shorter than 10

        # Top 10 highest error
        top_high_values, top_high_indices = error_for_topk_high.topk(
            k, dim=-1, largest=True
        )
        # Replace -inf sentinel values with NaN and indices with -1
        invalid_high = torch.isinf(top_high_values) & (top_high_values < 0)
        top_high_values = top_high_values.clone()
        top_high_values[invalid_high] = float("nan")
        top_high_indices = top_high_indices.clone()
        top_high_indices[invalid_high] = -1

        error_stats["top_10_highest_error_token_indices"] = top_high_indices  # [B, k]
        error_stats["top_10_highest_error_values"] = top_high_values  # [B, k]

        # Top 10 lowest error
        top_low_values, top_low_indices = error_for_topk_low.topk(
            k, dim=-1, largest=False
        )
        # Replace +inf sentinel values with NaN and indices with -1
        invalid_low = torch.isinf(top_low_values) & (top_low_values > 0)
        top_low_values = top_low_values.clone()
        top_low_values[invalid_low] = float("nan")
        top_low_indices = top_low_indices.clone()
        top_low_indices[invalid_low] = -1

        error_stats["top_10_lowest_error_token_indices"] = top_low_indices  # [B, k]
        error_stats["top_10_lowest_error_values"] = top_low_values  # [B, k]

        # Convert indices from padded space to filtered space
        # This ensures indices match the filtered token list (after attention mask filtering)
        if attention_mask is not None:
            error_stats["top_10_highest_error_token_indices"] = (
                self._padded_to_filtered_indices(top_high_indices, attention_mask)
            )
            error_stats["top_10_lowest_error_token_indices"] = (
                self._padded_to_filtered_indices(top_low_indices, attention_mask)
            )

        return error_stats

    def _compute_activation_stats(
        self, sae_features: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> dict:
        """Compute per-sequence activation statistics, respecting attention mask.

        Args:
            sae_features: [B, S, D_sae] SAE feature activations.
            attention_mask: Optional [B, S] mask where 1=valid, 0=padding.

        Returns:
            Dictionary with activation statistics. Returns NaN values for batches
            with no valid tokens.
        """
        B, S, _D = sae_features.shape
        device = sae_features.device

        # Check for empty mask (no valid tokens)
        if attention_mask is not None:
            has_valid_tokens = attention_mask.any()
            if not has_valid_tokens:
                # Return NaN/zero-filled stats for completely empty mask
                nan_tensor = torch.full((B,), float("nan"), device=device)
                zero_tensor = torch.zeros((B,), dtype=torch.long, device=device)
                return {
                    "actual_l0_mean": nan_tensor,
                    "actual_l0_max": nan_tensor,
                    "activation_magnitude_mean": nan_tensor,
                    "activation_magnitude_max": nan_tensor,
                    "total_active_features": zero_tensor,
                }

        # sae_features: [B, S, D_sae]
        # Use configurable threshold for L0 (some SAEs output tiny nonzeros)
        active_mask = sae_features.abs() > self.l0_threshold  # [B, S, D_sae]
        l0_per_token = active_mask.sum(dim=-1).float()  # [B, S]

        if attention_mask is not None:
            # Explicit dtype casting for robustness
            mask = attention_mask.to(dtype=l0_per_token.dtype)
            mask_bool = attention_mask.bool()
            # Per-batch valid counts - check if any batch has no valid tokens
            valid_counts_raw = mask.sum(dim=-1)  # [B]
            has_valid_per_batch = valid_counts_raw > 0
            valid_counts = valid_counts_raw.clamp(min=1.0)  # [B]

            # L0 mean/max over valid tokens only
            l0_masked_sum = (l0_per_token * mask).sum(dim=-1)
            actual_l0_mean = l0_masked_sum / valid_counts

            l0_for_max = l0_per_token.clone()
            l0_for_max[~mask_bool] = float("-inf")
            actual_l0_max = l0_for_max.max(dim=-1).values

            # Activation magnitude (sum of abs values / l0)
            abs_sum_per_token = sae_features.abs().sum(dim=-1)  # [B, S]
            abs_sum_valid = (abs_sum_per_token * mask).sum(dim=-1)
            l0_sum_valid = l0_masked_sum
            activation_magnitude_mean = abs_sum_valid / (l0_sum_valid + 1e-8)

            # Max activation magnitude
            abs_max_per_token = sae_features.abs().max(dim=-1).values  # [B, S]
            abs_max_for_valid = abs_max_per_token.clone()
            abs_max_for_valid[~mask_bool] = float("-inf")
            activation_magnitude_max = abs_max_for_valid.max(dim=-1).values

            # Total active features across valid tokens only
            # Mask out invalid positions in active_mask before any()
            active_mask_valid = active_mask.clone()
            active_mask_valid[~mask_bool.unsqueeze(-1).expand_as(active_mask_valid)] = (
                False
            )
            total_active_features = active_mask_valid.any(dim=1).sum(dim=-1)

            # Set NaN for batches with no valid tokens
            if not has_valid_per_batch.all():
                nan_mask = ~has_valid_per_batch
                actual_l0_mean[nan_mask] = float("nan")
                actual_l0_max[nan_mask] = float("nan")
                activation_magnitude_mean[nan_mask] = float("nan")
                activation_magnitude_max[nan_mask] = float("nan")
                # total_active_features stays 0 for empty batches
        else:
            actual_l0_mean = l0_per_token.mean(dim=-1)
            actual_l0_max = l0_per_token.max(dim=-1).values
            activation_magnitude_mean = sae_features.abs().sum(dim=-1).mean(dim=-1) / (
                l0_per_token.mean(dim=-1) + 1e-8
            )
            activation_magnitude_max = (
                sae_features.abs().max(dim=-1).values.max(dim=-1).values
            )
            total_active_features = active_mask.any(dim=1).sum(dim=-1)

        return {
            "actual_l0_mean": actual_l0_mean,  # [B]
            "actual_l0_max": actual_l0_max,  # [B]
            "activation_magnitude_mean": activation_magnitude_mean,  # [B]
            "activation_magnitude_max": activation_magnitude_max,  # [B]
            "total_active_features": total_active_features,  # [B]
        }

    def __str__(self):
        return f"SAEReconstructionErrorHook(module_name={self.module_name}, sae_config={self.sae_config})"
