# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""SAE-based intervention hooks for feature space and weight space steering."""

import json
import math
import time
import typing as t
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F

from ccdelta.hooks.intervention_hook import InterventionHook
from ccdelta.hooks.sae_feature_selection import (
    select_steering_features,
)
from ccdelta.utils.logging import get_logger
from ccdelta.utils.sae import load_sae_model_for_hook

logger = get_logger(__name__)


def compute_decay_strength(
    generation_index: int,
    steer_n_new_tokens: int,
    decay_schedule: str = "none",
    base_strength: float = 1.0,
) -> float:
    """
    Compute decay multiplier based on generation index and decay schedule.

    Args:
        generation_index: Current generation step (0 = prompt)
        steer_n_new_tokens: Total number of new tokens to steer (0 = unlimited)
        decay_schedule: Type of decay ("none", "linear", "cosine")
        base_strength: Base steering strength (0.0 to 1.0)

    Returns:
        Decay multiplier in range [0.0, base_strength]
    """
    # No decay if schedule is "none" or unlimited steering
    if decay_schedule == "none" or steer_n_new_tokens == 0:
        return base_strength

    # During prompt phase, use full strength
    if generation_index == 0:
        return base_strength

    # Clamp generation_index to valid range [1, steer_n_new_tokens]
    progress = min(generation_index, steer_n_new_tokens) / steer_n_new_tokens

    if decay_schedule == "linear":
        # Linear decay from base_strength to 0
        return base_strength * (1.0 - progress)

    elif decay_schedule == "cosine":
        # Cosine decay: smooth transition from base_strength to 0
        # Uses cosine schedule: 0.5 * (1 + cos(π * progress))
        return base_strength * 0.5 * (1.0 + math.cos(math.pi * progress))

    else:
        logger.warning(
            f"Unknown decay_schedule '{decay_schedule}', using 'none'. "
            f"Valid options: 'none', 'linear', 'cosine'"
        )
        return base_strength


class SAEFeatureSpaceHook(InterventionHook):
    """
    SAE Feature Space steering hook that modifies SAE feature activations directly.

    This hook applies steering by:
    1. Encoding activations into SAE feature space
    2. Modifying specific features based on feature masks
    3. Decoding back to activation space
    4. Optionally adding reconstruction error

    Args:
        skip_special_token_filtering: If True, steer all tokens including special ones
                                     (BOS, EOS, padding). If False (default), only steer
                                     non-special tokens. Note: intervention_position is
                                     always respected regardless of this setting.
    """

    def __init__(
        self,
        module_name: str,
        device: str,
        intervention_position: t.Literal["all", "last", "prompt"] = "all",
        dtype: torch.dtype = torch.float32,
        sae_config: dict | None = None,
        error_coefficient: float = 0.0,
        normalization_mode: str = "none",
        feature_masks: Optional[List[dict]] = None,
        normalization_strength: float = 1.0,
        steering_multiplier: float = 1.0,
        feature_selection_method: str = "statistical",
        skip_special_token_filtering: bool = True,
        steer_n_new_tokens: int = 0,  # 0 = unlimited, N = steer prompt + N generated tokens
        decay_schedule: str = "none",  # Decay schedule: "none", "linear", "cosine"
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

        self.error_coefficient = error_coefficient
        self.normalization_mode = normalization_mode
        self.normalization_strength = normalization_strength

        self.feature_masks = feature_masks or []
        self.steering_multiplier = steering_multiplier
        self.feature_selection_method = feature_selection_method
        self.skip_special_token_filtering = skip_special_token_filtering
        self.steer_n_new_tokens = steer_n_new_tokens
        self.decay_schedule = decay_schedule
        self.cache_dir = kwargs.get("cache_dir")

        # Validate feature selection method
        valid_methods = ["statistical", "none"]
        if feature_selection_method not in valid_methods:
            raise ValueError(
                f"Invalid feature_selection_method '{feature_selection_method}'. "
                f"Valid options: {valid_methods}"
            )

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

        # Compiled dense mode state
        self._compiled_compare = False
        self._compiled_compare_mode = kwargs.get("compiled_compare_mode", False)

        # Initialize steering tensors as buffers via _update_steering_tensors
        self._update_steering_tensors()

    def fit(
        self,
        responses: Union[torch.Tensor, List[dict]],
        labels: torch.Tensor,
        **kwargs,
    ) -> None:
        """
        Fit the hook from paired data using robust statistical analysis.
        """
        if not (
            isinstance(responses, list)
            and len(responses) > 0
            and isinstance(responses[0], dict)
        ):
            logger.error(f"Expected a list of dicts, but got {type(responses)}")
            return

        # Determine if we should use all features
        use_all_features = self.feature_selection_method == "none"
        if use_all_features:
            logger.info("Feature selection method set to 'none' - using all features")

        # Call the main analysis function
        start_time = time.time()
        logger.info("Starting select_steering_features")
        assert "feature_diffing_method" in kwargs

        # Capture ranking method for metadata
        ranking_method = kwargs.get("ranking_method", "unknown")

        sig_features = select_steering_features(
            paired_data=responses,
            target_metric=kwargs.get("target_metric", "activation_max"),
            threshold_percentile=kwargs.get("threshold_percentile", 10),
            min_observations_percent=kwargs.get("min_observations_percent", 20.0),
            min_observations_absolute=kwargs.get("min_observations_absolute", 5),
            p_value_threshold=kwargs.get("p_value_threshold", 0.05),
            feature_diffing_method=kwargs["feature_diffing_method"],
            num_features=kwargs.get("num_features", 50),
            ranking_method=ranking_method,
            use_all_features=use_all_features,
            max_observations_percent=kwargs.get("max_observations_percent", 100.0),
            balance_pos_and_neg=kwargs.get("balance_pos_and_neg", False),
        )
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"select_steering_features completed in {duration:.3f} seconds")

        positive_features = sig_features.get("positive", {})
        negative_features = sig_features.get("negative", {})

        logger.info(
            f"Found {len(positive_features)} positive and "
            f"{len(negative_features)} negative significant features."
        )

        all_features = sig_features.get("top_features", {})

        # Formulate the steering vector and capture full metadata
        self.feature_masks = []
        self.feature_metadata = {}  # Store full metadata for post-hoc analysis

        for rank, (feat_id, feat_info) in enumerate(all_features.items(), 1):
            steering_value = feat_info.get("mean_diff", 0.0)
            self.feature_masks.append(
                {"feature": feat_id, "add": steering_value * self.steering_multiplier}
            )

            # Store full metadata for post-hoc analysis (use NaN for missing values)
            self.feature_metadata[str(feat_id)] = {
                # Ranking information (NEW!)
                "rank": rank,

                # Original sample statistics (from _compute_original_sample_statistics)
                "mean_positive": feat_info.get("mean_positive", float('nan')),
                "std_positive": feat_info.get("std_positive", float('nan')),
                "mean_negative": feat_info.get("mean_negative", float('nan')),
                "std_negative": feat_info.get("std_negative", float('nan')),

                # Difference statistics (from feature selection analysis)
                "n_observations": feat_info.get("n_observations", float('nan')),
                "median_diff": feat_info.get("median_diff", float('nan')),
                "mean_diff": feat_info.get("mean_diff", float('nan')),
                "std_diff": feat_info.get("std_diff", float('nan')),
                "pos_count": feat_info.get("pos_count", float('nan')),
                "neg_count": feat_info.get("neg_count", float('nan')),
                "prop_pos": feat_info.get("prop_pos", float('nan')),
                "prop_neg": feat_info.get("prop_neg", float('nan')),

                # Statistical significance (NaN for magnitude-only ranking)
                "p_corrected_pos": feat_info.get("p_corrected_pos", float('nan')),
                "p_corrected_neg": feat_info.get("p_corrected_neg", float('nan')),
                "classification": feat_info.get("classification", "unknown"),

                # Derived measures
                "abs_strength": abs(feat_info.get("mean_diff", float('nan'))),
                "steering_multiplier": self.steering_multiplier,
                "feature_id": int(feat_id)
            }

        logger.info(f"Created {len(self.feature_masks)} feature masks for steering.")

        # Update the hook's internal tensors
        self._update_steering_tensors()
        self._post_load()

    def _post_load(self) -> None:
        """
        Override to skip buffer size check since empty buffers are valid for SAE hooks.
        If compiled_compare_mode is enabled and feature masks are loaded, compile to dense.
        """
        if (
            self._compiled_compare_mode
            and hasattr(self, "mul_indices")
            and len(self.mul_indices) > 0
        ):
            self.compile_to_dense()

    def update_feature_masks(self, feature_masks: List[dict]) -> None:
        """
        Update feature masks for dynamic steering without reloading SAE.

        Args:
            feature_masks: List of feature manipulation specifications
        """
        self.feature_masks = feature_masks
        self._update_steering_tensors()

    def _update_steering_tensors(self) -> None:
        """Convert declarative feature_masks into tensors and register as named buffers."""
        logger.info(
            f"Updating steering tensors with {len(self.feature_masks)} feature masks"
        )

        if not self.feature_masks:
            # Don't register empty buffers - they'll be loaded from state_dict if needed
            # This avoids size mismatch issues when loading learned parameters
            logger.debug("No feature masks to process")
            return

        indices, m_values, a_values = [], [], []
        for mask in self.feature_masks:
            indices.append(mask["feature"])
            if "set" in mask:
                m_values.append(0.0)
                a_values.append(mask["set"])
            elif "mul" in mask and "add" in mask:
                m_values.append(mask["mul"])
                a_values.append(mask["add"])
            elif "add" in mask:
                m_values.append(1.0)
                a_values.append(mask["add"])

        logger.info(f"Processing {len(indices)} features for steering")

        # Register mul_indices separately to preserve torch.long dtype for indexing
        # The parent register_named_buffers() converts everything to self.dtype, which breaks indexing
        self.register_buffer("mul_indices", torch.tensor(indices, dtype=torch.long))

        # Register other buffers normally (they can be converted to model dtype)
        self.register_named_buffers(
            mul_values=torch.tensor(m_values, dtype=self.dtype),
            add_values=torch.tensor(a_values, dtype=self.dtype),
        )

    def compile_to_dense(self) -> None:
        """Precompute the dense steering vector from SAE decoder weights.

        Validates that the mathematical simplification h' = h + decay * V_dense
        is valid for the current configuration, then precomputes V_dense.

        Safe to call multiple times (overwrites existing compiled_vector).

        Raises:
            ValueError: If error_coefficient != 1.0, normalization is enabled,
                        or tensor shapes are invalid.
        """
        # Validate method-level conditions
        if self.error_coefficient != 1.0:
            raise ValueError(
                f"compile_to_dense requires error_coefficient=1.0, got {self.error_coefficient}"
            )
        # Hydra serializes null as the string "None" in command-line overrides
        if self.normalization_mode not in (None, "none", "None"):
            raise ValueError(
                f"compile_to_dense requires normalization_mode=None or 'none', "
                f"got '{self.normalization_mode}'"
            )

        # Load SAE model if not loaded
        if self.sae_model is None:
            if self.sae_config is not None:
                self.sae_model = load_sae_model_for_hook(
                    config=self.sae_config,
                    device=self.device,
                    dtype=self.dtype,
                    cache_dir=self.cache_dir,
                )
            else:
                raise ValueError("Cannot compile_to_dense: no SAE model or config available")

        # Validate SAE-level normalization
        compile_guards_status = "passed"
        sae_cfg = getattr(self.sae_model, "cfg", None)
        if sae_cfg is not None:
            norm_act = getattr(sae_cfg, "normalize_activations", "none")
            if norm_act != "none":
                raise ValueError(
                    f"compile_to_dense requires SAE normalize_activations='none', "
                    f"got '{norm_act}'"
                )
        else:
            compile_guards_status = "sae_cfg_not_available"
            warnings.warn(
                "Cannot verify SAE normalize_activations — cfg not available. "
                "Empirical validation recommended."
            )

        # Compute dense vector under no_grad — this is an offline compile step
        with torch.no_grad():
            wdec = self.sae_model.W_dec
            device = wdec.device

            indices = self.mul_indices.to(device=device, dtype=torch.long)
            values = self.add_values.to(device=device, dtype=torch.float32)

            # Shape / bounds checks
            if indices.ndim != 1 or values.ndim != 1:
                raise ValueError("mul_indices and add_values must be 1D")
            if indices.numel() != values.numel():
                raise ValueError("mul_indices and add_values must have same length")
            if indices.numel() > 0 and (
                (indices < 0).any() or (indices >= wdec.shape[0]).any()
            ):
                raise ValueError("mul_indices out of bounds for W_dec")

            # Select rows first, then cast to float32 — avoids duplicating entire decoder
            if indices.numel() == 0:
                V = torch.zeros(wdec.shape[1], device=device, dtype=torch.float32)
            else:
                selected_rows = wdec.index_select(0, indices).float()  # [n_features, d_model]
                V = values @ selected_rows  # [d_model]

            V = V.detach()

        # Safe for repeated calls
        if "compiled_vector" in self._buffers:
            self.compiled_vector = V
        else:
            self.register_buffer("compiled_vector", V)

        # Initialize online accumulators for error stats
        # Elementwise absolute error stats
        self._err_count_elem = 0
        self._err_sum_abs = 0.0
        self._err_max_abs = 0.0
        self._err_min_abs = float("inf")
        self._err_welford_mean = 0.0
        self._err_welford_M2 = 0.0
        self._err_sum_sq = 0.0
        # Tokenwise relative L2 error stats
        self._rel_err_sum = 0.0
        self._rel_err_max = 0.0
        self._rel_err_count_tokens = 0
        # Shared
        self._nonfinite_count = 0
        self._hidden_dim = V.shape[0]
        # Latency tracking
        self._latency_compiled_total_ms = 0.0
        self._latency_naive_total_ms = 0.0
        self._latency_call_count = 0
        # Per-call error log for outlier identification
        self._per_call_errors: list = []
        self._compile_guards_status = compile_guards_status

        self._compiled_compare = True
        logger.info(
            f"[{self.module_name}] Compiled to dense vector, shape={V.shape}, "
            f"norm={V.norm().item():.4f}, guards={compile_guards_status}"
        )

    def get_compilation_error_stats(self) -> dict:
        """Return accumulated compilation error statistics with provenance metadata.

        Error stats are split into:
        - Elementwise absolute error (mean/max/min/std/rmse over individual hidden dims)
        - Tokenwise relative L2 error (per-token ||diff||_2 / ||naive||_2)
        """
        if not self._compiled_compare:
            return {"compiled": False}

        n_elem = self._err_count_elem
        if n_elem == 0:
            return {
                "compiled": True,
                "metadata": self._get_compile_metadata(),
                "error": {
                    "num_elements_abs": 0,
                    "num_token_positions_rel": 0,
                    "num_nonfinite_skipped": self._nonfinite_count,
                },
                "latency": {
                    "num_forward_calls": self._latency_call_count,
                    "compiled_total_ms": self._latency_compiled_total_ms,
                    "naive_total_ms": self._latency_naive_total_ms,
                    "compiled_mean_ms": self._latency_compiled_total_ms / max(self._latency_call_count, 1),
                    "naive_mean_ms": self._latency_naive_total_ms / max(self._latency_call_count, 1),
                    "speedup": self._latency_naive_total_ms / max(self._latency_compiled_total_ms, 1e-6),
                },
                "per_call_errors": self._per_call_errors,
            }
        variance = self._err_welford_M2 / (n_elem - 1) if n_elem > 1 else 0.0
        n_tokens = self._rel_err_count_tokens
        return {
            "compiled": True,
            "metadata": self._get_compile_metadata(),
            "error": {
                "mean_abs_error": self._err_sum_abs / n_elem,
                "max_abs_error": self._err_max_abs,
                "min_abs_error": self._err_min_abs,
                "std_abs_error": variance ** 0.5,
                "rmse": (self._err_sum_sq / n_elem) ** 0.5,
                "mean_rel_l2_error": (
                    self._rel_err_sum / n_tokens if n_tokens > 0 else 0.0
                ),
                "max_rel_l2_error": self._rel_err_max,
                "num_elements_abs": n_elem,
                "num_token_positions_rel": n_tokens,
                "hidden_dim": self._hidden_dim,
                "num_nonfinite_skipped": self._nonfinite_count,
            },
            "latency": {
                "num_forward_calls": self._latency_call_count,
                "compiled_total_ms": self._latency_compiled_total_ms,
                "naive_total_ms": self._latency_naive_total_ms,
                "compiled_mean_ms": self._latency_compiled_total_ms / max(self._latency_call_count, 1),
                "naive_mean_ms": self._latency_naive_total_ms / max(self._latency_call_count, 1),
                "speedup": self._latency_naive_total_ms / max(self._latency_compiled_total_ms, 1e-6),
            },
            "per_call_errors": self._per_call_errors,
        }

    def _get_compile_metadata(self) -> dict:
        """Build provenance metadata for compilation error stats."""
        meta = {
            "module_name": self.module_name,
            "feature_count": len(self.mul_indices) if hasattr(self, "mul_indices") else 0,
            "hidden_dim": getattr(self, "_hidden_dim", None),
            "decay_schedule": self.decay_schedule,
            "steer_n_new_tokens": self.steer_n_new_tokens,
            "error_coefficient": self.error_coefficient,
            "normalization_mode": self.normalization_mode,
            "compile_guards_status": getattr(self, "_compile_guards_status", "unknown"),
        }
        if self.sae_config is not None:
            meta["sae_release"] = self.sae_config.get("release", "unknown")
            meta["sae_id"] = self.sae_config.get("sae_id", "unknown")
        if self.sae_model is not None:
            sae_cfg = getattr(self.sae_model, "cfg", None)
            if sae_cfg is not None:
                meta["sae_normalize_activations"] = getattr(
                    sae_cfg, "normalize_activations", "unknown"
                )
        return meta

    def state_dict(self, *args, **kwargs):
        """Override to exclude sae_model from saved state, include only steering parameters and metadata."""
        state = super().state_dict(*args, **kwargs)

        # Filter out all SAE model parameters (which start with 'sae_model.')
        steering_state = {
            k: v for k, v in state.items() if not k.startswith("sae_model.")
        }

        # Add SAE config for reloading (convert config dict to tensor for state_dict compatibility)
        if self.sae_config is not None:
            config_str = json.dumps(self.sae_config, sort_keys=True)
            sae_config_tensor = torch.tensor(
                [ord(c) for c in config_str], dtype=torch.uint8
            )
            steering_state["sae_config_tensor"] = sae_config_tensor

        # Add feature metadata if available
        if hasattr(self, 'feature_metadata') and self.feature_metadata:
            # Use a custom JSON encoder that handles NaN values and numpy types properly
            def safe_json_encoder(obj):
                if isinstance(obj, float) and math.isnan(obj):
                    return "NaN"
                elif hasattr(obj, 'item'):  # Handle numpy scalars (int64, float64, etc.)
                    return obj.item()
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)  # Fallback for any other types

            metadata_dict = {k: {mk: safe_json_encoder(mv) for mk, mv in v.items()}
                           for k, v in self.feature_metadata.items()}
            metadata_str = json.dumps(metadata_dict, sort_keys=True)
            metadata_tensor = torch.tensor(
                [ord(c) for c in metadata_str], dtype=torch.uint8
            )
            steering_state["feature_metadata_tensor"] = metadata_tensor

        return steering_state

    def load_state_dict(self, state_dict, strict=True):
        """Override to reload SAE model and metadata after loading steering parameters."""
        # Make a copy to avoid modifying the original
        state_dict_copy = dict(state_dict)

        # Extract and convert SAE config back from tensor format if present
        if "sae_config_tensor" in state_dict_copy:
            sae_config_tensor = state_dict_copy.pop("sae_config_tensor")
            self.sae_config = json.loads(
                "".join(chr(int(c)) for c in sae_config_tensor)
            )

        # Extract and convert feature metadata back from tensor format if present
        if "feature_metadata_tensor" in state_dict_copy:
            metadata_tensor = state_dict_copy.pop("feature_metadata_tensor")
            metadata_str = "".join(chr(int(c)) for c in metadata_tensor)
            metadata_dict = json.loads(metadata_str)

            # Convert "NaN" strings back to float('nan')
            def nan_json_decoder(obj):
                if obj == "NaN":
                    return float('nan')
                return obj

            self.feature_metadata = {k: {mk: nan_json_decoder(mv) for mk, mv in v.items()}
                                   for k, v in metadata_dict.items()}
        else:
            # For backwards compatibility, initialize empty metadata if not present
            self.feature_metadata = {}

        # Manually register steering buffers since they weren't initialized at construction
        if "mul_indices" in state_dict_copy:
            self.register_buffer("mul_indices", state_dict_copy.pop("mul_indices"))
        if "mul_values" in state_dict_copy:
            self.register_buffer("mul_values", state_dict_copy.pop("mul_values"))
        if "add_values" in state_dict_copy:
            self.register_buffer("add_values", state_dict_copy.pop("add_values"))
        if "compiled_vector" in state_dict_copy:
            self.register_buffer("compiled_vector", state_dict_copy.pop("compiled_vector"))

        # Load any remaining parameters (should be empty now)
        if state_dict_copy:
            super().load_state_dict(state_dict_copy, strict=False)

        # Reload SAE model from config if needed (instance variable, not saved)
        if self.sae_config is not None and self.sae_model is None:
            self.sae_model = load_sae_model_for_hook(
                config=self.sae_config,
                device=self.device,
                dtype=self.dtype,
                cache_dir=self.cache_dir,
            )

        self._post_load()

    @staticmethod
    def _unwrap(output):
        return (output[0], True) if isinstance(output, tuple) else (output, False)

    @staticmethod
    def _rewrap(tensor, was_tuple, orig_output):
        return (tensor,) + orig_output[1:] if was_tuple else tensor

    @staticmethod
    def _compute_steer_mask_for_call(
        gen_ctx,
        batch_dim: int,
        seq_dim: int,
        device: torch.device,
        where: str,
        skip_special_token_filtering,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns [B,S] bool: True = steer (non-special), False = pass-through.

        Args:
            skip_special_token_filtering: If True, ignore special tokens and treat all as steerable
                                         (but still apply phase gating based on intervention_position)
            attention_mask: Optional [B,S] tensor where 1=real token, 0=padding. If provided,
                           padding positions will not be steered regardless of other settings.
        """
        # 1) Base: non-special tokens only (unless skipping special token filtering)
        if skip_special_token_filtering:
            # Treat all tokens as steerable (ignore special tokens)
            logger.debug(
                f"Skipping special token filtering - treating all {batch_dim}x{seq_dim} tokens as steerable"
            )
            base = torch.ones((batch_dim, seq_dim), dtype=torch.bool, device=device)
        elif gen_ctx is None or getattr(gen_ctx, "special_mask_cur", None) is None:
            logger.debug(
                f"No gen_ctx or special_mask - treating all {batch_dim}x{seq_dim} tokens as steerable"
            )
            base = torch.ones((batch_dim, seq_dim), dtype=torch.bool, device=device)
        else:
            m = gen_ctx.special_mask_cur.to(
                device=device, dtype=torch.bool
            )  # True = special
            num_special = m.sum().item()
            logger.debug(
                f"Special token mask shape: {m.shape}, {num_special} special tokens detected"
            )

            # Align tail defensively
            if m.shape[0] != batch_dim:
                logger.warning(
                    f"Batch dimension mismatch: mask={m.shape[0]}, expected={batch_dim}. "
                    "Treating all tokens as steerable."
                )
                base = torch.ones((batch_dim, seq_dim), dtype=torch.bool, device=device)
            elif m.shape[1] == seq_dim:
                base = ~m
            elif m.shape[1] > seq_dim:
                logger.debug(
                    f"Truncating mask from {m.shape[1]} to last {seq_dim} positions"
                )
                base = ~m[:, -seq_dim:]
            else:
                left_pad_size = seq_dim - m.shape[1]
                logger.debug(
                    f"Left-padding mask from {m.shape[1]} to {seq_dim} positions"
                )
                # F.pad syntax: (left_pad, right_pad) for last dimension
                base = F.pad(~m, (left_pad_size, 0), value=False)

        # 1.5) Apply attention_mask if available (filters padding tokens)
        if attention_mask is not None:
            # Convert to bool using != 0 to handle float masks (e.g., 0.0/1.0)
            attn_bool = (attention_mask != 0).to(device=device, dtype=torch.bool)

            # Guard against batch dimension mismatch
            if attn_bool.shape[0] != batch_dim:
                logger.warning(
                    f"Attention mask batch mismatch: mask={attn_bool.shape[0]}, expected={batch_dim}. "
                    "Ignoring attention_mask."
                )
            # Handle sequence dimension alignment
            elif attn_bool.shape[1] == seq_dim:
                num_padding = (~attn_bool).sum().item()
                base = base & attn_bool
                logger.debug(
                    f"Applied attention_mask - filtered {num_padding} padding tokens"
                )
            elif attn_bool.shape[1] > seq_dim:
                # Truncate to last seq_dim positions
                attn_aligned = attn_bool[:, -seq_dim:]
                num_padding = (~attn_aligned).sum().item()
                base = base & attn_aligned
                logger.debug(
                    f"Truncated attention_mask from {attn_bool.shape[1]} to {seq_dim}, "
                    f"filtered {num_padding} padding tokens"
                )
            elif attn_bool.shape[1] < seq_dim:
                # Left-pad with False (these positions are padding)
                left_pad_size = seq_dim - attn_bool.shape[1]
                # F.pad syntax: (left_pad, right_pad) for last dimension
                attn_aligned = F.pad(attn_bool, (left_pad_size, 0), value=False)
                num_padding = (~attn_aligned).sum().item()
                base = base & attn_aligned
                logger.debug(
                    f"Left-padded attention_mask from {attn_bool.shape[1]} to {seq_dim}, "
                    f"filtered {num_padding} padding tokens"
                )

        # 2) Phase gating (always applied, regardless of special token filtering)
        is_decode = bool(getattr(gen_ctx, "is_decode_call", seq_dim == 1))
        logger.debug(
            f"Phase gating: where='{where}', is_decode={is_decode}, seq_dim={seq_dim}"
        )

        if where == "all":
            return base
        if where == "prompt":
            return (
                base
                if not is_decode
                else torch.zeros((batch_dim, seq_dim), dtype=torch.bool, device=device)
            )
        if where == "last":
            return (
                base
                if is_decode
                else torch.zeros((batch_dim, seq_dim), dtype=torch.bool, device=device)
            )
        return base  # default

    def __call__(self, module, input, output):
        """Apply SAE steering with special token masking using Option B approach.

        This applies steering to all tokens, then selectively keeps steered results
        only for non-special tokens using torch.where masking.
        """

        out_tensor, was_tuple = self._unwrap(output)  # out_tensor: [B,S,H]
        # print(f"In sae_hooks __call__ func: {out_tensor.shape}")
        if out_tensor.dim() < 3:
            logger.debug(f"[{self.module_name}] Skipping steering - tensor dim < 3")
            return output

        batch, seq, hidden = out_tensor.shape
        # print(f"[{self.module_name}] Input shape: [{batch}, {seq}, {hidden}]")

        # Check generation index limit if configured
        gen_idx = getattr(self.gen_ctx, "generation_index", 0)
        if self.steer_n_new_tokens > 0 and gen_idx > self.steer_n_new_tokens:
            logger.debug(
                f"[{self.module_name}] Skipping steering - generation_index {gen_idx} "
                f"exceeds limit {self.steer_n_new_tokens}"
            )
            return output

        # Extract attention_mask from input if available (HuggingFace signature: forward(input_ids, attention_mask, ...))
        attention_mask = None
        if len(input) > 1 and isinstance(input[1], torch.Tensor):
            candidate = input[1]
            # Verify it looks like an attention mask (2D with batch dimension matching)
            if candidate.ndim == 2 and candidate.shape[0] == batch:
                attention_mask = candidate
                logger.debug(
                    f"Extracted attention_mask with shape {attention_mask.shape}"
                )

        # Compute mask: True = steer (non-special), False = pass-through (special)
        # skip_special_token_filtering (from config) controls whether to filter special tokens,
        # but intervention_position logic is always applied
        steer_mask = self._compute_steer_mask_for_call(
            self.gen_ctx,
            batch,
            seq,
            out_tensor.device,
            where=self.intervention_position,
            skip_special_token_filtering=self.skip_special_token_filtering,
            attention_mask=attention_mask,
        )

        num_steer_tokens = steer_mask.sum().item()
        num_total_tokens = batch * seq

        # Fast path: no steering needed
        if not steer_mask.any().item():
            # print(f"[{self.module_name}] All tokens are special - no steering applied")
            return output

        # Fast path: steer all tokens (no special tokens present)
        if steer_mask.all().item():
            # print(f"[{self.module_name}] No special tokens - steering all")
            steered_output = self.forward(
                module, input, output, generation_index=gen_idx
            )

            # Log steering effect even for fast path
            # with torch.no_grad():
            #     steered_tensor = (
            #         steered_output[0]
            #         if isinstance(steered_output, tuple)
            #         else steered_output
            #     )
            #     original_norm = out_tensor.norm().item()
            #     steered_norm = steered_tensor.norm().item()
            #     diff_norm = (steered_tensor - out_tensor).norm().item()
            #     print(
            #         f"[{self.module_name}] Steering norms - "
            #         f"original: {original_norm:.2f}, steered: {steered_norm:.2f}, "
            #         f"diff: {diff_norm:.2f} ({100*diff_norm/original_norm:.1f}%)"
            #     )

            return steered_output

        # Mixed case: Apply steering to full tensor, then selectively keep results
        # Only log for longer sequences (prompt processing) to reduce noise
        if seq > 1:
            print(f"[{self.module_name}] Mixed case - applying Option B masking")
            print(
                f"[{self.module_name}] Steering {num_steer_tokens}/{num_total_tokens} tokens "
                f"({100*num_steer_tokens/num_total_tokens:.1f}%)"
            )

        # Apply steering to ALL tokens (including special ones)
        steered_output = self.forward(module, input, output, generation_index=gen_idx)
        steered_tensor = (
            steered_output[0] if isinstance(steered_output, tuple) else steered_output
        )

        # Log steering effect (only for longer sequences)
        if seq > 1:
            with torch.no_grad():
                original_norm = out_tensor.norm().item()
                steered_norm = steered_tensor.norm().item()
                diff_norm = (steered_tensor - out_tensor).norm().item()
                print(
                    f"[{self.module_name}] Steering norms - "
                    f"original: {original_norm:.2f}, steered: {steered_norm:.2f}, "
                    f"diff: {diff_norm:.2f} ({100*diff_norm/original_norm:.1f}%)"
                )

        # Selectively apply: steered for non-special, original for special tokens
        # steer_mask is [B,S], need to expand to [B,S,H] for element-wise selection
        final = torch.where(
            steer_mask.unsqueeze(-1),  # [B,S,1] - broadcast across hidden dim
            steered_tensor,  # Use steered values where mask is True
            out_tensor,  # Use original values where mask is False (special tokens)
        )

        if seq > 1:
            with torch.no_grad():
                print(
                    f"[{self.module_name}] Option B masking applied - output shape: {final.shape}"
                )

        return self._rewrap(final, was_tuple, output)

    def _is_prompt_phase(self, input, output):
        """
        Detect if we're in prompt processing phase (vs autoregressive generation).

        Uses hybrid detection combining past_key_values and sequence length:
        - Primary: Check past_key_values (HuggingFace models)
        - Fallback: Check sequence length for other models

        Returns True if we should apply intervention (prompt phase).
        """
        # Primary detection: Check for past_key_values (HuggingFace models)
        # HF signature: (input_ids, attention_mask, past_key_values, ...)
        if len(input) > 2 and input[2] is not None:
            # past_key_values exists = autoregressive generation
            past_key_values = input[2]
            if past_key_values is not None:
                # Has cached keys/values = not first pass = autoregressive
                return False

        # Fallback detection: Check sequence length
        tensor_to_check = output[0] if isinstance(output, tuple) else output

        # For per-token features, check sequence length dimension (-2)
        if tensor_to_check.ndim >= 2:
            seq_len = tensor_to_check.shape[-2]
            # Multi-token sequences are prompts; single tokens are autoregressive.
            return seq_len > 1

        # apply intervention if we can't determine otherwise.
        return True

    def forward(self, module, input, output, **kwargs) -> Any:
        """Apply SAE feature space steering with optional decay.

        When compiled_compare mode is active, runs both compiled (additive) and
        naive (SAE roundtrip) paths, records equivalence error, returns compiled output.

        Kwargs:
            generation_index: Number of tokens generated after prompt (0 during prompt)
        """
        with torch.no_grad():
            generation_index = kwargs.get("generation_index", 0)
            # Check if we have SAE model and steering parameters
            if (
                self.sae_model is None
                or not hasattr(self, "mul_indices")
                or len(self.mul_indices) == 0
            ):
                logger.debug("SAE steering skipped - missing components")
                return output

            original_output = output[0] if isinstance(output, tuple) else output

            # Compute decay strength (shared by both paths)
            decay_strength = compute_decay_strength(
                generation_index=generation_index,
                steer_n_new_tokens=self.steer_n_new_tokens,
                decay_schedule=self.decay_schedule,
                base_strength=1.0,
            )

            # Log decay effect when applicable
            if (
                generation_index > 0
                and self.decay_schedule != "none"
                and self.steer_n_new_tokens > 0
            ):
                logger.info(
                    f"[{self.module_name}] Applied {self.decay_schedule} decay: "
                    f"gen_idx={generation_index}/{self.steer_n_new_tokens}, strength={decay_strength:.3f}"
                )

            # --- Compiled compare mode: dual-path with error recording ---
            if self._compiled_compare:
                is_cuda = original_output.is_cuda

                if is_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    compiled_output = self._forward_compiled(original_output, decay_strength)
                    end.record()
                    torch.cuda.synchronize()
                    compiled_ms = start.elapsed_time(end)

                    start.record()
                    naive_output = self._forward_naive(original_output, decay_strength)
                    end.record()
                    torch.cuda.synchronize()
                    naive_ms = start.elapsed_time(end)
                else:
                    t0 = time.perf_counter()
                    compiled_output = self._forward_compiled(original_output, decay_strength)
                    compiled_ms = (time.perf_counter() - t0) * 1000

                    t0 = time.perf_counter()
                    naive_output = self._forward_naive(original_output, decay_strength)
                    naive_ms = (time.perf_counter() - t0) * 1000

                self._latency_compiled_total_ms += compiled_ms
                self._latency_naive_total_ms += naive_ms
                self._latency_call_count += 1

                # 3. Record error (both in float32 for accurate comparison)
                self._record_compilation_error(compiled_output, naive_output)

                # 4. Return compiled output in original dtype
                result = compiled_output.to(original_output.dtype)
                if isinstance(output, tuple):
                    return (result,) + output[1:]
                return result

            # --- Standard naive path ---
            return self._forward_naive_wrapped(
                original_output, output, decay_strength
            )

    def _forward_compiled(self, original_output: torch.Tensor, decay_strength: float) -> torch.Tensor:
        """Run the compiled dense path: h' = h + decay_strength * V_dense."""
        if self._compiled_compare:
            # Compute in float32 for accurate comparison
            return original_output.float() + decay_strength * self.compiled_vector.to(
                device=original_output.device
            )
        return original_output + decay_strength * self.compiled_vector.to(
            dtype=original_output.dtype, device=original_output.device
        )

    def _forward_naive(self, original_output: torch.Tensor, decay_strength: float) -> torch.Tensor:
        """Run the naive SAE encode-modify-decode path. Returns the final output tensor."""
        sae_model_on_device = self.sae_model.to(original_output.device)
        to_reconstruct = original_output.type(sae_model_on_device.W_enc.dtype)
        feature_space = sae_model_on_device.encode(to_reconstruct)

        # Compute SAE error if needed
        sae_error = None
        if self.error_coefficient > 0:
            clean_reconstruction = sae_model_on_device.decode(feature_space)
            sae_error = to_reconstruct - clean_reconstruction

        # Apply feature steering
        if hasattr(self, "mul_indices") and len(self.mul_indices) > 0:
            device = feature_space.device
            mul_indices = self.mul_indices.to(device=device, dtype=torch.long)
            mul_values = self.mul_values.to(device=device, dtype=self.mul_values.dtype)
            add_values = self.add_values.to(device=device, dtype=self.add_values.dtype)

            edits = feature_space[..., mul_indices]
            feature_space[..., mul_indices] = (
                edits * mul_values + add_values * decay_strength
            )

        reconstructed = sae_model_on_device.decode(feature_space)

        # Apply normalization if requested
        if self.normalization_mode == "before_error":
            orig_norm = original_output.norm(p=2, dim=-1, keepdim=True)
            recon_norm = reconstructed.norm(p=2, dim=-1, keepdim=True)
            recon_norm = recon_norm + (recon_norm == 0).float() * 1e-8
            scaling_factor = (
                1 - self.normalization_strength
            ) + self.normalization_strength * (orig_norm / recon_norm)
            reconstructed = reconstructed * scaling_factor

        # Add SAE error if specified
        if sae_error is not None:
            final_output = reconstructed + (sae_error * self.error_coefficient)
        else:
            final_output = reconstructed

        if self.normalization_mode == "after_error":
            orig_norm = original_output.norm(p=2, dim=-1, keepdim=True)
            final_norm = final_output.norm(p=2, dim=-1, keepdim=True)
            final_norm = final_norm + (final_norm == 0).float() * 1e-8
            scaling_factor = (
                1 - self.normalization_strength
            ) + self.normalization_strength * (orig_norm / final_norm)
            final_output = final_output * scaling_factor

        if self._compiled_compare:
            # Keep in float32 for accurate comparison
            return final_output
        return final_output.type(original_output.dtype)

    def _forward_naive_wrapped(self, original_output, output, decay_strength):
        """Run the naive path and rewrap output tuple if needed."""
        final_output = self._forward_naive(original_output, decay_strength)
        if isinstance(output, tuple):
            return (final_output,) + output[1:]
        return final_output

    def _record_compilation_error(
        self, compiled_output: torch.Tensor, naive_output: torch.Tensor
    ) -> None:
        """Record online error statistics between compiled and naive outputs.

        Tracks two levels:
        - Elementwise absolute error (over individual hidden dim values)
        - Tokenwise relative L2 error (per token position)

        Skips NaN/inf values to avoid poisoning accumulators.
        """
        # Use float64 on CUDA/CPU for maximum precision; fall back to float32 on MPS
        err_dtype = torch.float64 if compiled_output.device.type != "mps" else torch.float32
        diff = (compiled_output.to(err_dtype) - naive_output.to(err_dtype))
        abs_diff = diff.abs()

        # --- Elementwise absolute error stats ---
        flat_abs = abs_diff.reshape(-1)
        finite_mask = torch.isfinite(flat_abs)
        n_nonfinite = (~finite_mask).sum().item()
        self._nonfinite_count += n_nonfinite
        flat_abs = flat_abs[finite_mask]
        n_new = flat_abs.numel()

        if n_new == 0:
            return

        self._err_count_elem += n_new
        self._err_sum_abs += flat_abs.sum().item()
        self._err_sum_sq += (flat_abs * flat_abs).sum().item()

        batch_max = flat_abs.max().item()
        batch_min = flat_abs.min().item()
        if batch_max > self._err_max_abs:
            self._err_max_abs = batch_max
        if batch_min < self._err_min_abs:
            self._err_min_abs = batch_min

        # Welford online mean/variance update (elementwise)
        batch_mean = flat_abs.mean().item()
        delta = batch_mean - self._err_welford_mean
        self._err_welford_mean += delta * n_new / self._err_count_elem
        delta2 = batch_mean - self._err_welford_mean
        self._err_welford_M2 += delta * delta2 * n_new

        # --- Tokenwise relative L2 error ---
        # Flatten to [num_tokens, hidden]
        diff_2d = diff.reshape(-1, diff.shape[-1])
        naive_2d = naive_output.to(err_dtype).reshape(-1, naive_output.shape[-1])
        diff_norms = diff_2d.norm(p=2, dim=-1)
        naive_norms = naive_2d.norm(p=2, dim=-1)
        rel_errors = diff_norms / (naive_norms + 1e-12)

        # Filter non-finite relative errors
        rel_finite = torch.isfinite(rel_errors)
        rel_errors = rel_errors[rel_finite]

        n_tokens = rel_errors.numel()
        if n_tokens > 0:
            self._rel_err_count_tokens += n_tokens
            self._rel_err_sum += rel_errors.sum().item()
            rel_max = rel_errors.max().item()
            if rel_max > self._rel_err_max:
                self._rel_err_max = rel_max

        # --- Per-call error log for outlier identification ---
        call_idx = len(self._per_call_errors)
        call_entry = {
            "call_index": call_idx,
            "num_tokens": n_tokens,
            "mean_abs_error": flat_abs.mean().item(),
            "max_abs_error": batch_max,
        }
        if n_tokens > 0:
            call_entry["mean_rel_l2_error"] = rel_errors.mean().item()
            call_entry["max_rel_l2_error"] = rel_max
            worst_idx = rel_errors.argmax().item()
            call_entry["worst_token_position"] = worst_idx
            call_entry["worst_token_rel_l2"] = rel_errors[worst_idx].item()
        self._per_call_errors.append(call_entry)
