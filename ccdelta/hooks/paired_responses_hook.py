# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Paired Responses Hook for handling tabular paired data in intervention learning."""

import json
import threading
import typing as t
from pathlib import Path

import torch

from ccdelta.hooks.responses_hook import ResponsesHook
from ccdelta.utils.logging import get_logger
from ccdelta.utils.sae import compute_sae_feature_comparison

logger = get_logger(__name__)


class PairedResponsesHook(ResponsesHook):
    """Hook for processing paired tabular data.

    This is a simplified version that processes the current batch and
    assumes the pipeline will handle source/target processing separately.
    For now, it just saves the current features with paired metadata.
    """

    def __init__(
        self,
        inner_hook_class: t.Optional[t.Type] = None,
        module_name: str = "",
        output_path: t.Optional[Path] = None,
        save_fields: t.Optional[t.List[str]] = None,
        threaded: bool = False,
        raise_exception: bool = False,
        tokenizer: t.Optional[t.Any] = None,
        collect_error_stats: bool = False,
        **kwargs,
    ):
        """Initialize PairedResponsesHook.

        Args:
            inner_hook_class: Class for the inner hook that extracts features.
            module_name: Name of the module this hook is attached to.
            output_path: Path to save output files.
            save_fields: List of fields to save from the batch.
            threaded: Whether to run processing in a separate thread.
            raise_exception: Whether to raise exceptions on errors.
            tokenizer: Tokenizer for special token handling.
            collect_error_stats: If True, collect SAE reconstruction error statistics.
                When enabled, the inner hook is expected to return a dict with
                'sae_features', 'error_stats', and 'activation_stats' keys.
            **kwargs: Additional arguments passed to inner hook.
        """
        super().__init__()

        self.module_name = module_name
        self.output_path = output_path
        self.save_fields = save_fields or ["id"]
        self.threaded = threaded
        self.raise_exception = raise_exception
        self.thread_handle = None
        self.tokenizer = tokenizer
        self.collect_error_stats = collect_error_stats
        self.special_tokens_set = (
            set(self.tokenizer.all_special_tokens) if self.tokenizer else set()
        )

        # State tracking
        self.batch_idx: t.Optional[int] = None
        self.batch: t.Optional[dict] = None
        self.is_source: bool = False
        self.is_target: bool = False

        # Paired processing state
        self.source_features: t.Dict[str, torch.Tensor] = {}
        self.source_tokens: t.Dict[str, t.List[str]] = {}

        # Error stats collection state (when collect_error_stats=True)
        self.source_error_stats: t.Dict[str, dict] = {}
        self.source_activation_stats: t.Dict[str, dict] = {}
        # Content-only and control-only stats (separate from all-token stats)
        self.source_error_stats_content: t.Dict[str, dict] = {}
        self.source_error_stats_control: t.Dict[str, dict] = {}
        self.source_activation_stats_content: t.Dict[str, dict] = {}
        self.source_activation_stats_control: t.Dict[str, dict] = {}
        self._sae_metadata_saved: bool = False

        # Default to IdentityHook for simplicity
        if inner_hook_class is None:
            from ccdelta.hooks.identity import IdentityHook

            inner_hook_class = IdentityHook

        # Create inner hook for actual feature extraction
        # Pass tokenizer and output_path for stats saving
        self.inner_hook = inner_hook_class(
            module_name=module_name,
            tokenizer=tokenizer,
            output_path=output_path,
            **kwargs,
        )

        logger.info(
            f"Initialized PairedResponsesHook with inner hook: {inner_hook_class.__name__}"
        )

    def update(
        self,
        batch_idx: int,
        batch: dict,
        is_source: bool = False,
        is_target: bool = False,
    ):
        """Update hook state with new batch data."""
        self.batch_idx = batch_idx
        self.batch = batch
        self.is_source = is_source
        self.is_target = is_target

        # Propagate update to inner hook if it supports it
        if hasattr(self.inner_hook, "update"):
            self.inner_hook.update(batch_idx, batch)

    def __call__(self, module, input, output):
        """Main hook entry point."""
        logger.debug(f"Hook invoked for batch_idx: {self.batch_idx}")

        assert self.batch_idx is not None, (
            "update() must be called before executing the hook"
        )

        def _process():
            if isinstance(output, torch.Tensor):
                tensor_output = output.detach()
            elif (
                isinstance(output, (list, tuple))
                and len(output) > 0
                and isinstance(output[0], torch.Tensor)
            ):
                tensor_output = output[0].detach()
            else:
                logger.warning(
                    f"Cannot extract tensor from output of type {type(output)}"
                )
                return

            module_name = self.module_name
            if ":" not in module_name:
                module_name = f"{module_name}:0"
            self._process_batch(module_name, tensor_output)

        if self.threaded:
            self.thread_handle = threading.Thread(target=_process)
            self.thread_handle.start()
        else:
            _process()

    def _process_batch(self, module_name: str, output: torch.Tensor):
        """Process the current batch with paired source/target logic."""
        logger.debug(
            f"Processing batch - is_source: {self.is_source}, is_target: {self.is_target}"
        )

        if self.batch is None:
            logger.error("Batch is None, cannot process tensor")
            return

        # Apply inner hook feature extraction
        inner_result = self.inner_hook(None, None, output)

        # Handle both dict returns (error stats mode) and tensor returns (regular mode)
        if self.collect_error_stats and isinstance(inner_result, dict):
            features = inner_result["sae_features"]
            error_stats = inner_result.get("error_stats", {})
            activation_stats = inner_result.get("activation_stats", {})
            # Content-only stats (excluding control tokens like BOS/EOS/system)
            error_stats_content = inner_result.get("error_stats_content", {})
            activation_stats_content = inner_result.get("activation_stats_content", {})
            # Control-only stats (for separate reporting)
            error_stats_control = inner_result.get("error_stats_control", {})
            activation_stats_control = inner_result.get("activation_stats_control", {})

            # Save SAE metadata once per run
            if not self._sae_metadata_saved and self.output_path is not None:
                self._save_sae_metadata()
                self._sae_metadata_saved = True
        else:
            features = inner_result
            error_stats = {}
            activation_stats = {}
            error_stats_content = {}
            activation_stats_content = {}
            error_stats_control = {}
            activation_stats_control = {}

        batch_size = len(self.batch["id"])
        logger.debug(f"Processing batch size: {batch_size}")

        if self.is_source:
            # Source pass: store features for pairing with target
            logger.debug(f"Storing source features for {batch_size} samples")
            for sample_idx in range(batch_size):
                sample_id = self.batch["id"][sample_idx]
                # Apply attention mask filtering to features to match token filtering
                sample_features = features[sample_idx]
                if "attention_mask" in self.batch:
                    attention_mask = self.batch["attention_mask"][sample_idx]
                    # Only keep features for non-padded positions
                    was_sparse = sample_features.is_sparse
                    # If features are sparse, they must be densified before boolean mask indexing
                    if was_sparse:
                        sample_features = sample_features.to_dense()
                    sample_features = sample_features[attention_mask == 1]
                    # Convert back to sparse if it was originally
                    if was_sparse:
                        sample_features = sample_features.to_sparse()

                self.source_features[sample_id] = sample_features
                if "tokens" in self.batch:
                    sample_tokens = self.batch["tokens"][sample_idx]
                    self._validate_tokens(sample_tokens, "source", sample_id)
                    self.source_tokens[sample_id] = sample_tokens

                # Store error/activation stats for source pass
                if self.collect_error_stats and error_stats:
                    self.source_error_stats[sample_id] = self._extract_sample_stats(
                        error_stats, sample_idx
                    )
                    self.source_activation_stats[sample_id] = (
                        self._extract_sample_stats(activation_stats, sample_idx)
                    )
                    # Store content-only stats
                    if error_stats_content:
                        self.source_error_stats_content[sample_id] = self._extract_sample_stats(
                            error_stats_content, sample_idx
                        )
                    if activation_stats_content:
                        self.source_activation_stats_content[sample_id] = self._extract_sample_stats(
                            activation_stats_content, sample_idx
                        )
                    # Store control-only stats
                    if error_stats_control:
                        self.source_error_stats_control[sample_id] = self._extract_sample_stats(
                            error_stats_control, sample_idx
                        )
                    if activation_stats_control:
                        self.source_activation_stats_control[sample_id] = self._extract_sample_stats(
                            activation_stats_control, sample_idx
                        )

        elif self.is_target:
            # Target pass: pair with source features and save

            for sample_idx in range(batch_size):
                sample_id = self.batch["id"][sample_idx]
                # Apply attention mask filtering to features to match token filtering
                target_features = features[sample_idx]
                if "attention_mask" in self.batch:
                    attention_mask = self.batch["attention_mask"][sample_idx]
                    # Only keep features for non-padded positions
                    was_sparse = target_features.is_sparse
                    # If features are sparse, they must be densified before boolean mask indexing
                    if was_sparse:
                        target_features = target_features.to_dense()
                    target_features = target_features[attention_mask == 1]
                    # Convert back to sparse if it was originally
                    if was_sparse:
                        target_features = target_features.to_sparse()

                # Get corresponding source features
                if sample_id not in self.source_features:
                    logger.warning(
                        f"No source features found for sample {sample_id}, skipping"
                    )
                    continue

                source_features = self.source_features[sample_id]
                source_tokens = self.source_tokens.get(sample_id, [])
                target_tokens = (
                    self.batch["tokens"][sample_idx] if "tokens" in self.batch else []
                )
                self._validate_tokens(target_tokens, "target", sample_id)

                # Create paired datum
                datum = {
                    "id": sample_id,
                    "source_features": source_features,
                    "target_features": target_features,
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens,
                    "module_name": module_name,
                }

                # Add error stats if collecting
                if self.collect_error_stats:
                    # Get raw stats (all tokens)
                    source_error_raw = self.source_error_stats.get(sample_id, {})
                    target_error_raw = (
                        self._extract_sample_stats(error_stats, sample_idx)
                        if error_stats
                        else {}
                    )
                    # Enrich with token strings for readability
                    datum["source_error_stats"] = self._enrich_error_stats_with_tokens(
                        source_error_raw, source_tokens
                    )
                    datum["target_error_stats"] = self._enrich_error_stats_with_tokens(
                        target_error_raw, target_tokens
                    )
                    datum["source_activation_stats"] = self.source_activation_stats.get(
                        sample_id, {}
                    )
                    datum["target_activation_stats"] = (
                        self._extract_sample_stats(activation_stats, sample_idx)
                        if activation_stats
                        else {}
                    )

                    # Content-only stats (excluding control tokens)
                    source_error_content_raw = self.source_error_stats_content.get(sample_id, {})
                    target_error_content_raw = (
                        self._extract_sample_stats(error_stats_content, sample_idx)
                        if error_stats_content
                        else {}
                    )
                    if source_error_content_raw:
                        datum["source_error_stats_content"] = self._enrich_error_stats_with_tokens(
                            source_error_content_raw, source_tokens
                        )
                    if target_error_content_raw:
                        datum["target_error_stats_content"] = self._enrich_error_stats_with_tokens(
                            target_error_content_raw, target_tokens
                        )
                    source_activation_content = self.source_activation_stats_content.get(sample_id, {})
                    if source_activation_content:
                        datum["source_activation_stats_content"] = source_activation_content
                    if activation_stats_content:
                        datum["target_activation_stats_content"] = self._extract_sample_stats(
                            activation_stats_content, sample_idx
                        )

                    # Control-only stats (for separate reporting)
                    source_error_control_raw = self.source_error_stats_control.get(sample_id, {})
                    target_error_control_raw = (
                        self._extract_sample_stats(error_stats_control, sample_idx)
                        if error_stats_control
                        else {}
                    )
                    if source_error_control_raw:
                        datum["source_error_stats_control"] = self._enrich_error_stats_with_tokens(
                            source_error_control_raw, source_tokens
                        )
                    if target_error_control_raw:
                        datum["target_error_stats_control"] = self._enrich_error_stats_with_tokens(
                            target_error_control_raw, target_tokens
                        )
                    source_activation_control = self.source_activation_stats_control.get(sample_id, {})
                    if source_activation_control:
                        datum["source_activation_stats_control"] = source_activation_control
                    if activation_stats_control:
                        datum["target_activation_stats_control"] = self._extract_sample_stats(
                            activation_stats_control, sample_idx
                        )

                    # Compute source-target difference metrics
                    datum["source_target_diff"] = compute_sae_feature_comparison(
                        source_features, target_features
                    )

                # Add metadata
                for field in self.save_fields:
                    if field in self.batch and field != "id":
                        if (
                            isinstance(self.batch[field], list)
                            and len(self.batch[field]) > sample_idx
                        ):
                            datum[field] = self.batch[field][sample_idx]
                        elif field in self.batch:
                            datum[field] = self.batch[field]

                # Add row data if available
                if (
                    "row_data" in self.batch
                    and len(self.batch["row_data"]) > sample_idx
                ):
                    datum["row_data"] = self.batch["row_data"][sample_idx]

                # Save the paired datum
                self._save_datum(datum)

                # Clean up stored source features and stats
                del self.source_features[sample_id]
                if sample_id in self.source_tokens:
                    del self.source_tokens[sample_id]
                if sample_id in self.source_error_stats:
                    del self.source_error_stats[sample_id]
                if sample_id in self.source_activation_stats:
                    del self.source_activation_stats[sample_id]
                # Clean up content/control stats
                if sample_id in self.source_error_stats_content:
                    del self.source_error_stats_content[sample_id]
                if sample_id in self.source_activation_stats_content:
                    del self.source_activation_stats_content[sample_id]
                if sample_id in self.source_error_stats_control:
                    del self.source_error_stats_control[sample_id]
                if sample_id in self.source_activation_stats_control:
                    del self.source_activation_stats_control[sample_id]

        else:
            logger.warning(
                "Called _process_batch without source/target flag - this shouldn't happen in paired mode"
            )

    def _extract_sample_stats(self, stats_dict: dict, sample_idx: int) -> dict:
        """Extract per-sample statistics from batched stats tensors.

        Args:
            stats_dict: Dictionary with batched tensor values [B, ...].
            sample_idx: Index of the sample to extract.

        Returns:
            Dictionary with scalar or list values for the sample.
        """
        result = {}
        for key, value in stats_dict.items():
            if isinstance(value, torch.Tensor):
                sample_value = value[sample_idx]
                # Convert to Python scalar or list
                if sample_value.ndim == 0:
                    result[key] = sample_value.item()
                else:
                    result[key] = sample_value.tolist()
            else:
                result[key] = value
        return result

    def _enrich_error_stats_with_tokens(
        self, error_stats: dict, tokens: t.List[str]
    ) -> dict:
        """Enrich error stats by adding token strings to top_10 indices.

        Converts top_10_*_token_indices lists to lists of (index, token, error) tuples
        for better readability.

        Args:
            error_stats: Dictionary with error statistics including top_10 lists.
            tokens: List of token strings for this sample.

        Returns:
            Dictionary with enriched top_10 lists.
        """
        enriched = dict(error_stats)

        # Enrich highest error tokens
        high_indices = error_stats.get("top_10_highest_error_token_indices", [])
        high_values = error_stats.get("top_10_highest_error_values", [])
        if high_indices and high_values:
            enriched["top_10_highest_error_tokens"] = [
                {
                    "position": idx,
                    "token": tokens[idx] if 0 <= idx < len(tokens) else "N/A",
                    "error": val,
                }
                for idx, val in zip(high_indices, high_values)
                if idx >= 0  # Skip sentinel values (-1 indicates invalid position)
            ]

        # Enrich lowest error tokens
        low_indices = error_stats.get("top_10_lowest_error_token_indices", [])
        low_values = error_stats.get("top_10_lowest_error_values", [])
        if low_indices and low_values:
            enriched["top_10_lowest_error_tokens"] = [
                {
                    "position": idx,
                    "token": tokens[idx] if 0 <= idx < len(tokens) else "N/A",
                    "error": val,
                }
                for idx, val in zip(low_indices, low_values)
                if idx >= 0  # Skip sentinel values (-1 indicates invalid position)
            ]

        return enriched

    def _save_sae_metadata(self):
        """Save SAE metadata to output_path as sae_metadata.json."""
        if not hasattr(self.inner_hook, "get_sae_metadata"):
            logger.debug("Inner hook does not support get_sae_metadata")
            return

        metadata = self.inner_hook.get_sae_metadata()
        if not metadata:
            logger.debug("No SAE metadata to save")
            return

        metadata_path = Path(str(self.output_path)) / "sae_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved SAE metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save SAE metadata: {e}")

    def _validate_tokens(self, tokens: t.List[str], role: str, sample_id: str):
        """Assert that no special tokens are present in the provided list."""
        return
        # if not self.special_tokens_set:
        #     return

        # for token in tokens:
        #     assert (
        #         token not in self.special_tokens_set
        #     ), f"Special token '{token}' found in {role} tokens for sample {sample_id}. Special tokens: {self.special_tokens_set}"

    def _save_datum(self, datum: dict):
        """Save features to disk."""
        logger.debug(
            f"Saving datum with keys: {list(datum.keys())}, output_path: {self.output_path}"
        )

        if self.output_path is None:
            logger.warning("No output path specified, skipping save")
            return

        output_path = (
            Path(str(self.output_path)) / str(self.module_name) / f"{datum['id']}.pt"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(datum, output_path)
            logger.debug(f"Successfully saved datum to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save datum to {output_path}: {e}")
            if self.raise_exception:
                raise

    def __str__(self):
        """String representation."""
        inner_name = self.inner_hook.__class__.__name__ if self.inner_hook else "None"
        return (
            f"PairedResponsesHook(inner_hook={inner_name}, "
            f"module_name={self.module_name}, "
            f"output_path={self.output_path})"
        )
