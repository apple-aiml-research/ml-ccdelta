# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""SAE Responses Hook for collecting SAE feature activations."""

import typing as t
from pathlib import Path

import torch

from ccdelta.hooks.postprocess_and_save_hook import PostprocessAndSaveHook
from ccdelta.utils.sae import load_sae_model_for_hook

from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)


class SAEResponsesHook(PostprocessAndSaveHook):
    """
    SAE Responses Hook that captures and saves SAE feature activations.

    This hook extends PostprocessAndSaveHook to apply SAE encoding to model
    activations before saving them. This allows learning interventions from
    SAE feature space rather than raw activation space.

    Parameters:
        module_name : str
            Name of the parent module whose outputs are being hooked to.
        sae_config : dict
            sae configuration
        pooling_op_names : list[str]
            List of pooling operation names to be applied on the SAE features.
        output_path : Path
            The location where the SAE features should be saved.
        save_fields : list[str]
            Fields of interest to be stored along with SAE features.
        device : str, optional
            Device to load SAE model on. Defaults to "cuda" if available, else "cpu".
        dtype : torch.dtype, optional
            Data type for SAE model. Defaults to torch.float32.
        cache_dir : str, optional
            Cache directory for SAE model downloads.
        skip_start_token : bool, optional
            Whether to skip encoding the start token (position 0). Many SAEs are trained
            without the start token. Defaults to True.
        **kwargs
            Additional arguments passed to parent class.
    """

    def __init__(
        self,
        module_name: str,
        sae_config: dict,
        pooling_op_names: t.List[str],
        output_path: t.Optional[Path],
        save_fields: t.List[str],
        device: t.Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        cache_dir: t.Optional[str] = None,
        skip_start_token: bool = True,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            pooling_op_names=pooling_op_names,
            output_path=output_path,
            save_fields=save_fields,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.skip_start_token = skip_start_token

        self.sae_config = sae_config
        self.sae_model = None
        if self.sae_config is not None:
            self.sae_model = load_sae_model_for_hook(
                config=self.sae_config,
                device=self.device,
                dtype=self.dtype,
                cache_dir=self.cache_dir,
            )

    def save(self, module_name: str, output: torch.Tensor) -> None:
        """
        Apply SAE encoding to activations and save SAE features.

        This method overrides the parent save method to apply SAE encoding
        to the raw activations before applying pooling and saving.

        Parameters:
            module_name : str
                Name of the parent module whose outputs are being processed.
            output : torch.Tensor
                Raw activations from the model layer.
        """
        if self.sae_model is None:
            logger.warning("SAE model not loaded, falling back to raw activations")
            super().save(module_name, output)
            return

        # Ensure output is on the same device as SAE model
        output = output.to(
            device=self.sae_model.W_enc.device, dtype=self.sae_model.W_enc.dtype
        )
        original_shape = output.shape
        original_attention_mask = self.batch.get("attention_mask")

        # Handle start token skipping if requested
        if self.skip_start_token:
            if len(output.shape) == 3 and output.shape[1] > 1:
                # Skip start token: only encode positions 1 onwards
                sae_features = self.sae_model.encode(output[:, 1:, :])
                if original_attention_mask is not None:
                    self.batch["attention_mask"] = original_attention_mask[:, 1:]
            else:
                # If not 3D or single token, encode normally
                sae_features = self.sae_model.encode(output)
        else:
            # Apply SAE encoding to all positions
            sae_features = self.sae_model.encode(output)

        # Convert back to original dtype for saving
        sae_features = sae_features.to(dtype=output.dtype)

        logger.debug(f"Encoded {module_name}: {original_shape} -> {sae_features.shape}")

        try:
            super().save(f"{module_name}", sae_features)
        finally:
            # Restore the original attention mask
            if original_attention_mask is not None:
                self.batch["attention_mask"] = original_attention_mask

    def __str__(self):
        txt = (
            f"SAEResponsesHook(module_name={self.module_name}, "
            f"sae_config={self.sae_config}, "
            f"skip_start_token={self.skip_start_token}, "
            f"pooling_ops={self.pooling_ops}, "
            f"output_path={self.output_path})\n"
        )
        return txt
