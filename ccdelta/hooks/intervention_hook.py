# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import abc
import typing as t
from pathlib import Path
from typing import Optional
import re

from logging import getLogger
import torch
import torch.nn.functional as F

logger = getLogger(__file__)


class InterventionHook(torch.nn.Module):
    """
    Abstract base class for a hook that intervenes during the forward pass of a PyTorch module.

    This class allows you to specify which part of the output tensor (or all outputs) to modify and at what point in the computation this modification should occur.

    Args:
        module_name (str): The name of the module or layer where the intervention is needed. If a specific output index is required, it can be specified after a colon (e.g., "module_name:output_index").
        intervention_position (str): Specifies when to intervene in the forward pass. 'all' means at every step, 'last' means only on the last element of the output tensor sequence, 'prompt' means only on multi-token prompts (not single-token autoregressive generation).
        dtype (torch.dtype): The desired data type for the intervention. Default is torch.float32.
    """

    def __init__(
        self,
        module_name: str,
        device: str,
        intervention_position: t.Literal["all", "last", "prompt"] = "all",
        dtype: torch.dtype = torch.float32,
        skip_special_token_filtering: bool = True,
    ):
        super().__init__()
        self.module_name = module_name
        self.device = device
        if ":" in module_name:
            self.select_output = int(module_name.split(":")[1])
            self.is_residual = bool(
                re.fullmatch(r".*layers\.\d+$", module_name.split(":")[0])
            )
        else:
            self.select_output = None
            self.is_residual = bool(re.fullmatch(r".*layers\.\d+$", module_name))
        self.intervention_position = intervention_position
        self.dtype = dtype
        self.skip_special_token_filtering = skip_special_token_filtering

        # generation context, will be filled in during hook registration
        # carries token level information about the current forward pass.
        self.gen_ctx = None

    def register_named_buffers(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.register_buffer(k, v.to(self.dtype))

    def save_state_dict(self, state_path: Path) -> None:
        logger.info("Saving state dict to %s", state_path)
        torch.save(self.state_dict(), state_path)

    def from_state_path(self, state_path: Path) -> None:
        """
        Loads intervention state from a state path pointing to a torch-saved state_dict.

        :param state_path: The state path to load.
        """
        self.load_state_dict(torch.load(state_path, weights_only=True))

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Method fit() must be implemented.")

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().

        Re-implement as needed, but do not forget to call super()._post_load() in the implementation.
        """
        # Check all buffers are duly initialized.
        for buffer_name, buffer in self.named_buffers():
            assert buffer.numel() > 0, f"Buffer {buffer_name} has not been initialized."

    def update(self, *args, **kwargs):
        """
        Updates the state or arguments of this hook with new input data at runtime.

        This method can be overridden by subclasses to provide custom updating logic. By default, it does nothing and returns None.

        Parameters:
            *args : variable-length argument list
                Variable length argument list that will be used as is for the update operation.

            **kwargs : keyworded arguments
                Keyworded arguments that can also be used to update state or arguments of this hook.

        Returns:
            None
        """
        return None

    def _is_prompt_phase(self, input, output):
        """Detect if we're in prompt processing phase (vs autoregressive generation)."""
        # Check for past_key_values (HuggingFace models)
        if len(input) > 2 and input[2] is not None:
            return False

        # Fallback: Check sequence length
        tensor_to_check = output[0] if isinstance(output, tuple) else output

        if tensor_to_check.ndim >= 2:
            seq_len = tensor_to_check.shape[-2]
            return seq_len > 1
        return True

    @staticmethod
    def _unwrap(output):
        """Unwrap tuple output if present."""
        return (output[0], True) if isinstance(output, tuple) else (output, False)

    @staticmethod
    def _rewrap(tensor, was_tuple, orig_output):
        """Rewrap tensor back to original format."""
        return (tensor,) + orig_output[1:] if was_tuple else tensor

    def _extract_attention_mask(self, input_, batch):
        """Extract attention mask from input[1] if available (HuggingFace convention)."""
        if len(input_) > 1 and isinstance(input_[1], torch.Tensor):
            candidate = input_[1]
            if candidate.ndim == 2 and candidate.shape[0] == batch:
                return candidate
        return None

    @staticmethod
    def _compute_steer_mask_for_call(
        gen_ctx,
        batch_dim: int,
        seq_dim: int,
        device: torch.device,
        where: str,
        skip_special_token_filtering: bool,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns [B,S] bool: True = steer (non-special), False = pass-through.

        Args:
            gen_ctx: Generation context with special_mask_cur and is_decode_call
            batch_dim: Batch size
            seq_dim: Sequence length
            device: Device for tensors
            where: Intervention position ("all", "prompt", or "last")
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

    def __call__(self, module, input_, output):
        """
        PyTorch call method overridden to implement the intervention logic.

        This method supports two modes based on skip_special_token_filtering:
        - skip_special_token_filtering=True (default): Legacy BOS slicing for backward compatibility
        - skip_special_token_filtering=False: Sophisticated semantic token masking

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (tuple or torch.Tensor): Output of the module's forward function.

        Returns:
            The modified output after intervention.
        """
        if self.skip_special_token_filtering:
            # LEGACY PATH: Use existing BOS slicing behavior for backward compatibility
            is_multi_token = self._is_prompt_phase(input_, output)

            # 1. For "prompt" position, we only intervene on multi-token sequences.
            if self.intervention_position == "prompt" and not is_multi_token:
                return output

            original_output_tensor = output[0] if isinstance(output, tuple) else output
            # 2. Else (we are not using "prompt") and it's not a multi-token sequence proceed as normal
            if not is_multi_token:
                final_output_tensor = self.forward(
                    module, input_, original_output_tensor
                )
            else:
                # 3. If we are intervening on a multi-token sequence, we want to skip the BOS token similar to SAE to be comparable
                bos_activation = original_output_tensor[:, :1, ...]
                activations_to_steer = original_output_tensor[:, 1:, ...]

                temp_output_for_forward = activations_to_steer

                # Steer only the non-BOS tokens
                steered_output = self.forward(module, input_, temp_output_for_forward)
                steered_activations_tensor = (
                    steered_output[0]
                    if isinstance(steered_output, tuple)
                    else steered_output
                )

                # Re-combine the BOS token with the steered activations
                final_output_tensor = torch.cat(
                    [bos_activation, steered_activations_tensor], dim=1
                )

            if isinstance(output, tuple):
                return (final_output_tensor,) + output[1:]
            else:
                return final_output_tensor

        else:
            out_tensor, was_tuple = self._unwrap(output)

            # Handle nested tuple case - keep unwrapping until we get a tensor
            while isinstance(out_tensor, tuple):
                out_tensor = out_tensor[0]

            # Ensure we have a tensor
            if not isinstance(out_tensor, torch.Tensor):
                # Fall back to legacy behavior
                is_multi_token = self._is_prompt_phase(input_, output)
                if self.intervention_position == "prompt" and not is_multi_token:
                    return output
                original_output_tensor = (
                    output[0] if isinstance(output, tuple) else output
                )
                if not is_multi_token:
                    final_output_tensor = self.forward(
                        module, input_, original_output_tensor
                    )
                else:
                    bos_activation = original_output_tensor[:, :1, ...]
                    activations_to_steer = original_output_tensor[:, 1:, ...]
                    steered_output = self.forward(module, input_, activations_to_steer)
                    steered_activations_tensor = (
                        steered_output[0]
                        if isinstance(steered_output, tuple)
                        else steered_output
                    )
                    final_output_tensor = torch.cat(
                        [bos_activation, steered_activations_tensor], dim=1
                    )
                if isinstance(output, tuple):
                    return (final_output_tensor,) + output[1:]
                else:
                    return final_output_tensor

            # Early exit for low-dimensional tensors
            if out_tensor.dim() < 3:
                return output

            batch, seq, _ = out_tensor.shape

            # Extract attention mask from input[1] if available
            attention_mask = self._extract_attention_mask(input_, batch)

            # Compute steer mask (handles special tokens, phase gating, attention mask)
            steer_mask = self._compute_steer_mask_for_call(
                self.gen_ctx,
                batch,
                seq,
                out_tensor.device,
                where=self.intervention_position,
                skip_special_token_filtering=False,  # Always use filtering in this path
                attention_mask=attention_mask,
            )

            # Fast path: no steering needed
            if not steer_mask.any().item():
                return output

            # Fast path: steer all tokens (no special tokens present)
            if steer_mask.all().item():
                # Pass unwrapped tensor to forward(), not the tuple
                steered = self.forward(module, input_, out_tensor)
                return self._rewrap(steered, was_tuple, output)

            # Mixed case: Apply steering to full tensor, then selectively keep results

            # Apply steering to ALL tokens (including special ones)
            # Pass unwrapped tensor to forward(), not the tuple
            steered_output = self.forward(module, input_, out_tensor)
            steered_tensor = (
                steered_output[0]
                if isinstance(steered_output, tuple)
                else steered_output
            )

            # Selectively apply: steered for non-special, original for special tokens
            # steer_mask is [B,S], need to expand to [B,S,H] for element-wise selection
            final = torch.where(
                steer_mask.unsqueeze(-1),  # [B,S,1] - broadcast across hidden dim
                steered_tensor,  # Use steered values where mask is True
                out_tensor,  # Use original values where mask is False (special tokens)
            )

            return self._rewrap(final, was_tuple, output)

    @abc.abstractmethod
    def forward(self, module, input_, output):
        """
        Abstract method to be implemented by subclasses. This method defines the logic for how the intervention should modify the output.

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (torch.Tensor): Output tensor of the module's forward function, modified according to the intervention logic.

        Returns:
            A modified version of the output tensor after applying the intervention logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
