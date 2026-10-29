# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Type definitions for the executor framework.

This module contains all the type aliases and protocols used by the executor system
to avoid circular imports between __init__.py and individual executor modules.
"""

from typing import Any, Iterable, Protocol

from ccdelta.models.model_with_hooks import ModelWithHooks

# TODO add structured object here so that prompts can carry mroe information
# e.g. prompt specific steering params
PromptBatch = list[str] | str

# Typically a pytorch dataloader or simple list
PromptLoader = Iterable[PromptBatch]

# Generation result from transformers pipeline
GenerationOutput = list[list[dict[str, str]]]


# Typically functools.partial of transformers pipeline
class GenerateFunction(Protocol):
    def __call__(self, prompts: PromptBatch, **kwargs: Any) -> GenerationOutput:
        """Generate text from prompts using transformers pipeline."""
        ...


# Tuple of [parameter_info, prompt_text, generated_text]
# TODO: Make parameter_info more generic. Probably a dict
ExecutorResult = list[float | int | str]


class ExecutorFunction(Protocol):
    """Protocol defining the signature for executor functions.

    All executor functions must accept these parameters and return a list of results
    where each result is a tuple containing (param_info, prompt, generation).
    """

    def __call__(
        self,
        model_hooks: ModelWithHooks,
        generate_fn: GenerateFunction,
        prompt_loader: PromptLoader,
        config: dict[str, Any],
        module_names: list[str],
    ) -> list[ExecutorResult]:
        """Execute parameter sweeping with the given configuration.

        Args:
            model_hooks: ModelWithHooks instance for hook management
            generate_fn: Text generation function (functools.partial of transformers pipeline)
                        Takes prompts and num_return_sequences, returns list[list[dict[str, str]]]
            prompt_loader: Iterator yielding batches of prompts
                          Can be PyTorch DataLoader or list of prompt batches
            config: Configuration dictionary containing:
                   - name: Hook name (e.g., "linear_ot", "sae_feature_space")
                   - hook_params: Parameters for hook creation
                   - state_path: Path to trained hook states (optional)
            module_names: list of module names to apply hooks to

        Returns:
            list of results, where each result is (param_info, prompt, generation)
            - For scalar_sweep: param_info is the parameter value (float/int)
            - For dict_sweep: param_info is the configuration identifier (str)
        """
        ...
