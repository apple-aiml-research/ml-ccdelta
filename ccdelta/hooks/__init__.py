# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
from omegaconf import OmegaConf

from .identity import IdentityHook
from .paired_responses_hook import PairedResponsesHook
from .postprocess_and_save_hook import PostprocessAndSaveHook
from .responses_hook import ResponsesHook
from .return_outputs_hook import ReturnOutputsHook
from .sae_hooks import SAEFeatureSpaceHook
from .sae_responses_hook import SAEResponsesHook
from .caa_hook import CAAHook
from .caa_context_conditioned_hook import CAAContextConditionedHook

HOOK_REGISTRY = {
    "postprocess_and_save": PostprocessAndSaveHook,
    "return_outputs": ReturnOutputsHook,
    "sae_feature_space": SAEFeatureSpaceHook,
    "sae_responses": SAEResponsesHook,
    "paired_sae_responses": PairedResponsesHook,
    "paired_dense_responses": PairedResponsesHook,
    "paired_sae_error_analysis": PairedResponsesHook,
    "identity": IdentityHook,
    "none": IdentityHook,
    "caa": CAAHook,
    "caa_context_conditioned": CAAContextConditionedHook,
}


def get_hook(name: str, *args, **kwargs) -> ResponsesHook:
    # Convert OmegaConf objects to plain Python objects
    # This is needed for complex nested structures like feature_masks
    converted_kwargs = {}
    for key, value in kwargs.items():
        if OmegaConf.is_config(value):
            converted_kwargs[key] = OmegaConf.to_object(value)
        else:
            converted_kwargs[key] = value
    kwargs = converted_kwargs

    hook_cls = HOOK_REGISTRY[name]
    hook = hook_cls(*args, **kwargs)
    return hook
