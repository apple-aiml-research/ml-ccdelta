# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import os
import typing as t
from functools import partial
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)

EXTRA_KWARGS = {
    "google/gemma-2-2b": {"attn_implementation": "eager"},
    "google/gemma-2-9b": {"attn_implementation": "eager"},
}


def get_model(
    model_path: t.Union[str, Path],
    cache_dir: t.Optional[t.Union[str, Path]],
    dtype: t.Any = None,
    device: t.Any = None,
    model_task: str = None,
    **kwargs,
):
    """Loads a pre-trained model based on the specified task.

    This function acts as a dispatcher, selecting the appropriate
    model loading function from `TASK_MAPPING` based on the provided `model_task`.

    Args:
        model_path (Union[str, Path]): The path to the pre-trained model weights.
        cache_dir (Optional[Union[str, Path]], optional): Directory to cache downloaded models. Defaults to None.
        dtype (Any, optional): Data type for the model weights. Defaults to None (inferred from the model).
        device (Any, optional): Device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None (uses default device).
        model_task (str): The task the model is intended for (e.g., "text-classification", "text-generation").
        **kwargs: Additional keyword arguments passed to the specific model loading function.

    Returns:
        Any: The loaded pre-trained model.

    Raises:
        AssertionError: If `model_task` is not provided.
    """
    assert model_task is not None
    return TASK_MAPPING[model_task](
        model_path=model_path,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        **kwargs,
    )


def get_huggingface_model(
    model_path: t.Union[str, Path],
    dtype: t.Any = None,
    device: str = None,
    seq_len: t.Optional[int] = None,
    rand_weights: bool = False,
    model_class: t.Optional[str] = AutoModel,
    padding_side: str = "left",
    **kwargs,
) -> t.Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a Huggingface model and tokenizer.

    This function downloads and loads a pretrained model from the Huggingface Model Hub.
    It supports loading various types of models (e.g., text generation, sequence classification)
    and allows for customization of device, data type, sequence length, and padding side.

    Args:
        model_path (str or Path): The path to the model on the Huggingface Model Hub.
        cache_dir (str or Path, optional): The directory where cached models are stored.
                                          Defaults to using the default Huggingface cache if not specified.
        dtype (Any, optional): The desired data type for the model tensors. Defaults to the default dtype of the current device.
        device (str, optional): The device to load the model on ("cuda" or "cpu"). Defaults to "cuda:0" if available, otherwise "cpu".
        seq_len (int, optional): The maximum sequence length for the tokenizer. Defaults to None.
        rand_weights (bool, optional): If True, initializes a new model with random weights instead of loading pretrained weights.
                                     Defaults to False.
        model_class (str, optional): The class of the Huggingface model to load. Defaults to AutoModel.
        padding_side (str, optional): Specifies which side to pad tokens on ("left" or "right"). Defaults to "left".
        **kwargs: Additional keyword arguments passed to the model loading function.

    Returns:
        tuple: A tuple containing the loaded Huggingface model and tokenizer.

    """

    # If HF_HUB_CACHE is set, use it. Otherwise, use the default Huggingface one.
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    # Also fetching your huggingface token for specific model downloads.
    hf_token = os.environ.get("HF_TOKEN", None)

    # Defaults for device and dtype
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.get_default_dtype()
    else:
        dtype = dtype

    if cache_dir is not None:
        cache_dir = Path(cache_dir).absolute()
        full_model_path = cache_dir / model_path
        if full_model_path.exists():
            model_path = full_model_path

    if rand_weights:
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device
        )
        model = model_class.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        extra_kwargs = EXTRA_KWARGS.get(model_path, {})

        model_path = str(model_path)
        tokenizer_path = model_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            device_map=device,
            dtype=dtype,
            force_download=False,
            token=hf_token,
            **extra_kwargs,
        )
        model = model_class.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map=device,
            force_download=False,
            torch_dtype=dtype,
            token=hf_token,
            **extra_kwargs,
        )

    if seq_len:
        tokenizer.model_max_length = seq_len
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        if tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = "<pad>"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # stick a reference to the tokenier onto here to make getting it elsewhere easy.
    model.tokenizer = tokenizer

    return model, tokenizer


TASK_MAPPING = {
    "text-generation": partial(get_huggingface_model, model_class=AutoModelForCausalLM)
}
