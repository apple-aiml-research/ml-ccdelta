# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import logging
import re
import typing as t
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> t.Union[t.List, t.Dict]:
    # Adding float resolver that includes "1e-3" like floats. Otherwise they are loaded as strings.
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path, "r") as infile:
        return yaml.load(infile, Loader=loader)


def load_json(path: Path) -> t.Union[t.List, t.Dict]:
    with open(path, "r") as infile:
        return json.load(infile)


def seed_all(seed=42):
    """Set all random seeds to a given value for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def is_module_name_in_regex(module_name: str, regex_list: t.List[str]) -> t.List[str]:
    """Returns True if module_name matches any of the regular expressions in the list

        The intended behavior is the following:

        - module_name="foo.bar", regex=".*bar" --> ["foo.bar"]
        - module_name="foo.bar", regex=".*bar:0" --> ["foo.bar:0"]
        - module_name="foo.bar", regex=".*bar:1" --> ["foo.bar:1"]
        - module_name="foo.bar:0", regex=".*bar" --> ["foo.bar:0"]
        - module_name="foo.bar:1", regex=".*bar" --> ["foo.bar:1"]
        - module_name="foo.bar:0", regex=".*bar:0" --> ["foo.bar:0"]
        - module_name="foo.bar:0", regex=".*bar:1" --> []

        The goal is to signal on which tensors we should create hooks.
        If any of the returned tensor names does not really exist, the flow will fail at hook creation/save.
        This might happen specially for the case:
         - module_name="foo.bar", regex=".*bar:1" --> ["foo.bar:1"]

    Args:
        module_name (str): name of pytorch module to find
        regex_list (t.List[str]): list with regex expressions

    Returns:
        list: the list of module names that match the expression
    """

    ret = []
    for regex in regex_list:
        # Just for the weird case that module_name has :num. Unlikely with current torch api.
        # In such case, we match the base part of the modulename if no specific :num is requested through regex.
        module_name_base = module_name
        if re.fullmatch(r".*(:[0-9]+)", module_name) is not None and ":" not in regex:
            module_name_base = module_name.split(":")[0]
        elif re.fullmatch(r".*(:[0-9]+)", module_name) is None and ":" in regex:
            # In case module_name does not contain :num but regex does, remove :num from regex
            regex, regex_num_tensor = regex.split(":")
            module_name = module_name + f":{regex_num_tensor}"
        match = re.fullmatch(regex, module_name_base)
        if match is not None:
            ret.append(module_name)
    ret = list(set(ret))
    return ret


def log_memory_usage(step_name: str) -> None:
    """Log CUDA memory usage for memory monitoring between major steps.

    Args:
        step_name: Name of the current step for logging context
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        logger.info(
            f"{step_name} - CUDA Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
        )


def cleanup_cuda_memory() -> None:
    """Clean up CUDA memory and synchronize GPU operations.

    This function:
    1. Forces garbage collection to free Python objects
    2. Clears CUDA cache to free GPU memory
    3. Synchronizes CUDA operations to ensure completion

    torch.cuda.synchronize() blocks until all queued CUDA operations complete.
    Without synchronization, torch.cuda.empty_cache() might not immediately free
    memory if operations are still pending.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
