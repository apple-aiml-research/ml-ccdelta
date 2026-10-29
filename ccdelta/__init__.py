# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import hydra
from omegaconf import OmegaConf

# Register global OmegaConf resolvers that are used across multiple scripts
# Fix to cast dtypes in yaml config
OmegaConf.register_new_resolver(
    name="dtype", resolver=lambda dtype: hydra.utils.get_object(dtype)
)
