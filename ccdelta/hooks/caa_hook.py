# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
import logging
import typing as t

import torch

from .intervention_hook import InterventionHook

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CAAHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
            **kwargs,
        )
        self.strength = float(strength)
        self.register_buffer("vector", torch.empty(0))
        # print(self)

    def __str__(self):
        txt = f"CAA(module_name={self.module_name} strength={self.strength})"
        return txt

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
    ):
        self.vector = state_dict["vector"].to(self.device).to(self.dtype)
        self._post_load()

    def fit(
        self,
        responses: torch.Tensor,
        labels=torch.Tensor,
        **kwargs,
    ) -> None:
        z_f64 = responses.to(torch.float64)
        # 1 is source
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.vector = self.mu2 - self.mu1
        self._post_load()

    def forward(self, module, input, output) -> t.Any:
        if output.ndim == 4:
            raise DeprecationWarning(
                "InterventionHook already takes care of formatting the input to dims"
            )

        return output + self.strength * self.vector
