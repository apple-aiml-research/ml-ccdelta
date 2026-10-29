#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Load SAEs via sae_lens and dump class + source for encode, decode, forward.

Edit the SAES list below to specify which SAEs to inspect.
"""

import gc
import inspect

import torch

# ── Configure SAEs to inspect here ──────────────────────────────────────
SAES = [
    {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_14/width_65k/canonical",
    },
    {
        "release": "gemma-scope-9b-it-res",
        "sae_id": "layer_20/width_131k/average_l0_153",
    },
    {"release": "llama_scope_lxr_32x", "sae_id": "l17r_32x"},
    {
        "release": "qwen2.5-7b-instruct-andyrdt",
        "sae_id": "resid_post_layer_19_trainer_1",
    },
]

OUTPUT_DIR = "sae_inspect_output"
DEVICE = "cpu"
# Set to True to skip weight download and resolve class from config only.
# Note: the class resolved from config may differ from what from_pretrained
# actually returns (e.g. qwen BatchTopK → JumpReLU workaround).
CONFIG_ONLY = True
# ────────────────────────────────────────────────────────────────────────


def _dump_class_hierarchy(sae_class) -> list[str]:
    """Dump full source for a class and each parent up to nn.Module."""
    lines = []
    for cls in sae_class.__mro__:
        if cls is torch.nn.Module or cls is object:
            break
        lines.append("=" * 72)
        lines.append(f"CLASS: {cls.__module__}.{cls.__qualname__}")
        lines.append("=" * 72)
        try:
            lines.append(inspect.getsource(cls))
        except (TypeError, OSError) as e:
            lines.append(f"  Could not retrieve source: {e}")
        lines.append("")
    return lines


def inspect_sae_full(release: str, sae_id: str, device: str) -> str:
    """Load the actual SAE (downloads weights) and inspect the real class."""
    from sae_lens import SAE

    print(f"Loading SAE: release={release!r}, sae_id={sae_id!r} ...")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

    sae_class = type(sae)
    lines = []
    lines.append(f"SAE class: {sae_class.__module__}.{sae_class.__qualname__}")
    lines.append(f"MRO: {[c.__name__ for c in sae_class.__mro__]}")
    lines.append("")

    # Log the SAE config
    lines.append("=" * 72)
    lines.append("SAE CONFIG")
    lines.append("=" * 72)
    if hasattr(sae, "cfg"):
        for k, v in vars(sae.cfg).items():
            lines.append(f"  {k}: {v!r}")
    else:
        lines.append("  (no .cfg attribute)")
    lines.append("")

    lines.extend(_dump_class_hierarchy(sae_class))

    # Free the SAE
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "\n".join(lines)


def inspect_sae_config_only(release: str, sae_id: str) -> str:
    """Load only the config (no weights download) and resolve the class from the registry."""
    from sae_lens.loading.pretrained_sae_loaders import load_sae_config_from_huggingface
    from sae_lens.registry import get_sae_class

    print(f"Loading config only: release={release!r}, sae_id={sae_id!r} ...")
    cfg = load_sae_config_from_huggingface(release=release, sae_id=sae_id)
    architecture = cfg.get("architecture", "unknown")

    sae_class, sae_config_class = get_sae_class(architecture)

    lines = []
    lines.append(f"architecture: {architecture}")
    lines.append(
        f"SAE class (from registry): {sae_class.__module__}.{sae_class.__qualname__}"
    )
    lines.append(
        f"Config class: {sae_config_class.__module__}.{sae_config_class.__qualname__}"
    )
    lines.append(f"MRO: {[c.__name__ for c in sae_class.__mro__]}")
    lines.append("")

    # Log the raw config dict
    lines.append("=" * 72)
    lines.append("SAE CONFIG (from cfg.json)")
    lines.append("=" * 72)
    for k, v in sorted(cfg.items()):
        lines.append(f"  {k}: {v!r}")
    lines.append("")

    lines.extend(_dump_class_hierarchy(sae_class))

    return "\n".join(lines)


def main():
    import os
    import re

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, entry in enumerate(SAES):
        release = entry["release"]
        sae_id = entry["sae_id"]

        if CONFIG_ONLY:
            body = inspect_sae_config_only(release, sae_id)
        else:
            body = inspect_sae_full(release, sae_id, DEVICE)

        # Generate a filename from release + sae_id
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{release}__{sae_id}")
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        with open(filepath, "w") as f:
            f.write(f"release: {release}\nsae_id: {sae_id}\n\n{body}\n")
        print(f"  [{i+1}/{len(SAES)}] Written to {filepath}")

    print(f"\nDone — {len(SAES)} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
