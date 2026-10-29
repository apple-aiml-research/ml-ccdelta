# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Utilities for loading and working with Sparse Autoencoders (SAEs)."""

import functools
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)


class SparseAutoencoder(nn.Module):
    """Basic Sparse Autoencoder implementation."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space."""
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction


class GatedSparseAutoencoder(nn.Module):
    """Gated Sparse Autoencoder with separate gating mechanism."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model))
        self.W_gate = nn.Parameter(torch.randn(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_gate = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space with gating."""
        gate_values = torch.sigmoid(x @ self.W_gate + self.b_gate)
        feature_values = torch.relu(x @ self.W_enc + self.b_enc)
        return gate_values * feature_values

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the gated autoencoder."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder with threshold-based activation."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space with JumpReLU activation."""
        pre_activation = x @ self.W_enc + self.b_enc
        return torch.relu(pre_activation - self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the JumpReLU autoencoder."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction


def load_sae_file(
    file_path: Union[str, Path],
) -> Union[SparseAutoencoder, GatedSparseAutoencoder, JumpReLUSAE]:
    """
    Load a SAE model from a safetensors file.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Loaded SAE model
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors package required for loading SAE models")

    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # Determine SAE type and dimensions from state dict
    d_model = state_dict["W_enc"].shape[0]
    d_sae = state_dict["W_enc"].shape[1]

    # Determine SAE type based on available parameters
    if "W_gate" in state_dict:
        model = GatedSparseAutoencoder(d_model, d_sae)
    elif "threshold" in state_dict:
        model = JumpReLUSAE(d_model, d_sae)
    else:
        model = SparseAutoencoder(d_model, d_sae)

    model.load_state_dict(state_dict, strict=False)
    return model


def get_cached_remote_file(url: str, cache_dir: Optional[str] = None) -> Path:
    """
    Download and cache a remote file locally.

    Args:
        url: URL to download
        cache_dir: Optional cache directory

    Returns:
        Path to cached file
    """
    import hashlib
    from pathlib import Path

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "act_sae"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from URL hash
    url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
    file_extension = Path(url).suffix or ".safetensors"
    cached_file = cache_dir / f"{url_hash}{file_extension}"

    if not cached_file.exists():
        logger.info(f"Downloading {url} to {cached_file}")
        import urllib.request

        urllib.request.urlretrieve(url, cached_file)  # nosemgrep

    return cached_file


@functools.lru_cache(maxsize=10)
def load_sae_model(
    sae_url: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
) -> Union[SparseAutoencoder, GatedSparseAutoencoder, JumpReLUSAE]:
    """
    Load a SAE model from URL with caching.

    Args:
        sae_url: URL to SAE model
        device: Target device for model
        dtype: Target dtype for model
        cache_dir: Cache directory for downloaded files

    Returns:
        Loaded SAE model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"Loading SAE model from {sae_url} onto device={device} and dtype={dtype}"
    )

    # Handle different URL formats
    if not sae_url.endswith("/SAE_model.safetensors"):
        if not sae_url.endswith("/"):
            sae_url += "/"
        sae_url += "SAE_model.safetensors"

    local_file = get_cached_remote_file(sae_url, cache_dir)
    sae_model = load_sae_file(local_file)
    return sae_model.to(device=device, dtype=dtype)


def load_sae_lens_model(release, sae_id, device):
    from sae_lens import SAE

    # WORKAROUND: Force JumpReLU loading for Qwen BatchTopK SAE
    # The sae_lens loader doesn't recognize "BatchTopKSAE" in config.json and incorrectly
    # loads as StandardSAE without the threshold. BatchTopK SAEs are numerically equivalent
    # to JumpReLU at inference time (verified: max diff ~2e-5, same L0, same active positions).
    if release == "qwen2.5-7b-instruct-andyrdt":
        logger.info(
            "Detected Qwen BatchTopK SAE - using manual JumpReLU loading to preserve threshold"
        )
        from huggingface_hub import hf_hub_download
        from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

        # Download weights from HuggingFace
        # sae_id format: "resid_post_layer_15_trainer_1"
        # path format: "resid_post_layer_15/trainer_1/ae.pt"
        # Only replace the underscore before "trainer" with a slash
        filename = sae_id.replace("_trainer_", "/trainer_") + "/ae.pt"
        weights_path = hf_hub_download(
            repo_id="andyrdt/saes-qwen2.5-7b-instruct",
            filename=filename,
        )

        # Load raw state dict
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)  # nosemgrep

        # Create JumpReLU config
        d_in = state_dict["encoder.weight"].shape[1]
        d_sae = state_dict["encoder.weight"].shape[0]

        cfg = JumpReLUSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            dtype="float32",
            device=str(device),
        )

        # Create JumpReLU SAE
        sae = JumpReLUSAE(cfg)

        # Map state dict keys from BatchTopK format to JumpReLU format
        # BatchTopK stores encoder/decoder weights transposed relative to JumpReLU
        mapped_state_dict = {
            "W_enc": state_dict["encoder.weight"].T,  # [d_in, d_sae]
            "b_enc": state_dict["encoder.bias"],      # [d_sae]
            "W_dec": state_dict["decoder.weight"].T,  # [d_sae, d_in]
            "b_dec": state_dict["b_dec"],             # [d_in]
            "threshold": state_dict["threshold"].expand(d_sae),  # scalar → [d_sae]
        }

        # Load weights
        sae.load_state_dict(mapped_state_dict, strict=False)

        threshold_val = state_dict["threshold"].item()
        logger.info(
            f"Loaded Qwen SAE as JumpReLU: d_in={d_in}, d_sae={d_sae}, threshold={threshold_val:.4f}"
        )

        return sae

    # Default: use sae_lens loader
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)

    # Log the SAE class and forward method source for debugging
    import inspect

    sae_class = type(sae)
    logger.info(
        f"Loaded SAE class: {sae_class.__module__}.{sae_class.__qualname__}"
    )
    try:
        class_source = inspect.getsource(sae_class)
        logger.info(f"SAE class source:\n{class_source}")
    except (TypeError, OSError) as e:
        logger.info(f"Could not retrieve class source: {e}")

    return sae


def load_sae_model_for_hook(
    config: dict, device: str, dtype: torch.dtype, cache_dir: Optional[str] = None
):
    """Load SAE model for hook with proper validation.

    Also attaches metadata attributes to the model for analysis purposes.
    """
    # Validate config parameter
    if config is None:
        raise ValueError("SAE config cannot be None")

    if config.get("use_sae_lens", False):
        # Validate sae_lens requirements
        required_fields = ["sae_id", "release"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields for sae_lens: {missing_fields}")

        sae_id = config["sae_id"]
        release = config["release"]
        logger.info(f"Loading SAE from sae_lens: release={release}, sae_id={sae_id}")
        sae_model = load_sae_lens_model(
            release=release, sae_id=sae_id, device=device
        ).to(dtype)

        # Attach config identifiers for metadata retrieval
        sae_model.sae_lens_release = release
        sae_model.sae_lens_sae_id = sae_id

        # Try to get pretrained directory info for additional metadata
        try:
            from sae_lens.loading.pretrained_saes_directory import (
                get_pretrained_saes_directory,
            )

            directory = get_pretrained_saes_directory()
            if release in directory:
                sae_info = directory[release]
                sae_model.expected_l0 = sae_info.expected_l0.get(sae_id)
                sae_model.expected_var_explained = sae_info.expected_var_explained.get(
                    sae_id
                )
                sae_model.neuronpedia_id = sae_info.neuronpedia_id.get(sae_id)
                sae_model.repo_id = sae_info.repo_id
                sae_model.model_name_from_directory = sae_info.model
                logger.info(
                    f"Attached SAE metadata: expected_l0={sae_model.expected_l0}, "
                    f"expected_var_explained={sae_model.expected_var_explained}"
                )
        except Exception as e:
            logger.warning(f"Could not load pretrained SAE directory info: {e}")
    else:
        # Validate URL-based requirements
        if "url" not in config:
            raise ValueError("Missing required field 'url' for URL-based SAE loading")

        sae_url = config["url"]
        logger.info(f"Loading SAE from URL: {sae_url}")
        sae_model = load_sae_model(
            sae_url, device=device, dtype=dtype, cache_dir=cache_dir
        )

    return sae_model


def compute_sae_feature_comparison(
    source_features: torch.Tensor, target_features: torch.Tensor
) -> dict:
    """Compute sequence-level difference metrics between source and target SAE features.

    This function compares two sets of SAE features (e.g., from paired source/target
    examples) using sequence-level aggregation. This is appropriate when the sequences
    have different lengths and no natural token alignment (e.g., jailbreak wrapper vs
    forbidden prompt).

    All metrics use the FULL sequences (no truncation) since we care about the
    aggregate feature distribution for steering purposes.

    Args:
        source_features: Source SAE features [S1, D_sae].
        target_features: Target SAE features [S2, D_sae].

    Returns:
        Dictionary with sequence-level comparison metrics:
        - cosine_similarity_mean: Cosine similarity between mean-pooled feature vectors
        - feature_overlap_ratio: Jaccard ratio of features ever active in each sequence
        - activation_diff_norm_mean: L2 distance between mean-pooled feature vectors
        - sequence_length_source: Number of tokens in source
        - sequence_length_target: Number of tokens in target
    """
    # Ensure dense tensors for computation
    if source_features.is_sparse:
        source_features = source_features.to_dense()
    if target_features.is_sparse:
        target_features = target_features.to_dense()

    S1, D_sae = source_features.shape
    S2, _ = target_features.shape

    # Handle edge case: empty sequences
    if S1 == 0 or S2 == 0:
        return {
            "cosine_similarity_mean": float("nan"),
            "feature_overlap_ratio": 0.0,
            "activation_diff_norm_mean": float("nan"),
            "sequence_length_source": S1,
            "sequence_length_target": S2,
        }

    # === Sequence-level metrics (no truncation) ===

    # 1. Mean-pooled feature vectors: average activation per feature across sequence
    src_mean = source_features.mean(dim=0)  # [D_sae]
    tgt_mean = target_features.mean(dim=0)  # [D_sae]

    # 2. Cosine similarity between mean-pooled vectors
    src_norm = src_mean.norm().clamp(min=1e-8)
    tgt_norm = tgt_mean.norm().clamp(min=1e-8)
    cosine_sim = (src_mean * tgt_mean).sum() / (src_norm * tgt_norm)

    # 3. Feature overlap: Jaccard over features ever active in each sequence
    # "What fraction of features that fire anywhere in either sequence fire in both?"
    src_ever_active = (source_features != 0).any(dim=0)  # [D_sae] - feature ever active?
    tgt_ever_active = (target_features != 0).any(dim=0)  # [D_sae]
    intersection = (src_ever_active & tgt_ever_active).sum().float()
    union = (src_ever_active | tgt_ever_active).sum().float()

    if union == 0:
        overlap_ratio = 0.0
    else:
        overlap_ratio = intersection / union

    # 4. L2 distance between mean-pooled vectors
    diff_norm = (src_mean - tgt_mean).norm()

    return {
        "cosine_similarity_mean": cosine_sim.item(),
        "feature_overlap_ratio": overlap_ratio.item() if isinstance(overlap_ratio, torch.Tensor) else overlap_ratio,
        "activation_diff_norm_mean": diff_norm.item(),
        "sequence_length_source": S1,
        "sequence_length_target": S2,
    }
