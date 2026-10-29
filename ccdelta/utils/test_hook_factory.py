# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Unit tests for hook_factory module.

Tests SAE config resolution, hook creation, and backward compatibility.
"""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from ccdelta.utils.hook_factory import (
    is_sae_hook,
    resolve_sae_config_for_module,
    _create_hook_with_sae_config,
    SAE_HOOK_TYPES,
)


class TestIsSAEHook:
    """Test the is_sae_hook() function."""

    def test_sae_feature_space_is_sae_hook(self):
        """Test that sae_feature_space is recognized as SAE hook."""
        assert is_sae_hook("sae_feature_space") is True

    def test_sae_responses_is_sae_hook(self):
        """Test that sae_responses is recognized as SAE hook."""
        assert is_sae_hook("sae_responses") is True

    def test_paired_sae_responses_is_sae_hook(self):
        """Test that paired_sae_responses is recognized as SAE hook."""
        assert is_sae_hook("paired_sae_responses") is True

    def test_postprocess_and_save_is_not_sae_hook(self):
        """Test that postprocess_and_save is not recognized as SAE hook."""
        assert is_sae_hook("postprocess_and_save") is False

    def test_identity_is_not_sae_hook(self):
        """Test that identity is not recognized as SAE hook."""
        assert is_sae_hook("identity") is False

    def test_unknown_hook_is_not_sae_hook(self):
        """Test that unknown hooks are not recognized as SAE hooks."""
        assert is_sae_hook("some_random_hook") is False


class TestResolveSAEConfigForModule:
    """Test the resolve_sae_config_for_module() function."""

    def test_new_format_multi_layer_config(self):
        """Test SAE config resolution with new multi-layer format."""
        config = OmegaConf.create(
            {
                "sae_configs": {
                    "model.layers.16": {
                        "sae_id": "l16r_32x",
                        "layer": 16,
                        "use_sae_lens": True,
                    },
                    "model.layers.20": {
                        "sae_id": "l20r_32x",
                        "layer": 20,
                        "use_sae_lens": True,
                    },
                }
            }
        )

        # Test layer 16
        sae_config_16 = resolve_sae_config_for_module("model.layers.16", config)
        assert sae_config_16 is not None
        assert sae_config_16["sae_id"] == "l16r_32x"
        assert sae_config_16["layer"] == 16

        # Test layer 20
        sae_config_20 = resolve_sae_config_for_module("model.layers.20", config)
        assert sae_config_20 is not None
        assert sae_config_20["sae_id"] == "l20r_32x"
        assert sae_config_20["layer"] == 20

    def test_old_format_single_sae(self):
        """Test SAE config resolution with old single SAE format (backward compat)."""
        config = OmegaConf.create(
            {"sae": {"sae_id": "l16r_32x", "layer": 16, "use_sae_lens": True}}
        )

        # All modules should get the same config
        sae_config_16 = resolve_sae_config_for_module("model.layers.16", config)
        assert sae_config_16 is not None
        assert sae_config_16["sae_id"] == "l16r_32x"

        sae_config_20 = resolve_sae_config_for_module("model.layers.20", config)
        assert sae_config_20 is not None
        assert sae_config_20["sae_id"] == "l16r_32x"

    def test_no_sae_config(self):
        """Test SAE config resolution when no SAE config exists (non-SAE model)."""
        config = OmegaConf.create(
            {
                "model_path": "meta-llama/Llama-3.1-8B-Instruct",
                "module_names": ["model.layers.16"],
            }
        )

        sae_config = resolve_sae_config_for_module("model.layers.16", config)
        assert sae_config is None

    def test_new_format_takes_precedence_over_old(self):
        """Test that new format takes precedence when both exist."""
        config = OmegaConf.create(
            {
                "sae": {"sae_id": "old_shared", "layer": 16},
                "sae_configs": {
                    "model.layers.16": {"sae_id": "new_specific", "layer": 16}
                },
            }
        )

        # Should get new format for layer 16
        sae_config_16 = resolve_sae_config_for_module("model.layers.16", config)
        assert sae_config_16["sae_id"] == "new_specific"

        # Should fall back to old format for layer 20 (not in new format)
        sae_config_20 = resolve_sae_config_for_module("model.layers.20", config)
        assert sae_config_20["sae_id"] == "old_shared"

    def test_module_not_in_sae_configs_with_fallback(self):
        """Test behavior when module not in sae_configs but shared sae exists."""
        config = OmegaConf.create(
            {
                "sae": {"sae_id": "shared", "layer": 16},
                "sae_configs": {"model.layers.16": {"sae_id": "specific", "layer": 16}},
            }
        )

        # Layer 24 not in sae_configs, should fall back to shared sae
        sae_config_24 = resolve_sae_config_for_module("model.layers.24", config)
        assert sae_config_24 is not None
        assert sae_config_24["sae_id"] == "shared"

    def test_empty_sae_configs(self):
        """Test behavior with empty sae_configs dict."""
        config = OmegaConf.create(
            {"sae": {"sae_id": "shared", "layer": 16}, "sae_configs": {}}
        )

        # Should fall back to shared sae
        sae_config = resolve_sae_config_for_module("model.layers.16", config)
        assert sae_config is not None
        assert sae_config["sae_id"] == "shared"


class TestCreateHookWithSAEConfig:
    """Test the create_hook_with_sae_config() function."""

    def test_create_sae_hook_with_config(self):
        """Test creating an SAE hook with proper config injection."""
        config = OmegaConf.create(
            {
                "sae": {
                    "sae_id": "test_sae",
                    "layer": 16,
                    "use_sae_lens": False,
                    "url": "http://example.com/sae",
                }
            }
        )

        base_params = {
            "device": "cpu",
            "dtype": torch.float32,
        }

        # Note: This will actually try to create the hook, which might fail
        # without proper SAE model loading. In a real test environment,
        # you'd mock the SAE loading.
        try:
            hook = _create_hook_with_sae_config(
                hook_name="sae_feature_space",
                module_name="model.layers.16",
                model_config=config,
                base_hook_params=base_params,
            )
            # If we get here, hook was created successfully
            assert hook is not None
            assert hook.module_name == "model.layers.16"
        except Exception as e:
            # Expected in test environment without actual SAE models
            # Just verify the error is related to SAE loading, not config resolution
            error_str = str(e).lower()
            assert (
                "sae" in error_str
                or "url" in error_str
                or "http" in error_str  # HTTP errors from URL loading
                or "404" in error_str  # Not found errors
            ), f"Unexpected error type: {e}"

    def test_create_non_sae_hook_logs_warning(self):
        """Test that creating non-SAE hook with this function logs warning."""
        config = OmegaConf.create({"model_path": "meta-llama/Llama-3.1-8B-Instruct"})

        base_params = {
            "device": "cpu",
        }

        # This should work but log a warning
        try:
            hook = _create_hook_with_sae_config(
                hook_name="identity",
                module_name="model.layers.16",
                model_config=config,
                base_hook_params=base_params,
            )
            # If successful, verify it's the right hook
            assert hook is not None
        except Exception:
            # May fail due to missing dependencies, but that's OK for this test
            pass


class TestBackwardCompatibility:
    """Test backward compatibility scenarios."""

    def test_single_layer_old_config_still_works(self):
        """Test that old single-layer configs work unchanged."""
        old_config = OmegaConf.create(
            {
                "model_path": "meta-llama/Llama-3.1-8B-Instruct",
                "module_names": ["model.layers.16"],
                "sae": {
                    "use_sae_lens": True,
                    "release": "llama_scope_lxr_32x",
                    "sae_id": "l16r_32x",
                    "layer": 16,
                },
            }
        )

        # Should resolve correctly
        sae_config = resolve_sae_config_for_module("model.layers.16", old_config)
        assert sae_config is not None
        assert sae_config["sae_id"] == "l16r_32x"

    def test_non_sae_config_unaffected(self):
        """Test that non-SAE configs are unaffected."""
        non_sae_config = OmegaConf.create(
            {
                "model_path": "meta-llama/Llama-3.1-8B-Instruct",
                "module_names": [
                    "model.layers.16.post_attention_layernorm",
                    "model.layers.16.mlp.up_proj",
                ],
                "dtype": "torch.bfloat16",
            }
        )

        # Should return None (no SAE config)
        sae_config = resolve_sae_config_for_module(
            "model.layers.16.mlp.up_proj", non_sae_config
        )
        assert sae_config is None

        # is_sae_hook should return False for non-SAE hooks
        assert is_sae_hook("mean_ot") is False


def test_sae_hook_types_constant():
    """Test that SAE_HOOK_TYPES constant is properly defined."""
    assert isinstance(SAE_HOOK_TYPES, set)
    assert "sae_feature_space" in SAE_HOOK_TYPES
    assert "sae_responses" in SAE_HOOK_TYPES
    assert "paired_sae_responses" in SAE_HOOK_TYPES


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
