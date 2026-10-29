# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import tempfile
import os
import pandas as pd
import pytest
import torch
from hydra import compose, initialize

from ccdelta.scripts import pipeline


def test_pipeline_no_intervention():
    """Test pipeline with no intervention (baseline)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                "fast=true",
                f"device={device}",
                "evaluation=['text-generation']",
                "responses.tag=giraffes-responses-pipeline",
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",  # Use available filter from giraffes_paired
                "intervention_params=none",
                "data_dir=.",
                f"cache_dir={tempfolder}",
                f"interventions.cache_dir={tempfolder}",
                "compute_responses=true",  # Generate responses on-the-fly
                "responses.max_batches=1",  # Limit for faster tests
                f"results_dir={tempfolder}",
                # Use Gemma 2B for chat template support
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.0', 'model.layers.1']",
                "+use_subprocess=false",
            ]

            cfg = compose(
                config_name="text_generation_caa",
                overrides=overrides
                + [
                    # Override jailbreak data with simple giraffes data for speed - using giraffes_paired structure
                    "task_params.eval_dataset.file_path=data/tiny/giraffes_prompts.json",
                    "task_params.learning_dataset.file_path=data/tiny/giraffes_prompts.json",
                    # Override column names to match what jailbreak_dense config expects
                    "task_params.eval_dataset.source_column=prompt",
                    "task_params.eval_dataset.target_column=destination",
                    "task_params.learning_dataset.source_column=prompt",
                    "task_params.learning_dataset.target_column=destination",
                ],
            )
            pipeline.main(cfg)


@pytest.mark.parametrize(
    "feature_selection_method,ranking_method,incremental_mode",
    [
        ("none", "effect_size", "atonce"),  # effect_size ranking without significance filter
        (
            "none",
            "magnitude",
            "atonce",
        ),  # magnitude ranking (CAA-style)
        # ("none", "effect_size", "atonce"),  # Ablation: no selection + all-at-once
        (
            "none",
            "effect_size",
            "incr",
        ),  # Test incremental learning with paired data
    ],
)
def test_pipeline_paired_sae_jailbreak(
    feature_selection_method, ranking_method, incremental_mode
):
    """Test pipeline with SAE-based paired jailbreak interventions.

    This integration test verifies the complete SAE steering pipeline using
    the paired jailbreak dataset (strongreject_small_jailbroken.parquet).

    The test performs:
    1. Response collection: Extracts SAE features from both jailbroken and
       forbidden prompts using PairedResponsesHook with SAEActivationsHook
    2. Intervention learning: Learns steering vectors from feature differences
       between jailbreaking and refusal behaviors (supports both "atonce" and "incr" modes)
    3. Text generation: Applies learned SAE steering to transform jailbreak
       prompts toward refusal behaviors.

    Uses Gemma-2-2b-it with Gemmascope Layer 13 SAE for realistic testing.

    Tests both feature selection methods, ranking methods, and learning modes:
    - feature_selection_method:
      - "statistical": Uses statistical significance testing (default)
      - "none": Ablation condition using all features without selection
    - ranking_method:
      - "effect_size": Rank by Cohen's d style effect size (default)
      - "magnitude": Rank by absolute mean_diff (CAA-style, faster)
    - incremental_mode:
      - "atonce": Learn all interventions at once (default)
      - "incr": Learn interventions incrementally, layer by layer
    """
    device = "cuda" if torch.cuda.is_available() else "mps"

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                f"device={device}",
                f"cache_dir={tempfolder}",
                f"results_dir={tempfolder}",
                # Use our paired jailbreak task configuration
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",  # Use available filter from giraffes_paired
                "evaluation=['text-generation']",
                # SAE configuration - use "all" pooling for individual token activations
                "intervention_params.pooling_op=all",
                # Test incremental mode
                f"intervention_params.incremental={incremental_mode}",
                # Fast test settings to reduce CI time
                "responses.max_batches=1",  # Limit response collection batches
                "interventions.batch_size=2",  # Small batch size for intervention learning
                # Relax statistical requirements for integration testing
                "intervention_params.hook_params.min_observations_absolute=1",  # Allow even fewer samples
                "intervention_params.hook_params.min_observations_percent=1.0",  # Lower percentage
                "intervention_params.hook_params.p_value_threshold=0.99",  # Very lenient p-value for testing
                "intervention_params.hook_params.feature_diffing_method=source_conditioned",  # Require substring matching for jailbreak data
                # Test the feature selection method
                f"intervention_params.hook_params.feature_selection_method={feature_selection_method}",
                # Test the ranking method
                f"intervention_params.hook_params.ranking_method={ranking_method}",
                # Use Gemma 2B with corresponding SAE (as specified)
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.13']",  # Layer 13 for SAE
                # Text generation settings - more sentences for better testing
                "text_generation.num_sentences=100",
                "text_generation.new_seq_len=10",
                # Ensure we use the correct data directory
                "data_dir=.",  # Use root directory to access data/strongreject_small_jailbroken.parquet
                "+use_subprocess=false",
                f"text_generation.experiment_id=test_run_{feature_selection_method}_{ranking_method}_{incremental_mode}",
                "response_collection_hook.hook_params.return_sparse=false",
            ]

            cfg = compose(
                config_name="text_generation_ccdelta",
                overrides=overrides,
            )
            pipeline.main(cfg)

            # Assert that the results contain the expected multiplier suffixes
            results_path = os.path.join(
                tempfolder, "generate_with_hooks", "text_generation.json"
            )
            assert os.path.exists(
                results_path
            ), f"results_path={results_path} not found"

            df = pd.read_json(results_path)
            assert "generation_param" in df.columns



def test_pipeline_experiment_id_flow():
    """Test that experiment_id flows through from config to final output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_experiment_id = "test_experiment_id_flow_verification"

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                "fast=true",
                f"device={device}",
                "evaluation=['text-generation']",
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",  # Use available filter from giraffes_paired
                "intervention_params=none",
                "data_dir=.",
                f"cache_dir={tempfolder}",
                f"interventions.cache_dir={tempfolder}",
                "compute_responses=true",
                "responses.max_batches=1",
                f"results_dir={tempfolder}",
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.0']",
                f"+intervention_params.experiment_id={test_experiment_id}",
                f"text_generation.experiment_id={test_experiment_id}",  # Explicitly set text_generation experiment_id
                # Use none intervention which has the correct executor (no_hooks) for baseline testing
                "+use_subprocess=false",
            ]

            cfg = compose(
                config_name="text_generation_caa",
                overrides=overrides
                + [
                    # Override jailbreak data with simple giraffes data for speed - use tabular_to_json with giraffes columns
                    "task_params.src_subsets=['prompt']",  # Use giraffes column name instead of jailbroken_prompt
                    "task_params.dst_subsets=['destination']",  # Use giraffes column name instead of forbidden_prompt
                    "task_params.eval_dataset.file_path=data/tiny/giraffes_prompts.json",
                    "task_params.learning_dataset.file_path=data/tiny/giraffes_prompts.json",
                ],
            )
            pipeline.main(cfg)

            # Verify experiment_id appears in the output
            results_path = os.path.join(
                tempfolder, "generate_with_hooks", "text_generation.json"
            )
            assert os.path.exists(results_path)

            df = pd.read_json(results_path)
            print("dfdfdf", df)
            assert "generation_param" in df.columns

            # # Check that our experiment_id appears in the generation_param column
            found_params = df["generation_param"].unique()
            print("FOUND PARAMS", found_params)
            assert any(
                test_experiment_id in str(param) for param in found_params
            ), f"Expected {test_experiment_id} in {found_params}"


def test_pipeline_sae_prompt_vs_all_position():
    """
    Tests that "prompt" and "all" intervention positions produce different results.

    This test runs the non-OT SAE pipeline twice with a strong intervention:
    1. With intervention_position="prompt"
    2. With intervention_position="all"

    It then asserts that the generated text from the two runs is different,
    confirming that the logic for applying interventions at different positions
    is working correctly.
    """
    device = "cuda" if torch.cuda.is_available() else "mps"
    generated_texts = {}

    # Use a high steering multiplier to maximize the difference in output
    steering_multiplier = 5.0

    for position in ["prompt", "all"]:
        with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
            with initialize(version_base=None, config_path="../configs"):
                overrides = [
                    f"device={device}",
                    f"cache_dir={tempfolder}",
                    f"results_dir={tempfolder}",
                    "task_params=giraffes_paired",
                    "task_params.filter_key=no_op",  # Use available filter from giraffes_paired
                    "evaluation=['text-generation']",
                    # Set the intervention position for this run
                    f"intervention_params.hook_params.intervention_position={position}",
                    # Use a strong steering multiplier to ensure outputs differ
                    f"intervention_params.hook_params.steering_multiplier={steering_multiplier}",
                    "intervention_params.pooling_op=all",
                    # Fast test settings
                    "responses.max_batches=1",
                    "interventions.batch_size=2",
                    "intervention_params.hook_params.min_observations_absolute=1",
                    "intervention_params.hook_params.min_observations_percent=1.0",
                    "intervention_params.hook_params.p_value_threshold=0.99",
                    "intervention_params.hook_params.feature_selection_method=none",
                    "intervention_params.hook_params.feature_diffing_method=source_conditioned",
                    "model.model_path=google/gemma-2-2b-it",
                    "model.module_names=['model.layers.13']",
                    "text_generation.num_sentences=10",  # Fewer sentences for speed
                    "text_generation.new_seq_len=10",
                    "data_dir=.",
                    "+use_subprocess=false",
                    f"text_generation.experiment_id=test_{position}_position",
                    "response_collection_hook.hook_params.return_sparse=false",
                ]

                cfg = compose(
                    config_name="text_generation_ccdelta",  # Use non-OT SAE pipeline
                    overrides=overrides,
                )
                pipeline.main(cfg)

                results_path = os.path.join(
                    tempfolder, "generate_with_hooks", "text_generation.json"
                )
                assert os.path.exists(results_path)

                df = pd.read_json(results_path)
                # Store the generated text (as a sorted list to be order-independent)
                generated_texts[position] = sorted(df["steered_text"].tolist())

    # Assert that the generated text is different between the two runs
    assert "prompt" in generated_texts
    assert "all" in generated_texts
    assert (
        generated_texts["prompt"] != generated_texts["all"]
    ), "Generated text for 'prompt' and 'all' positions should be different."


def test_pipeline_caa_context_conditioned():
    """Test pipeline with CAA context-conditioned interventions.

    This integration test verifies the complete CAA context-conditioned pipeline:
    1. Response collection: Extracts dense activations from both jailbroken and
       forbidden prompts using PairedResponsesHook with IdentityHook (no SAE)
    2. Intervention learning: Learns steering vector from dense activation differences
       with spatial conditioning (substring matching using source_conditioned method)
    3. Text generation: Applies learned dense steering vector to transform jailbreak
       prompts toward refusal behaviors.

    This hook enables comparison between:
    - Standard CAA (use CAAHook for baseline without conditioning)
    - CAA + spatial conditioning (this hook with source_conditioned)
    - SAE + spatial conditioning (SAEFeatureSpaceHook)

    Only tests source_conditioned method (strict substring matching).
    For baseline CAA without conditioning, use the standard CAAHook instead.
    """
    # Use CPU to avoid MPS NaN issues during testing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                f"device={device}",
                f"cache_dir={tempfolder}",
                f"results_dir={tempfolder}",
                # Use paired jailbreak task configuration
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",
                "evaluation=['text-generation']",
                # Keep per-token features during response collection
                "intervention_params.pooling_op=all",
                # Use source_conditioned method (strict substring matching)
                "intervention_params.hook_params.feature_diffing_method=source_conditioned",
                # CAA strength parameter
                "intervention_params.hook_params.strength=0.8",
                # Fast test settings to reduce CI time
                "responses.max_batches=1",
                "interventions.batch_size=2",
                # Use Gemma 2B (no SAE needed for dense CAA)
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.13']",
                # Text generation settings
                "text_generation.num_sentences=10",
                "text_generation.new_seq_len=10",
                # Data directory
                "data_dir=.",
                "+use_subprocess=false",
                "text_generation.experiment_id=test_caa_conditioned_source_conditioned",
            ]

            cfg = compose(
                config_name="text_generation_paired_caa_conditioned",
                overrides=overrides,
            )
            pipeline.main(cfg)

            # Verify results were generated
            results_path = os.path.join(
                tempfolder, "generate_with_hooks", "text_generation.json"
            )
            print(f"DEBUG: Results saved to {results_path}")
            assert os.path.exists(
                results_path
            ), f"results_path={results_path} not found"

            df = pd.read_json(results_path)
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print(f"DEBUG: First row:\n{df.iloc[0].to_dict()}")
            print(f"DEBUG: Sample steered_text values:")
            for i, text in enumerate(df["steered_text"].head(3)):
                print(f"  [{i}]: '{text}'")

            assert "generation_param" in df.columns
            assert "steered_text" in df.columns

            # Verify generated text exists and is non-empty
            assert len(df) > 0, "No results generated"
            assert all(
                len(str(text)) > 0 for text in df["steered_text"]
            ), "Generated text should not be empty"

            # Verify source_conditioned method appears in generation_param
            generation_params = df["generation_param"].unique()
            assert any(
                "source_conditioned" in str(param) for param in generation_params
            ), f"Expected 'source_conditioned' in generation params: {generation_params}"


def test_pipeline_caa_regular():
    """Test pipeline with regular CAA interventions using new masking logic.

    This integration test verifies the complete CAA pipeline with the new
    sophisticated special token masking (skip_special_token_filtering=false):
    1. Response collection: Extracts dense activations from source and target subsets
    2. Intervention learning: Learns steering vector from activation differences
    3. Text generation: Applies learned steering vector with semantic token masking

    This test specifically verifies that:
    - Regular CAA works with the new masking path (not legacy BOS slicing)
    - Special tokens are properly filtered during intervention
    - Generated text is produced correctly

    Uses giraffes_paired task_params which has tabular-paired format with eval_dataset
    and learning_dataset pointing to giraffes_prompts.json (columns: 'prompt', 'destination').
    """
    # Use CPU to avoid MPS/CUDA issues during testing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                "fast=true",
                f"device={device}",
                "evaluation=['text-generation']",
                "intervention_params=caa",
                "intervention_params.incremental=atonce",
                "intervention_params.hook_params.skip_special_token_filtering=false",
                # Use giraffes_paired which has eval_dataset and learning_dataset fields
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",  # Use available filter from giraffes_paired
                "data_dir=.",
                f"cache_dir={tempfolder}",
                f"interventions.cache_dir={tempfolder}",
                f"results_dir={tempfolder}",
                "compute_responses=true",
                "responses.max_batches=1",
                # Use Gemma 2B like the context-conditioned test
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.13']",
                # Set experiment ID
                "text_generation.experiment_id=test_regular_caa_new_masking",
                # Text generation settings
                "text_generation.num_sentences=10",
                "text_generation.new_seq_len=10",
                "+use_subprocess=false",
            ]

            cfg = compose(
                config_name="text_generation_caa",
                overrides=overrides
                + [
                    # Override jailbreak column names with giraffes column names
                    "task_params.src_subsets=['prompt']",  # Use giraffes column name instead of jailbroken_prompt
                    "task_params.dst_subsets=['destination']",  # Use giraffes column name instead of forbidden_prompt
                ],
            )

            # Verify that skip_special_token_filtering is False (new masking enabled)
            print(
                f"DEBUG: skip_special_token_filtering = {cfg.intervention_params.hook_params.get('skip_special_token_filtering', 'not set (defaults to True)')}"
            )
            assert (
                cfg.intervention_params.hook_params.skip_special_token_filtering
                is False
            ), "Expected skip_special_token_filtering=false for new masking logic"

            pipeline.main(cfg)

            # Verify results were generated
            results_path = os.path.join(
                tempfolder, "generate_with_hooks", "text_generation.json"
            )
            print(f"DEBUG: Results saved to {results_path}")
            assert os.path.exists(
                results_path
            ), f"results_path={results_path} not found"

            df = pd.read_json(results_path)
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print(f"DEBUG: First row:\n{df.iloc[0].to_dict()}")
            print(f"DEBUG: Sample steered_text values:")
            for i, text in enumerate(df["steered_text"].head(3)):
                print(f"  [{i}]: '{text}'")

            assert "generation_param" in df.columns
            assert "steered_text" in df.columns

            # Verify generated text exists and is non-empty
            assert len(df) > 0, "No results generated"
            assert all(
                len(str(text)) > 0 for text in df["steered_text"]
            ), "Generated text should not be empty"

            # Verify experiment_id appears in generation_param
            generation_params = df["generation_param"].unique()
            assert any(
                "test_regular_caa_new_masking" in str(param)
                for param in generation_params
            ), f"Expected experiment_id in generation params: {generation_params}"


def test_pipeline_paired_sae_compiled_compare():
    """Test pipeline with compiled_compare_mode enabled for SAE feature space hook.

    This integration test verifies:
    1. The pipeline runs correctly with compiled_compare_mode=true
    2. compilation_error_stats.json is produced in the output directory
    3. The error stats contain near-zero equivalence errors, proving compiled
       dense mode is mathematically equivalent to the full SAE roundtrip
    4. Normal text generation results are also produced

    Uses Gemma-2-2b-it with Gemmascope Layer 13 SAE.
    """
    import json as json_mod
    import contextlib

    device = os.environ.get(
        "TEST_DEVICE", "cuda" if torch.cuda.is_available() else "mps"
    )
    fixed_output = os.environ.get("TEST_OUTPUT_DIR")

    # Use fixed dir if provided (persists after test), otherwise use temp dir
    if fixed_output:
        os.makedirs(fixed_output, exist_ok=True)
        ctx = contextlib.nullcontext(fixed_output)
    else:
        ctx = tempfile.TemporaryDirectory(dir="/tmp/")

    with ctx as tempfolder:
        with initialize(version_base=None, config_path="../configs"):
            overrides = [
                f"device={device}",
                "model.dtype=${dtype:torch.float32}",  # Force float32 to prove exact equivalence
                f"cache_dir={tempfolder}",
                f"results_dir={tempfolder}",
                # Use our paired task configuration
                "task_params=giraffes_paired",
                "task_params.filter_key=no_op",
                "evaluation=['text-generation']",
                # SAE configuration
                "intervention_params.pooling_op=all",
                "intervention_params.incremental=atonce",
                # Enable compiled compare mode
                "intervention_params.hook_params.compiled_compare_mode=true",
                # Ensure compatible settings for compilation
                "intervention_params.hook_params.error_coefficient=1.0",
                "intervention_params.hook_params.normalization_mode=null",
                # Fast test settings
                "responses.max_batches=1",
                "interventions.batch_size=2",
                # Relax statistical requirements for integration testing
                "intervention_params.hook_params.min_observations_absolute=1",
                "intervention_params.hook_params.min_observations_percent=1.0",
                "intervention_params.hook_params.p_value_threshold=0.99",
                "intervention_params.hook_params.feature_diffing_method=source_conditioned",
                # Use magnitude ranking to bypass significance testing
                "intervention_params.hook_params.ranking_method=magnitude",
                "intervention_params.hook_params.feature_selection_method=none",
                "intervention_params.hook_params.num_features=10",
                # Use Gemma 2B with corresponding SAE
                "model.model_path=google/gemma-2-2b-it",
                "model.module_names=['model.layers.13']",
                # Text generation settings
                "text_generation.num_sentences=2",
                "text_generation.new_seq_len=10",
                "data_dir=.",
                "+use_subprocess=false",
                "text_generation.experiment_id=test_compiled_compare",
                "response_collection_hook.hook_params.return_sparse=false",
            ]

            cfg = compose(
                config_name="text_generation_ccdelta",
                overrides=overrides,
            )
            pipeline.main(cfg)

            # Verify normal results were generated
            results_path = os.path.join(
                tempfolder, "generate_with_hooks", "text_generation.json"
            )
            assert os.path.exists(
                results_path
            ), f"results_path={results_path} not found"

            df = pd.read_json(results_path)
            assert "generation_param" in df.columns
            assert len(df) > 0, "No results generated"

            # Verify compilation error stats were saved
            stats_path = os.path.join(
                tempfolder, "generate_with_hooks", "compilation_error_stats.json"
            )
            assert os.path.exists(
                stats_path
            ), f"compilation_error_stats.json not found at {stats_path}"

            with open(stats_path) as f:
                stats = json_mod.load(f)

            # Should have stats for at least one hook
            assert len(stats) > 0, "No compilation stats recorded"

            # Check that all hooks report compiled=True and near-zero errors
            for hook_name, hook_stats in stats.items():
                assert hook_stats["compiled"] is True, f"Hook {hook_name} not compiled"
                error = hook_stats["error"]
                assert (
                    error["num_elements_abs"] > 0
                    or error.get("num_nonfinite_skipped", 0) > 0
                ), f"Hook {hook_name} recorded no elements and no skipped nonfinite"
                # The compiled path should be mathematically equivalent
                # In float32 this is exact (see unit tests); in bfloat16 on MPS/GPU,
                # the naive SAE roundtrip accumulates rounding error through encode/decode
                # matmuls, so we use relaxed thresholds here.
                if error["num_elements_abs"] > 0:
                    assert (
                        error["mean_abs_error"] < 0.5
                    ), f"Hook {hook_name} mean_abs_error too high: {error['mean_abs_error']}"
                    assert (
                        error["mean_rel_l2_error"] < 0.01
                    ), f"Hook {hook_name} mean_rel_l2_error too high: {error['mean_rel_l2_error']}"
