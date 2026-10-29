# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Common utilities for executors.
"""

import logging
from typing import Any, List

import tqdm

from .types import ExecutorResult, GenerateFunction, PromptLoader

logger = logging.getLogger(__name__)


def generate_with_prompt_loader(
    prompt_loader: PromptLoader,
    generate_fn: GenerateFunction,
    desc: str,
    param_value: Any,
) -> List[ExecutorResult]:
    """
    Common text generation loop used by all executors.

    Args:
        prompt_loader: Iterator yielding batches of prompts from DataLoader
        generate_fn: Text generation function (transformers pipeline)
        desc: Description string for tqdm progress bar
        param_value: Parameter value to pass to dataset.augment_collated_batch
                    (e.g., strength value, config_id, or "no_hooks")

    Returns:
        List of ExecutorResult tuples from dataset.augment_collated_batch
    """
    results = []

    for batch_idx, row_batch in enumerate(tqdm.tqdm(prompt_loader, desc=desc)):
        # DataLoader collates dicts into a single dict with lists as values
        # e.g., {"prompt": ["text1", "text2"], "row_data": {...}, "row_index": [...]}
        prompts = row_batch["prompt"]
        gen = generate_fn(prompts, num_return_sequences=1)

        # Extract generated text from generation results
        processed_gen = []
        for i, prompt in enumerate(prompts):
            generated_text = gen[i][0]["generated_text"].replace(prompt, "")
            processed_gen.append(generated_text)

        # Log generations from every 10th batch
        if batch_idx % 10 == 0:
            logger.info(f"--- Sample generations from batch {batch_idx + 1} ---")
            for i, (p, g) in enumerate(zip(prompts, processed_gen)):
                logger.info(f"  [{i+1}] Prompt: '{p[:80]}...'")
                logger.info(f"  [{i+1}] Generation: '{g[:120]}...'")

        # Delegate batch augmentation to dataset - it knows its own structure
        batch_results = prompt_loader.dataset.augment_collated_batch(
            row_batch, processed_gen, param_value
        )
        results.extend(batch_results)

    return results
