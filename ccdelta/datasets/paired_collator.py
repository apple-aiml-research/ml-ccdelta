# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Paired collator for processing both source and target texts."""

import typing as t
from collections import defaultdict

import transformers

from ccdelta.utils.logging import get_logger

logger = get_logger(__name__)


class PairedDictCollator:
    """Collator for paired data that tokenizes both source and target texts."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        return_tensors: str = "pt",
        pad_to_multiple_of: int = 2,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.base_collator = transformers.DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )

    def __call__(self, batch: t.List) -> t.Dict:
        """Process batch with both source and target texts for sequential processing.

        Args:
            batch: List of samples with source_text, target_text, etc.

        Returns:
            Dictionary with separate tokenized source and target batches plus metadata
        """
        logger.debug(f"Processing batch of {len(batch)} samples")

        # Separate source and target texts for tokenization
        source_texts = [sample["source_text"] for sample in batch]
        target_texts = [sample["target_text"] for sample in batch]

        logger.debug(f"Source texts: {[text[:30] + '...' for text in source_texts]}")
        logger.debug(f"Target texts: {[text[:30] + '...' for text in target_texts]}")

        # Tokenize both sets of texts separately, excluding special tokens to avoid substring matching issues
        source_tokens = self.tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=2,
            add_special_tokens=False,
        )

        target_tokens = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=2,
            add_special_tokens=False,
        )

        # Collect metadata
        metadata = defaultdict(list)
        for sample in batch:
            for key, value in sample.items():
                if key not in ["source_text", "target_text"]:
                    metadata[key].append(value)

        # Build result dictionary with source/target tokens
        # process_paired_batch will extract these into proper batches
        result = {
            "source_input_ids": source_tokens["input_ids"],
            "source_attention_mask": source_tokens["attention_mask"],
            "target_input_ids": target_tokens["input_ids"],
            "target_attention_mask": target_tokens["attention_mask"],
            **dict(metadata),
        }

        logger.debug(f"Result keys: {list(result.keys())}")
        logger.debug(f"Source input_ids shape: {result['source_input_ids'].shape}")
        logger.debug(f"Target input_ids shape: {result['target_input_ids'].shape}")

        return result
