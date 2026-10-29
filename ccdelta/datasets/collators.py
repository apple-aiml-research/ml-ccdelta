# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import typing as t
from collections import defaultdict

import torch
import transformers


class DictCollatorWithPadding(transformers.DataCollatorWithPadding):
    """Helper class that pads text sequences of multiple lengths
    to a contiguous batch by applying huggingface's
    DataCollatorWithPadding. For the rest of data types
    it converts them from list of dicts to dict of lists.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        return_tensors: str = "pt",
        pad_to_multiple_of: int = 2,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )
        self.tensor_set = set(["input_ids", "attention_mask"])

    def __call__(self, batch: t.List) -> t.Tuple[t.Dict[str, torch.Tensor], t.Dict]:
        """Function to be applied on list of samples from dataloader to form a batch

        Args:
            batch (t.List): list of samples

        Returns:
            t.Tuple[t.Dict[str, torch.Tensor], t.Dict]: Tuple with tokens and additional metadata like labels
        """
        ret = defaultdict(list)
        tensors = []
        meta_set = set(batch[0].keys()) - self.tensor_set
        for sample in batch:
            tensors.append({k: sample[k] for k in self.tensor_set})
            for k in meta_set:
                ret[k].append(sample[k])
        tensors = transformers.DataCollatorWithPadding.__call__(
            self, tensors
        )  # huggingface does the padding
        ret.update(tensors)
        return ret


class CollationMixin:
    """Mixin class that provides collation-aware batch augmentation functionality.

    This mixin abstracts away the complexity of PyTorch DataLoader collation,
    allowing datasets to only implement single-row augmentation logic.
    """

    def augment_collated_batch(
        self, collated_batch: dict, generations: list, param_value: t.Union[float, str]
    ) -> list:
        """Augment collated batch with generation results to create individual result rows.

        Args:
            collated_batch: The collated batch from DataLoader
            generations: List of processed generation text strings
            param_value: The parameter value used for generation

        Returns:
            List of individual result dictionaries
        """
        results = []
        batch_size = len(collated_batch["prompt"])

        for i in range(batch_size):
            # Reconstruct individual row from collated batch
            row = self._uncollate_row(collated_batch, i)

            # Delegate to subclass for augmentation
            augmented_row = self.augment_row(row, generations[i], param_value)
            results.append(augmented_row)

        return results

    def _uncollate_row(self, collated_batch: dict, index: int) -> dict:
        """Reconstruct individual row data from PyTorch DataLoader's collated batch.

        PyTorch DataLoader collates list of dicts into dict of lists:
        [{"prompt": "text1", "row_data": {...}}, {"prompt": "text2", "row_data": {...}}]
        becomes:
        {"prompt": ["text1", "text2"], "row_data": {"key": [val1, val2]}, "row_index": tensor([0, 1])}

        This method extracts an index from the collated batch and turns it back to a dict.

        Args:
            collated_batch: The collated batch from DataLoader
            index: Index of the specific row to reconstruct

        Returns:
            Individual row data as a dict
        """
        row = {}

        for key, values in collated_batch.items():
            if isinstance(values, (list, tuple)):
                # Simple list/tuple
                row[key] = values[index]
            elif isinstance(values, dict):
                # Nested dict structure (e.g., row_data) - index into each nested list
                row[key] = {k: v[index] for k, v in values.items()}
            elif hasattr(values, "__getitem__") and hasattr(values, "__len__"):
                # Tensor or other indexable sequence
                value = values[index]
                # Convert tensor scalars to Python native types for serialization compatibility
                if hasattr(value, "item"):
                    row[key] = value.item()
                else:
                    row[key] = value
            else:
                # Scalar values that weren't collated.
                row[key] = values

        return row

    def augment_row(
        self, row_data: dict, generation: str, param_value: t.Union[float, str]
    ) -> dict:
        """Augment a single input dataset row with generation results.

        This method should be implemented by subclasses to provide simple
        single-row augmentation logic.

        Args:
            row_data: dataset row
            generation: generated text
            param_value: parameter value used for generation

        Returns:
            Augmented row dictionary
        """
        raise NotImplementedError("Subclasses must implement augment_single_row")
