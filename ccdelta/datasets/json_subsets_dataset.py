# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import typing as t
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from ccdelta.datasets.collators import DictCollatorWithPadding


class JsonSubsetsDataset(Dataset):
    """
    A dataset class for loading and handling text data from a JSON file,
    where the data is organized into subsets. Each subset corresponds to
    a key in the JSON file, and the associated value is a list of sentences
    or text entries.

    Attributes:
        data (list): A list containing all the sentences from the specified subsets.
        subsets (list): A list indicating the subset each sentence belongs to.

    Methods:
        get_all_subsets():
            Returns a set of all unique subsets present in the dataset.
        __len__():
            Returns the total number of text entries in the dataset.
        __getitem__(idx):
            Retrieves a dictionary containing the ID, subset, and text entry
            for the specified index.

    Args:
        json_path (str): Path to the JSON file containing the dataset.
        subsets (list, optional): A list of keys corresponding to the subsets
            to be included in the dataset. If not specified or set to None,
            all subsets will be included. Defaults to "*" (all subsets).
    """

    def __init__(
        self,
        json_path: Path,
        subsets: t.List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        use_chat_template: bool = True,
        **kwargs,
    ):
        super().__init__()
        if subsets is None:
            subsets = "*"
        if not isinstance(subsets, (list, tuple, set)):
            subsets = [subsets]
        with open(json_path, "r") as f:
            data: t.Dict = json.load(f)

        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.data = []
        self.subsets = []
        self.idx_in_subset = []
        for key, sentences in data.items():
            if subsets == "*" or key in subsets:
                subset = key
            else:
                continue
            self.data += sentences
            self.subsets += [subset] * len(sentences)
            self.idx_in_subset += np.arange(len(sentences)).tolist()

    def get_all_subsets(self):
        return set(self.subsets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = {
            "id": idx,
            "subset": self.subsets[idx],
            "text": self.data[idx],
            "idx_in_subset": self.idx_in_subset[idx],
        }
        if self.use_chat_template:
            datum["non_chat_templated_text"] = datum["text"]
            datum["text"] = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": datum["text"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        tokens = self.tokenizer(datum["text"], truncation=True, padding=False)
        datum.update(tokens)
        return datum


def get_json_subsets_dataset(
    *args,
    # file_path=Path("data/giraffe_eagle_situations.json"),
    subsets=None,
    tokenizer=None,
    use_chat_template: bool = False,
    **kwargs,
) -> torch.utils.data.Dataset:
    assert tokenizer is not None, "Must pass a tokenizer"
    return (
        JsonSubsetsDataset(
            subsets=subsets,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            **kwargs,
        ),
        DictCollatorWithPadding(tokenizer),
    )


def get_json_from_tabular_dataset(
    datasets_folder: Path,
    split: str,
    subsets: t.Set[str],
    tokenizer: transformers.PreTrainedTokenizer,
    file_path: t.Union[str, t.List[str]],
    query: t.Optional[str] = None,
    use_chat_template: bool = False,
    **kwargs,
) -> t.Tuple[torch.utils.data.Dataset, t.Callable]:
    """
    Convert tabular data (parquet files) to JSON format and return as JsonSubsetsDataset.

    This function reads tabular parquet files and converts them to the JSON format
    used by giraffes.json, extracting columns whose names are present in subsets.
    Supports filtering by query expressions (e.g., "jailbreak == 'aim'").

    Args:
        datasets_folder: Path to datasets folder (unused but required by interface)
        split: Dataset split (unused but required by interface)
        subsets: Set of column names to extract as subsets (e.g., {"jailbroken_prompt", "forbidden_prompt"})
        tokenizer: Tokenizer for text processing
        file_path: Path(s) to parquet file(s) - can be string or list of strings
        query: Optional pandas query string for filtering (e.g., "jailbreak == 'aim'")
        **kwargs: Additional arguments

    Returns:
        Tuple of (JsonSubsetsDataset, collator)
    """
    import logging
    import tempfile
    from .tabular_dataset import load_tabular_data

    logger = logging.getLogger(__name__)

    # Resolve file paths first
    if isinstance(file_path, str):
        resolved_file_path = str(Path(file_path))
    else:
        # List of file paths - resolve each one
        resolved_file_path = [str(Path(fp)) for fp in file_path]

    # Load tabular data using the existing utility
    combined_df = load_tabular_data(resolved_file_path, query=query)

    # Validate that subset columns exist in the dataframe
    available_columns = set(combined_df.columns)
    subset_columns = set(subsets)
    missing_columns = subset_columns - available_columns
    if missing_columns:
        raise ValueError(f"Missing required subset columns: {missing_columns}")

    # Extract data for each subset column
    data = {}
    for column_name in subset_columns:
        column_data = combined_df[column_name].dropna().tolist()
        data[column_name] = column_data
        logger.info(f"Extracted {len(column_data)} entries for subset '{column_name}'")

    # Write to temporary JSON file for JsonSubsetsDataset
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2, ensure_ascii=False)
        temp_json_path = Path(tmp_file.name)

    logger.info(f"Created temporary JSON with keys: {list(data.keys())}")
    logger.info(f"Sizes: {[len(v) for v in data.values()]}")

    # Create JsonSubsetsDataset from the temporary file
    dataset = JsonSubsetsDataset(
        json_path=temp_json_path,
        subsets=list(data.keys()),  # Load all subsets by default
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        **kwargs,
    )

    # Clean up temporary file
    temp_json_path.unlink()

    return dataset, DictCollatorWithPadding(tokenizer)
