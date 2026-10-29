# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import typing as t
from ccdelta.datasets.tabular_dataset import TabularDataset
from ccdelta.datasets.paired_collator import PairedDictCollator


class TabularPairedDataset(TabularDataset):
    """A PyTorch Dataset for loading paired tabular data for intervention learning.

    This dataset extends TabularDataset to handle paired text data where each row
    contains both source and target text columns. Used for SAE feature extraction
    with matched pairs for computing feature differences.

    Attributes:
        source_column (str): Column name containing source/desired text
        target_column (str): Column name containing target/undesired text
    """

    def __init__(
        self,
        file_path: str | list[str],
        source_column: str,
        target_column: str,
        max_rows: t.Optional[int] = None,
        query: t.Optional[str] = None,
        tokenizer=None,
        use_chat_template: bool = True,
    ):
        """Initialize TabularPairedDataset.

        Args:
            file_path (str | list[str]): Path(s) to the tabular data file(s) (parquet, JSON, etc.).
                If multiple files are provided, they will be concatenated.
            source_column (str): Column name containing source/desired text
            target_column (str): Column name containing target/undesired text
            max_rows (int, optional): Maximum number of rows to load
            query (str, optional): Pandas query string to filter the data after loading
            tokenizer (optional): Tokenizer for tokenization and optional chat templating.
            add_special_tokens (bool): Whether to add special tokens when applying chat template.
            use_chat_template (bool): Whether to apply chat templating to prompts.
        """
        # Initialize parent with source as prompt_column and pass all parameters
        super().__init__(
            file_path,
            source_column,
            "output",
            max_rows,
            query=query,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )

        if target_column not in self.data.columns:
            raise ValueError(
                f"Column '{target_column}' not found in data. Available columns: {list(self.data.columns)}"
            )

        self.source_column = source_column
        self.target_column = target_column
        self.data_type = "tabular_paired"

    def __getitem__(self, idx):
        """Return paired data for a single row."""
        row = self.data.iloc[idx]

        # Extract both texts, handling None/NaN values
        source_text = row[self.source_column]
        if self._is_null(source_text):
            source_text = ""

        target_text = row[self.target_column]
        if self._is_null(target_text):
            target_text = ""

        original_source = str(source_text)
        original_target = str(target_text)

        if self.use_chat_template:
            try:
                source_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": original_source}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                source_text = original_source
            try:
                target_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": original_target}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                target_text = original_target

        row_data = row.to_dict()  # For mapping back to original row

        # Add the following so we can do our substring search without the chat template token
        # noise.
        row_data["original_source"] = original_source
        row_data["original_target"] = original_target

        tokenized = self.tokenizer(
            original_target,
            add_special_tokens=False,
        )
        row_data["original_target_token_ids"] = tokenized["input_ids"]
        row_data["original_target_tokens"] = self.tokenizer.convert_ids_to_tokens(
            tokenized["input_ids"]
        )

        result = {
            "id": idx,
            "source_text": str(source_text),
            "target_text": str(target_text),
            "non_chat_templated_source": original_source,
            "non_chat_templated_target": original_target,
            "row_data": row_data,
        }
        return result

    def _is_null(self, value):
        """Check if value is None, NaN, or empty."""
        import pandas as pd

        return pd.isna(value) or value is None

    def get_paired_texts(self, idx):
        """Get both source and target texts for a given index."""
        item = self[idx]
        return item["source_text"], item["target_text"]


def get_tabular_paired_dataset(
    datasets_folder,
    split: str,
    subsets: t.Set[str],
    tokenizer,
    file_path: str | list[str],
    source_column: str,
    target_column: str,
    query: t.Optional[str] = None,
    use_chat_template: bool = False,
    **kwargs,
) -> t.Tuple[TabularPairedDataset, t.Callable]:
    """Factory function to create TabularPairedDataset and collator.

    Args:
        datasets_folder: Base datasets folder (unused for tabular data)
        split: Dataset split (unused for tabular data)
        subsets: Subsets to load (unused for tabular data)
        tokenizer: Tokenizer for text processing
        file_path: Path(s) to tabular data file(s). If multiple files provided, they will be concatenated.
        source_column: Name of source text column
        target_column: Name of target text column
        query: Optional pandas query string to filter the data after loading
        **kwargs: Additional arguments passed to dataset

    Returns:
        Tuple of (dataset, collate_fn)
    """
    # Handle both single file path and list of file paths
    from pathlib import Path

    if isinstance(file_path, str):
        resolved_file_path = str(Path(file_path))
    else:
        # List of file paths - resolve each one
        resolved_file_path = [str(Path(fp)) for fp in file_path]

    dataset = TabularPairedDataset(
        file_path=resolved_file_path,
        source_column=source_column,
        target_column=target_column,
        query=query,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        **kwargs,
    )

    collate_fn = PairedDictCollator(tokenizer)

    return dataset, collate_fn
