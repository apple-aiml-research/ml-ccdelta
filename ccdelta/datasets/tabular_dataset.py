# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import typing as t
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .collators import CollationMixin


def load_tabular_data(
    file_path: str | list[str], query: str | None = None
) -> pd.DataFrame:
    """Load tabular data from local files.

    Args:
        file_path: Single file path or list of file paths to load and concatenate
        query: Optional pandas query string to filter the data after loading

    Returns:
        pd.DataFrame: Loaded data, concatenated if multiple files provided, filtered if query provided
    """
    # Handle single file path (backward compatibility)
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = file_path

    if not file_paths:
        raise ValueError("At least one file path must be provided")

    dataframes = []
    for path in file_paths:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            df = (
                pd.DataFrame(data)
                if isinstance(data, list)
                else pd.json_normalize(data)
            )
        elif path.endswith(".csv"):
            # Try different delimiters for CSV files,
            try:
                df = pd.read_csv(path, index_col=0)
            except:
                try:
                    df = pd.read_csv(path, delimiter=";", index_col=0)
                except Exception as exc:
                    raise ValueError(f"Failed to read CSV file {path}: {exc}")
        else:
            raise ValueError(f"Unsupported tabular data format for file: {path}")
        dataframes.append(df)

    # Concatenate all dataframes
    if len(dataframes) == 1:
        combined_df = dataframes[0]
    else:
        # Reset index to avoid duplicate indices when concatenating
        combined_df = pd.concat(dataframes, ignore_index=True)

    # Apply query filter if provided
    if query is not None:
        try:
            print(f"Applying query filter: {query}")
            num_original_rows = len(combined_df)

            # Instrumentation: Show jailbreak distribution before filtering
            if "jailbreak" in combined_df.columns:
                jailbreaks_before = set(combined_df["jailbreak"].unique())
                print(
                    f"  Jailbreak types BEFORE filter ({len(jailbreaks_before)}): {sorted(jailbreaks_before)}"
                )

            combined_df = combined_df.query(query)
            num_filtered_rows = len(combined_df)
            print(
                f"Query filtered data from {num_original_rows} to {num_filtered_rows} rows"
            )

            # Instrumentation: Show jailbreak distribution after filtering
            if "jailbreak" in combined_df.columns:
                jailbreaks_after = set(combined_df["jailbreak"].unique())
                print(
                    f"  Jailbreak types AFTER filter ({len(jailbreaks_after)}): {sorted(jailbreaks_after)}"
                )

                # Verify exclusion for wrapper_exclude_* filters
                excluded = jailbreaks_before - jailbreaks_after
                if excluded:
                    print(f"  EXCLUDED jailbreak types: {sorted(excluded)}")

            # Reset index after filtering to maintain consecutive indices
            combined_df = combined_df.reset_index(drop=True)
        except Exception as e:
            raise ValueError(f"Query '{query}' failed: {e}")

    return combined_df


class TabularDataset(Dataset, CollationMixin):
    """A PyTorch Dataset for loading structured tabular data with prompt extraction.

    This dataset loads tabular data from files (parquet & JSON) using pandas,
    extracts prompts from a specified column, and maintains the original data structure
    for result merging.

    Attributes:
        data (pd.DataFrame): The loaded tabular data.
        prompt_column (str): The column name containing prompts for text generation.
    """

    def __init__(
        self,
        file_path: str | list[str],
        prompt_column: str,
        output_column: str,
        max_rows: t.Optional[int] = None,
        query: t.Optional[str] = None,
        tokenizer=None,
        use_chat_template: bool = True,
    ):
        """Initializes TabularDataset.

        Args:
            file_path (str | list[str]): The path(s) to the data file(s) (parquet, JSON, CSV, etc.).
                If multiple files are provided, they will be concatenated.
            prompt_column (str): The column name containing prompts for generation.
            output_column (str): The column name for generated text output.
            max_rows (int, optional): Maximum number of rows to load. If None, all rows are loaded.
            query (str, optional): Pandas query string to filter the data after loading.
            tokenizer (optional): Tokenizer instance. If provided and has chat_template, applies chat templating to prompts.

        """
        self.data = load_tabular_data(file_path, query=query)

        if prompt_column not in self.data.columns:
            raise ValueError(
                f"Column '{prompt_column}' not found in data. Available columns: {list(self.data.columns)}"
            )

        if max_rows and max_rows < len(self.data):
            self.data = self.data.head(max_rows)

        self.prompt_column = prompt_column
        self.output_column = output_column
        self.data_type = "tabular"
        self.tokenizer = tokenizer

        self.use_chat_template = tokenizer is not None and hasattr(
            tokenizer, "apply_chat_template"
        )

        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return row data in unified dict format with minimal type conversion."""
        row = self.data.iloc[idx]

        # Handle None/NaN values in prompt column - prompts must be strings
        prompt_value = row[self.prompt_column]
        if pd.isna(prompt_value) or prompt_value is None:
            prompt_value = ""  # Default to empty string for generation

        original_prompt = prompt_value
        if self.use_chat_template:
            try:
                prompt_value = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": original_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                prompt_value = original_prompt

        # Convert row to dict with minimal processing
        row_dict = row.to_dict()
        clean_row_dict = {}

        for key, value in row_dict.items():
            if isinstance(value, (list, np.ndarray)):
                clean_row_dict[key] = value
            elif pd.isna(value) or value is None:
                # Convert None/NaN to PyTorch-compatible values
                if pd.api.types.is_numeric_dtype(self.data[key].dtype):
                    clean_row_dict[key] = float("nan")
                else:
                    # For non-numeric columns, use empty string instead of None
                    # PyTorch collation can't handle None values
                    clean_row_dict[key] = ""
            else:
                # Preserve all other types as-is (numpy types, dicts, etc.)
                clean_row_dict[key] = value

        return {
            "prompt": prompt_value,
            "non_chat_templated": original_prompt,
            "row_data": clean_row_dict,
            "row_index": idx,
        }

    def augment_row(
        self, row_data: dict, generation: str, param_value: t.Union[float, str]
    ) -> dict:
        """Augment a single row with generation results.

        Args:
            row_data: Individual row data (already reconstructed from collated batch)
            generation: Generated text for this row
            param_value: Parameter value used for generation

        Returns:
            Augmented row dictionary
        """
        result = {
            "prompt": row_data["prompt"],
            "row_index": row_data["row_index"],
            "generation_param": param_value,
            self.output_column: generation,
        }

        # Add original data (flattened from row_data)
        for key, value in row_data["row_data"].items():
            result[f"{key}"] = value

        return result


def create_tabular_dataloader(
    file_path: str | list[str],
    prompt_column: str,
    output_column: str = "steered_text",
    batch_size: int = 1,
    max_rows: t.Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    query: t.Optional[str] = None,
    tokenizer=None,
    use_chat_template: bool = False,
) -> DataLoader:
    """Create DataLoader for tabular data with prompt extraction.

    Args:
        file_path: Single file path or list of file paths to load and concatenate
        prompt_column: Column name containing prompts for generation
        output_column: Column name for generated text output
        batch_size: Batch size for DataLoader
        max_rows: Maximum number of rows to load
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for DataLoader
        query: Optional pandas query string to filter the data after loading
        tokenizer: Optional tokenizer for tokenization and optional chat templating.
        use_chat_template: Whether to apply chat templating to prompts.
    Returns:
        DataLoader for the tabular dataset
    """
    dataset = TabularDataset(
        file_path,
        prompt_column,
        output_column,
        max_rows=max_rows,
        query=query,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
