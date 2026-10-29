# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Applies prompt defense templates to OR-bench parquet chunks."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from ccdelta.utils.defense import sample_and_apply_defense

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def apply_prompt_defense_orbench(input_dir: Path, output_dir: Path, payload_last: bool) -> None:
    input_files = sorted(input_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Found {len(input_files)} parquet files in {input_dir}")

    for file_path in input_files:
        df = pd.read_parquet(file_path)
        logger.info(f"Processing {file_path.name} ({len(df)} rows)")

        df["jailbroken_prompt"] = df["jailbroken_prompt"].apply(
            lambda text: sample_and_apply_defense(
                text=text,
                prompt_templates=None,
                split_str=None,
                payload_last=payload_last,
            )
        )

        out_path = output_dir / file_path.name
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {out_path}")

    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply prompt defense to OR-bench chunks")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/or_bench/chunks_seed42"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/or_bench/chunks_seed42_prompt_defense"),
    )
    parser.add_argument(
        "--payload-last",
        action="store_true",
        default=True,
        help="Put payload (user query) last in the defense template (default: True)",
    )
    args = parser.parse_args()
    apply_prompt_defense_orbench(args.input_dir, args.output_dir, args.payload_last)
