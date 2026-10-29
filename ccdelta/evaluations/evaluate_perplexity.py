# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import logging
import os
import typing as t
from pathlib import Path
from sys import platform

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ccdelta.models import get_model
from ccdelta.utils import utils
from ccdelta.utils.perplexity import measure_perplexity
from ccdelta.datasets.tabular_dataset import load_tabular_data

if platform == "darwin":
    # Xavi: MacOS, remove dynamo errors, or test_perplexity will fail.
    # Xavi: This happened for transformers==4.44.2
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True


logging.getLogger().setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


@torch.inference_mode()
def evaluate(cfg: DictConfig) -> t.Dict[str, float]:
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir, output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # Set random seed
    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Setup device and distributed learning
    if cfg.device in ["cuda", None] and torch.cuda.is_available():
        cfg.device = "cuda"
    elif cfg.device == "cuda":
        raise (RuntimeError("Cuda not available"))
    elif cfg.device is None:
        cfg.device = "cpu"

    # Models and Tokenizers
    module, tokenizer = get_model(
        model_path=cfg.perplexity_model_path,
        cache_dir=cfg.data_dir,
        device=cfg.device,
        dtype=cfg.dtype,
        rand_weights=False,
        seq_len=cfg.seq_len,
        model_task="text-generation",
    )
    # module = torch.compile(module)

    # Load data using tabular dataset functionality
    df = load_tabular_data(cfg.data_path)
    model_name = Path(cfg.perplexity_model_path).name

    # Support both nested list and flat list configurations for column_sentences
    # Check if first element is itself a list/ListConfig (indicating nested structure)
    if (
        cfg.column_sentences
        and len(cfg.column_sentences) > 0
        and hasattr(cfg.column_sentences[0], "__len__")
        and not isinstance(cfg.column_sentences[0], str)
    ):
        # Nested list: [["steered_text"], ["steered_text", "prompt"]]
        eval_configs = cfg.column_sentences
    else:
        # Flat list (backward compatibility): ["steered_text", "prompt"]
        eval_configs = [cfg.column_sentences]

    # Run perplexity evaluation for each configuration
    for config in eval_configs:
        sentences = df[config[0]].values.tolist()
        sentences = [s.replace("<s> ", "") for s in sentences]

        if len(config) > 1:
            prompts = df[config[1]].values.tolist()
            prompts = [s.replace("<s> ", "").strip() for s in prompts]
        else:
            prompts = None

        # Create config name from column names
        config_name = "_".join(config)

        logging.info(
            f"Computing PPL ({config_name}) with {cfg.perplexity_model_path} on {len(sentences)} sentences."
        )

        ppl = measure_perplexity(
            continuations=sentences,
            prompts=prompts,
            model=module,
            tokenizer=tokenizer,
            device=cfg.device,
            batch_size=cfg.batch_size,
            autoregressive=(
                cfg.intervention_params.hook_params.intervention_position == "last"
            ),
        )

        # Add PPL column with descriptive name
        col = f"ppl_{config_name}_{model_name}"
        if col in df.columns:
            col = col + "-v2"
        df[col] = ppl
        logging.info(f"Average {config_name} ppl_{model_name}: {df[col].mean()}")

    logging.info(df)
    if output_path is not None:
        output_file = Path(output_path) / "model_perplexity.parquet"
        df.to_parquet(output_file)
        logging.info(output_file)

    return df


@hydra.main(
    config_path="../act/configs", config_name="text_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
