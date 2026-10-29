# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import logging
import pickle
import typing as t
from pathlib import Path
import os
import hydra

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from omegaconf import DictConfig

# Local imports
from ccdelta.models import get_model
from ccdelta.models.model_with_hooks import ModelWithHooks
from ccdelta.utils.defense import sample_and_apply_defense


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_results(results: t.Dict, output_path: str) -> None:
    """
    Saves results dictionary as pickle named after args used for evaluation run in cfg.output_dir (which is created if it does not exist yet)
    Args:
        results: nested dictionary containing the results to be saved
        args: args used for evaluation run

    Returns: None

    """
    out_file = Path(output_path)
    logger.info(f"Saving eleuther eval results to {out_file}")
    with out_file.open("wb") as fp:
        pickle.dump(results, fp)


class DefenseModel(HFLM):
    """
    Custom HFLM wrapper that applies defense transformations to prompts.
    Works with both apply_chat_template=True and apply_chat_template=False.
    """

    def __init__(
        self,
        *args,
        defense: bool = False,
        use_chat_template: bool = False,
        payload_last: bool = False,
        **kwargs,
    ):
        """
        Args:
            defense: Whether to apply defense or not to text.
            *args, **kwargs: Arguments passed to HFLM
        """
        super().__init__(*args, **kwargs)
        self.defense = defense
        self.payload_last = payload_last
        self.use_chat_template = use_chat_template

        if self.defense:
            print(f"DefenseModel initialized with defense method: {self.defense}")

    def _apply_defense_to_text(self, text: str) -> str:
        """Apply defense to a text string"""
        assert isinstance(text, str), "Cannot apply defense to text type not a string."
        if self.defense:
            try:
                return sample_and_apply_defense(text, payload_last=self.payload_last)
            except Exception as e:
                raise RuntimeError(f"Warning: Failed to apply defense: {e}")
        return text

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> list[int]:
        """
        Override tok_encode to apply defense before encoding.
        This handles the apply_chat_template=False case.
        """
        # Apply defense to the string before encoding
        # Do not apply if use_chat_template==True, since the defense is applied at
        # apply_chat_template() level (before tokenizing)
        defensed_string = (
            string if self.use_chat_template else self._apply_defense_to_text(string)
        )
        logger.debug(f"\ntok_encode {self.defense}\n{defensed_string}\n")
        return super().tok_encode(
            defensed_string, left_truncate_len, add_special_tokens
        )

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> tuple:
        """
        Override tok_batch_encode to apply defense to batch of strings.
        This handles the apply_chat_template=False case for batch encoding.
        """
        # Apply defense to all strings in the batch.
        # Do not apply if use_chat_template==True, since the defense is applied at
        # apply_chat_template() level (before tokenizing)
        if self.defense and not self.use_chat_template:
            defensed_strings = [self._apply_defense_to_text(s) for s in strings]
        else:
            defensed_strings = strings
        logger.debug(f"\ntok_batch_encode {self.defense}\n{defensed_strings}\n")
        return super().tok_batch_encode(
            defensed_strings, padding_side, left_truncate_len, truncation
        )

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Override apply_chat_template to apply defense to user messages.
        This handles the apply_chat_template=True case.
        """
        if self.defense and self.use_chat_template and chat_history:
            # Create a copy to avoid modifying the original
            chat_history = [msg.copy() for msg in chat_history]
            chat_history_check = [msg.copy() for msg in chat_history]

            # Apply defense to all user messages
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i].get("role") == "user":
                    original_content = chat_history[i]["content"]
                    defensed_content = self._apply_defense_to_text(original_content)
                    chat_history[i]["content"] = defensed_content
                    break

            assert chat_history != chat_history_check, (
                "The chat templates haven't changed! Make sure the template has the field 'user'."
            )

        return super().apply_chat_template(chat_history, add_generation_prompt)


def evaluate(cfg: DictConfig) -> dict:
    """

    Args:
        args: the argument namespace from the argparser

    Returns: a results object which is the standard output of the lm-eval-harness

    """
    module, tokenizer = get_model(
        model_path=cfg.model_params.model_path,
        cache_dir=cfg.cache_dir,
        dtype=cfg.dtype,
        device=cfg.device,
        model_task="text-generation",
        seq_len=getattr(cfg, "seq_len", 5000),
    )
    tokenizer.padding_side = "left"

    assert cfg.model_params.module_names is not None, logging.error(
        f"Intervention specified as {cfg.intervention_params.name}, but no module names passed (passed {cfg.model_params.module_names})"
    )

    # Create hooked model
    model_with_hooks = ModelWithHooks(
        module=module,
    )

    if cfg.intervention_params.state_path is not None:
        model_with_hooks.load_hooks_from_folder(
            folder=Path(cfg.intervention_params.state_path),
            module_names=cfg.model_params.module_names,
            hook_type=cfg.intervention_params.name,
            **cfg.intervention_params.hook_params,
        )
    model_with_hooks.register_hooks()
    logger.info("The hooks attached for mmlu evaluation are:")
    for key in model_with_hooks.hooks:
        logger.info(model_with_hooks.hooks[key])
    model = model_with_hooks.module

    logger.info(f"Eval batch size: {cfg.batch_size}")

    use_chat_template = cfg.get("use_chat_template", True)
    use_defense = cfg.get("use_defense", False)
    defense_payload_last = cfg.get("defense_payload_last", True)

    lm = DefenseModel(
        model,
        defense=use_defense,
        payload_last=defense_payload_last,
        use_chat_template=use_chat_template,
        tokenizer=tokenizer,
        device=cfg.device,
        max_length=getattr(cfg, "seq_len", 5000),
        batch_size=cfg.batch_size,
        max_batch_size=32,
        dtype=cfg.dtype,
    )
    task_list = list(cfg.tasks)

    # Run evaluation
    results = evaluator.simple_evaluate(
        lm,
        tasks=task_list,
        num_fewshot=cfg.num_fewshot,
        limit=cfg.limit,  # can set a limit for quicker testing
        bootstrap_iters=cfg.bootstrap_iters,  # for statistical significance estimation
        random_seed=cfg.rs,
        numpy_random_seed=cfg.nrs,
        torch_random_seed=cfg.trs,
        device=cfg.device,
        # batch_size=cfg.batch_size,
        cache_requests=False,  # Not using cache to avoid conflicts when using defense on multiple eval runs.
        apply_chat_template=use_chat_template,
        task_manager=None,  # Will create a fresh task manager
    )

    return results


def run_eleuther_eval(cfg: DictConfig, task: str = "") -> None:
    results = evaluate(cfg)

    # remove samples to reduce eval file size.
    # write them to a different file
    samples = results.pop("samples")

    # Save complete results dict as pickle
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir) / output_path
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    save_results(results, os.path.join(output_path, f"eleuther_{task}.pkl"))
    save_results(samples, os.path.join(output_path, f"eleuther_{task}_samples.pkl"))

    # Console printing of summarized results
    logger.info(
        f"{cfg.model_params.model_path}, limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}"
    )
    logger.info(make_table(results))
    if "groups" in results:
        logger.info(make_table(results, "groups"))


@hydra.main(
    config_path="../act/configs", config_name="text_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    run_eleuther_eval(cfg)


if __name__ == "__main__":
    main()
