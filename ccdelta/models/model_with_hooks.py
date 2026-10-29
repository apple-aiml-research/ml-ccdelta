#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#


import re
import json
import typing as t
from pathlib import Path
from dataclasses import dataclass
from threading import Thread

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from ccdelta.hooks import get_hook
from ccdelta.hooks.responses_hook import ResponsesHook
from ccdelta.utils.utils import is_module_name_in_regex
from ccdelta.utils.logging import get_logger
from ccdelta.utils.hook_factory import create_or_load_hook

logger = get_logger(__name__)

TASK_MAPPING = {
    "text-generation": AutoModelForCausalLM,
    "sequence-classification": AutoModelForSequenceClassification,
}


@dataclass
class GenCtx:
    special_mask_cur: torch.Tensor | None = None  # [B,S] True = special
    cur_tokens: torch.Tensor | None = None  # [B,S]
    is_decode_call: bool = False
    generation_index: int = 0  # Tracks tokens generated after prompt (0 during prompt)
    _control_ids: list[int] | None = None
    _control_tokens: list[int] | None = None
    _special_ids: set[int] | None = None  # cached set (minus EOS if allowed)
    _special_toks: set[int] | None = None  # cached set (minus EOS if allowed)


def collect_control_ids(
    tokenizer, include_whitespace: bool = True, include_system_prompts: bool = True
) -> list[int]:
    """
    Return a superset of 'control' token ids, including:
    - HF-declared specials in the backend 'added_tokens' with special=true
    - Tokens referenced in tokenizer.chat_template like <|start_header_id|>, etc.
    - Optionally: whitespace/formatting characters like \n, \t, etc.
    - Optionally: common system prompt boilerplate phrases

    Args:
        tokenizer: The tokenizer instance
        include_whitespace: If True, include newlines, tabs, and other formatting chars
        include_system_prompts: If True, include tokens from common system prompt phrases
    """
    ids: set[int] = set(tokenizer.all_special_ids or [])

    # 1) Parse backend 'added_tokens' to capture specials not in special_tokens_map
    try:
        data = json.loads(tokenizer.backend_tokenizer.to_str())
        for t in data.get("added_tokens", []):
            if t.get("special"):
                tok = t.get("content")
                tid = t.get("id")
                if tok is not None and tid is not None:
                    ids.add(int(tid))
    except Exception:
        pass  # backend not available; keep going

    # 2) From chat_template: pull any <|...|> literals
    tmpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tmpl, str):
        print("CHAT TEMPLATE", tmpl)
        for m in re.finditer(r"<\|[^|>]+?\|>", tmpl):
            tok = m.group(0)
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)

    # 3) Add whitespace/formatting characters if requested
    if include_whitespace:
        # Common whitespace and formatting characters
        whitespace_chars = [
            "\n",  # newline (LF)
            "\r",  # carriage return (CR)
            "\r\n",  # Windows newline (CRLF)
            "\t",  # tab
            "  ",  # double space
            "   ",  # triple space
        ]

        for char in whitespace_chars:
            # Try to encode the character
            try:
                # Some tokenizers might split these, so we try encoding
                encoded = tokenizer.encode(char, add_special_tokens=False)
                # Only add if it's a single token (not split)
                if len(encoded) == 1:
                    ids.add(encoded[0])
                    logger.debug(
                        f"Added whitespace token '{repr(char)}' with ID {encoded[0]}"
                    )
            except Exception:
                pass  # Skip if encoding fails

    # 4) Add common system prompt phrases if requested
    if include_system_prompts:
        # Common boilerplate phrases in system prompts across models
        system_phrases = [
            # Llama system prompts
            "Cutting Knowledge Date:",
            "Today Date:",
            # Gemma system prompt
            "user",
            "model",
            # Qwen
            "system",
            "<|im_start|>",
            "<|im_end|>",
        ]

        for phrase in system_phrases:
            try:
                # Encode phrase without special tokens
                encoded = tokenizer.encode(phrase, add_special_tokens=False)
                # Add all tokens from this phrase
                ids.update(encoded)
                logger.debug(
                    f"Added {len(encoded)} tokens from system phrase '{phrase[:30]}...'"
                )
            except Exception:
                pass  # Skip if encoding fails

    return list(ids)


class ModelWithHooks:
    """
    Class wrapping a Pytorch model so that we can apply forward hooks on its responses.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        hooks: t.Optional[t.List[t.Callable]] = None,
        tokenizer=None,
        device: str = None,
        **kwargs,
    ) -> None:
        """
        Initializes the ModelWithHooks instance with a Pytorch module and optional hooks.

        Args:
            module (nn.Module): The PyTorch module to be wrapped.
            hooks (Optional[List[ResponsesHook]]): List of hooks to apply on the responses. Default is None.
            device (str): The device where the model should run (e.g., "cpu", "cuda:0"). Default is None.

        Returns:
            None
        """
        self.hooks: t.Dict[str, torch.nn.Module] = (
            {h.module_name: h for h in hooks} if hooks is not None else {}
        )
        self.module = module
        self._forward_hook_handles: t.List[RemovableHandle] = []
        self._postprocessing_threads: t.List[Thread] = []
        self.device = device
        self.tokenizer = module.tokenizer
        self.ctx = GenCtx()

    @staticmethod
    def find_module_names(
        model: torch.nn.Module,
        regex_list: t.List[str],
        skip_first: int = 0,  # first layer tends to be the model root
    ) -> t.List[str]:
        """
        Finds module names of a PyTorch model that match any of the given regular expressions.

        Args:
            model (nn.Module): The PyTorch model.
            regex_list (List[str]): List of regular expressions to match module names against.
            skip_first (int): Number of modules to skip at the beginning. Default is 0, meaning no modules are skipped.

        Returns:
            List[str]: Sorted list of matching module names.
        """
        module_names = [name[0] for name in model.named_modules()][skip_first:]
        _module_names = []
        for module_name in module_names:
            if module_name == "":
                continue
            matches = is_module_name_in_regex(module_name, regex_list)
            for match in matches:
                if match not in _module_names:
                    _module_names.append(match)
        return _module_names

    def find_module_state_paths(
        self, path: Path, regex_list: t.List[str], strict: bool = False
    ) -> t.List[Path]:
        """
        Find matching state dict files in a directory based on given regular expressions and verify consistency with model module names.

        Args:
        - path (Path): The directory path to search for state dict files.
        - regex_list (List[str]): A list of regular expressions used to match state dict file names.
        - strict (bool, optional): If True, raise an error if any state dict does not match a module name. Default is False.

        Returns:
        - List[Path]: A list of matching state dict file paths.

        Raises:
        - RuntimeError: If strict is True and any state dict does not match a module name.
        """
        all_state_paths = list(Path(path).glob("*.statedict"))
        _regex_list = [f"{pattern}(:[0-9]+)?" for pattern in regex_list]
        ret = []
        for state_path in all_state_paths:
            for pattern in _regex_list:
                if re.fullmatch(pattern, state_path.stem):
                    ret.append(Path(state_path))
                    break
        if strict:
            module_names = self.find_module_names(self.module, regex_list)
            for state_path in ret:
                found = False
                for module_name in module_names:
                    if str(module_name) in str(state_path):
                        found = True
                        break
                if not found:
                    raise RuntimeError("State dicts do not match all module names")
        logger.info(f"Found {len(ret)} state dicts in {path}")
        return ret

    @staticmethod
    def sort_module_names(
        model: torch.nn.Module,
        module_names: t.List[str],
    ) -> t.List[str]:
        """
        Sorts a list of module names based on their order in the PyTorch model graph.

        Args:
            model (nn.Module): The PyTorch model.
            module_names (List[str]): List of module names to sort.

        Returns:
            List[str]: Sorted list of module names.
        """
        sorted_module_names = [name[0] for name in model.named_modules()]
        indices = [sorted_module_names.index(v) for v in module_names]
        sorted_indices = np.sort(indices)
        output = [sorted_module_names[i] for i in sorted_indices]
        return output

    def get_module(self) -> torch.nn.Module:
        """
        Gets the wrapped PyTorch module.

        Returns:
            nn.Module: The wrapped PyTorch module.
        """
        return self.module

    @staticmethod
    def make_pre_forward_top(ctx, tokenizer):
        """
        Pre-forward hook that computes:
        - ctx.special_mask_cur: [B, S] bool, True where token is special/control (incl. EOS if allow_eos=False)
        - ctx.is_decode_call: True if S==1 (typical during generate's step-wise decode)
        """

        # Build once
        ctx._control_ids: list[int] = list(set(collect_control_ids(tokenizer)))
        ctx._control_tokens = sorted(ctx._control_ids)
        ctx._control_tokens = [
            tokenizer.convert_ids_to_tokens(i) for i in ctx._control_ids
        ]
        print("_control_ids", ctx._control_ids)
        print("_control_tokens:", ctx._control_tokens)

        control_ids_tensor = None  # materialized per-device

        def _pre_forward_top(_module, _args, kwargs):
            # Support positional calls too
            ids = kwargs.get("input_ids", None)
            if ids is None and _args:
                # common: forward(input_ids, attention_mask, ...)
                ids = _args[0] if isinstance(_args[0], torch.Tensor) else None

            if ids is None:
                ctx.special_mask_cur = None
                ctx.is_decode_call = False
                return None

            # Enforce long dtype for isin
            if ids.dtype != torch.long:
                ids = ids.long()
                kwargs["input_ids"] = (
                    ids  # keep kwargs consistent for downstream forward
                )

            B, S = ids.shape
            device = ids.device

            nonlocal control_ids_tensor
            if (control_ids_tensor is None) or (control_ids_tensor.device != device):
                # Empty-guard to keep torch.isin happy
                if len(ctx._control_ids) == 0:
                    control_ids_tensor = torch.empty(0, dtype=torch.long, device=device)
                else:
                    control_ids_tensor = torch.tensor(
                        ctx._control_ids, dtype=torch.long, device=device
                    )

            special_mask = torch.isin(ids, control_ids_tensor)  # [B, S] bool
            ctx.special_mask_cur = special_mask
            ctx.is_decode_call = S == 1

            # Track generation index: reset on prompt, increment on decode
            if S == 1:  # Decode phase (generating new tokens)
                ctx.generation_index += 1
            else:  # Prompt phase (processing input)
                ctx.generation_index = 0  # Reset counter

            # Debug-only: recovering tokens is slow, so guard it.
            if getattr(ctx, "debug_tokens", False):
                ids_cpu = ids.detach().cpu()
                ctx.cur_tokens = [
                    [tokenizer.convert_ids_to_tokens(int(t)) for t in row]
                    for row in ids_cpu.tolist()
                ]

            return None

        return _pre_forward_top

    def register_hooks(
        self,
        hooks=None,
    ):
        """
        Registers forward hooks on the modules of the network with the given names.

        Args:
            hooks (Optional[List[ResponsesHook]]): List of hooks to register. Default is None.

        Returns:
            None
        """
        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        if hooks is not None:
            assert len(self.hooks) == 0, "Hooks already registered"
            self.hooks: t.Dict[str, torch.nn.Module] = (
                {h.module_name: h for h in hooks} if hooks is not None else {}
            )

        logger.info(f"Pre-Registering global hook.")
        h_top = self.module.register_forward_pre_hook(
            self.make_pre_forward_top(self.ctx, self.tokenizer),
            with_kwargs=True,
        )

        logger.info(f"Registering {len(self.hooks)}.")
        self._forward_hook_handles = [h_top]
        module_dict = {name: module for name, module in self.module.named_modules()}
        logger.debug(f"Found {len(module_dict)} available modules")
        for module_name in self.hooks:
            base_module_name = module_name.split(":")[0]
            if base_module_name in module_dict:
                module = module_dict[base_module_name]
                hook_fn = self.hooks[module_name]
                # Attach gen_ctx to the hook.
                # This is a little bit hacky but is the easiest way to guarantee that
                # this happens everyhere.
                hook_fn.gen_ctx = self.ctx
                logger.debug(f"Registering forward hook on module: {base_module_name}")
                logger.debug(f"Hook function: {hook_fn}")
                handle = module.register_forward_hook(hook_fn)
                self._forward_hook_handles.append(handle)
            else:
                logger.warning(f"Module {base_module_name} not found in model!")

    def get_hooks(self) -> list:
        return self.hooks

    def set_hooks(self, hooks: dict) -> None:
        self.remove_hooks()
        self.hooks = hooks

    def remove_hooks(self):
        """
        Removes all registered hooks and joins any remaining threads.

        Returns:
            None
        """
        logger.info(f"Removing {len(self.hooks)} hooks.")
        for h in self._forward_hook_handles:
            h.remove()
        for h in self.hooks.values():
            try:
                h.join()
            except:
                pass
        self._forward_hook_handles = []
        self.hooks = {}

    def load_hooks_from_folder(
        self,
        folder: Path,
        module_names: t.List[str],
        hook_type: str,
        **hook_params,
    ) -> t.Dict[str, t.Callable]:
        assert self.hooks == {}, "Trying to overwrite existing hooks!"
        # Create hooks
        state_paths = self.find_module_state_paths(
            path=Path(folder),
            regex_list=module_names,
            strict=True,
        )
        for state_path in state_paths:
            saved_module_name = str(Path(state_path).stem)
            hook = create_or_load_hook(
                hook_name=hook_type,
                module_name=saved_module_name,
                base_hook_params=hook_params,
                state_path=state_path,
            )
            self.hooks[saved_module_name] = hook
        logger.info(
            f"Loaded {len(state_paths)}/{len(list(Path(folder).glob('*.statedict')))} hooks from {folder}"
        )
        assert (
            len(self.hooks) > 0 or hook_type == "none"
        ), f"No hooks loaded from {folder}!"
        return list(self.hooks.values())

    def get_hook_outputs(self):
        """
        Gets the outputs of all registered hooks.

        Returns:
            Dict: Dictionary mapping module names to their corresponding outputs.
        """
        outputs = {}
        for module_name, hook in self.hooks.items():
            outputs.update(hook.outputs)
        return outputs

    def update_hooks(self, *args, **kwargs):
        assert len(self.hooks) > 0
        for module_name, hook in self.hooks.items():
            try:
                hook.join()
            except:
                pass
            logger.debug(f"Updating hook for module: {module_name}")
            hook.update(*args, **kwargs)

    def get_tokenizer(self):
        """
        Gets the tokenizer module associated with this model.

        Returns:
            Optional[nn.Module]: The tokenizer module, or None if no tokenizer is available.
        """
        return None

    def get_target_module_names(self) -> t.List[str]:
        """
        Returns a list of names of the modules to which hooks are registered.

        Returns:
            List[str]: A list of module names that have associated hooks.
        """
        return list(self.hooks.keys())

    def print_module_names_responses_sizes(self) -> None:
        """
        Prints the names and sizes of the responses for each registered hook.

        This method is not yet implemented and will raise a NotImplementedError if called.

        Raises:
            NotImplementedError: Indicates that this method has not been implemented yet.
        """
        raise NotImplementedError

    def __call__(self, batch):
        """
        Forward pass for huggingface models. This method is called when an instance of the ModelWithHooks class is invoked as a function.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input tensors for the model. It should include 'input_ids' and 'attention_mask' keys with corresponding tensor values.

        Returns:
            Dict: The output of the model after processing the inputs. This is typically a dictionary or tuple depending on the model architecture, but here it is assumed to return a dictionary containing the model outputs.
        """
        input_ids, attention_mask = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


TASK_REGISTRY = {
    "text-generation": ModelWithHooks,
    "text-classification": ModelWithHooks,
    "question-answering": ModelWithHooks,
}


def get_model_with_hooks(
    module: torch.nn.Module,
    hooks: t.Optional[t.List[ResponsesHook]] = None,
    model_task: str = None,
    device: str = None,
    **model_params: DictConfig,
) -> ModelWithHooks:
    """
    Returns a hookable model for the given task.

    This function takes in a PyTorch module and returns a model with the
    specified hooks for a specific task. If no task is provided, it defaults to None.

    Args:
        module (torch.nn.Module): The PyTorch module that needs to be processed.
        hooks (t.Optional[t.List[ResponsesHook]], optional): A list of hooks to apply on the model. Defaults to None.
        model_task (str, optional): The specific task for which the model is being created. Defaults to None.
        device (str, optional): The device where the computation will take place. Defaults to None.

    Returns:
        ModelWithHooks: A model with specified hooks for the given task.
    """
    return TASK_REGISTRY[model_task](
        module=module,
        hooks=hooks,
        device=device,
        **model_params,
    )
