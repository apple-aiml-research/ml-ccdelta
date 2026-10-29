# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import ccdelta.hooks as hooks
from ccdelta.datasets import get_dataloader, get_dataset
from ccdelta.datasets.responses_io import ResponsesLoader
from ccdelta.datasets.paired_responses_io import PairedResponsesLoader
from ccdelta.models import get_model
from ccdelta.models.model_with_hooks import get_model_with_hooks
from ccdelta.utils import utils
from ccdelta.utils.hook_factory import create_or_load_hook
from ccdelta.utils.logging import get_logger

# Get logger with auto-configured environment variable support
logger = get_logger(__name__)


class ResponsesManager:
    """Computes and saves responses from specified modules within a pretrained language model.

    This class handles loading a pretrained model, setting up custom hooks to extract responses from desired modules,
    and iterating over a dataset to compute and save the extracted responses.

    Attributes:
        cfg (DictConfig): Configuration object containing model parameters, task settings, data paths, and intervention details.
        output_path (Path): Path where computed responses will be saved.
        model (nn.Module): Loaded pretrained language model with registered hooks for response extraction.
        module_names (List[str]): List of module names from which to extract responses.

    Args:
        cfg (DictConfig): Configuration object defining the model, task, and intervention parameters.

    Example usage:

    ```python
    # Assuming cfg is a loaded DictConfig object
    responses_manager = ResponsesManager(cfg)
    output_path = responses_manager.compute_responses()
    logger.info(f"Responses saved to: {output_path}")
    ```

    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_path = self.get_output_path(cfg)

        assert cfg.device != "cuda" or torch.cuda.is_available()
        logger.info("Loading model to extract responses.")
        # Models and Tokenizers
        module, tokenizer = get_model(
            cache_dir=cfg.cache_dir,
            device=cfg.device,
            rand_weights=False,
            model_task=cfg.task_params.model_task,
            **cfg.model_params,
        )
        logger.info("Done loading model.")
        self.model = get_model_with_hooks(
            module=module,
            model_task=cfg.task_params.model_task,
            device=cfg.device,
            **cfg.model_params,
        )
        self.tokenizer = tokenizer

        # Datasets and dataloaders - handle different dataset types appropriately
        dataset_type = cfg.task_params.dataset

        # Use learning_dataset params if available, otherwise fall back to dataset_params for backward compatibility
        learning_params = cfg.task_params.get(
            "learning_dataset", cfg.task_params.get("dataset_params", {})
        )
        logger.info(
            f"Using dataset params for intervention learning: {learning_params.get('file_path', 'N/A')}"
        )

        use_chat_template = cfg.get("use_chat_template", False)

        if dataset_type == "tabular-paired":
            # Paired datasets: no balanced sampling concept
            train_dataset, collate_fn = get_dataset(
                name=dataset_type,
                datasets_folder=Path(cfg.data_dir),
                split="train",
                subsets=set(cfg.task_params.src_subsets + cfg.task_params.dst_subsets),
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                **learning_params,
            )

            self.dataloader = get_dataloader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=cfg.shuffle,
                # No balanced parameter for paired datasets
                seed=cfg.seed,
            )

        elif dataset_type in ["json-subsets", "tabular_to_json"]:
            # Subset-based datasets: support balanced sampling
            train_dataset, collate_fn = get_dataset(
                name=dataset_type,
                datasets_folder=Path(cfg.data_dir),
                split="train",
                subsets=set(cfg.task_params.src_subsets + cfg.task_params.dst_subsets),
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                **learning_params,
            )

            self.dataloader = get_dataloader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=cfg.shuffle,
                balanced=cfg.get("balanced_data", True),
                seed=cfg.seed,
            )

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        self.module_names = self.get_module_names(cfg.model_params.module_names)

    def get_module_names(self, module_names: t.List[str]):
        """Get the names of modules in the model.

        Args:
            module_names (list of str): A regex list of module names.

        Returns:
            None
        """
        return self.model.find_module_names(self.model.module, module_names)

    def _set_hooks(
        self, module_names: t.List[str], extra_hooks: t.Optional[list] = None
    ) -> None:
        """Registers hooks to extract responses from specified modules and save them.

        This method iterates through the list of desired module names (`module_names`) and registers a forward hook on each module.
        The hook function, defined using `hooks.get_hook`, captures the output of the module during the forward pass
        and saves it to disk along with other relevant metadata specified in `self.cfg.save_fields`.

        The `pooling_op` parameter determines how the outputs from different modules are aggregated.  It can be a single string
        representing a pooling operation (e.g., "mean", "max") or a list of strings for more complex aggregation schemes.

        Args:
            module_names (t.List[str]): A list of module names within the model from which to extract responses.
            extra_hooks (list, optional): A list of additional hook functions to register. Defaults to None.

        Raises:
            KeyError: If a specified module name does not exist within the model.

        """
        if isinstance(self.cfg.intervention_params.pooling_op, str):
            pooling_op = [self.cfg.intervention_params.pooling_op]
        else:
            pooling_op = self.cfg.intervention_params.pooling_op

        # Use explicit hook configuration from pipeline_hooks
        hook_name = self.cfg.intervention_params.name

        # Build common kwargs for response collection hooks
        common_kwargs = {
            "pooling_op_names": pooling_op,
            "output_path": self.output_path,
            "save_fields": ["id"] + self.cfg.save_fields,
            "threaded": False,
        }

        # Add tokenizer and inner_hook_class for paired hooks
        if hook_name == "paired_sae_responses":
            # Use SAEActivationsHook for SAE-based paired responses
            from ccdelta.hooks.sae_activations_hook import SAEActivationsHook

            common_kwargs["inner_hook_class"] = SAEActivationsHook
            common_kwargs["tokenizer"] = self.tokenizer
        elif hook_name == "paired_dense_responses":
            # Use IdentityHook for dense paired responses
            from ccdelta.hooks.identity import IdentityHook

            common_kwargs["inner_hook_class"] = IdentityHook
            common_kwargs["tokenizer"] = self.tokenizer
        elif hook_name == "paired_sae_error_analysis":
            # Use SAEReconstructionErrorHook for SAE reconstruction error analysis
            from ccdelta.hooks.sae_reconstruction_error_hook import SAEReconstructionErrorHook

            common_kwargs["inner_hook_class"] = SAEReconstructionErrorHook
            common_kwargs["tokenizer"] = self.tokenizer
            # Note: collect_error_stats is set via config hook_params, not here

        # Create hooks using universal function (automatically handles SAE vs non-SAE)
        logger.info(f"Creating {hook_name} hooks for {len(module_names)} modules")
        hook_fns = [
            create_or_load_hook(
                hook_name=hook_name,
                module_name=module_name,
                base_hook_params=self.cfg.intervention_params.hook_params,
                **common_kwargs,
            )
            for module_name in module_names
        ]

        if extra_hooks is not None:
            hook_fns += extra_hooks
        self.model.remove_hooks()
        self.model.register_hooks(hook_fns)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Returns the output path for the computed responses."""
        # Sanity check, only allowing multiplicity for args.module_names and args.subset
        model_name = Path(cfg.model_params.model_path).name
        # Setting paths
        return Path(cfg.cache_dir) / cfg.tag / model_name / cfg.task_params.dataset

    def compute_responses(
        self,
        module_names: t.Optional[t.List[str]] = None,
        extra_hooks: t.Optional[list] = None,
    ) -> Path:
        """Computes and saves responses from specified modules in the model.

        This method orchestrates the process of extracting responses from designated modules within the model during inference. It handles hook registration, batch processing, and saving the extracted responses to disk.

        Args:
            module_names (List[str], optional): List of module names from which to extract responses. If None, defaults to the configured module names (`self.module_names`).
            extra_hooks (list, optional): Additional hook functions to register during computation. Defaults to an empty list.

        Returns:
            Path: The path where the computed responses are saved.

        """
        if module_names is None:
            module_names = self.module_names

        self._set_hooks(module_names, extra_hooks or [])
        logger.info(f"Computing responses for {len(module_names)} modules.")
        logger.debug("\n".join(module_names))

        current_batch = 0
        max_batches = (
            len(self.dataloader)
            if self.cfg.max_batches is None
            else min(len(self.dataloader), self.cfg.max_batches)
        )
        if current_batch == self.cfg.max_batches:
            logger.info("All batches found, nothing to compute")
        else:
            if current_batch > 0:
                logger.info(f"Resuming from batch {current_batch}")
            else:
                logger.info("Computing batch responses")

            iloader = iter(self.dataloader)
            for idx in tqdm(range(max_batches), desc="Computing responses"):
                batch = next(iloader)
                logger.debug(f"Batch {idx} keys: {list(batch.keys())}")
                if idx >= current_batch:
                    with torch.inference_mode():
                        # Check if this is a paired batch
                        if self._is_paired_batch(batch):
                            logger.debug(f"Processing paired batch {idx}")
                            self._process_paired_batch(idx, batch)
                        else:
                            # Regular processing
                            logger.debug(f"Processing regular batch {idx}")
                            self.model.update_hooks(batch_idx=idx, batch=batch)
                            try:
                                self.model(batch)
                            except hooks.custom_exceptions.TargetModuleReached:
                                pass
            logger.info("Done")
        return self.output_path

    def _is_paired_batch(self, batch: dict) -> bool:
        """Check if batch contains paired data."""
        has_source = "source_input_ids" in batch
        has_target = "target_input_ids" in batch
        is_paired = has_source and has_target
        logger.debug(
            f"Batch check - source: {has_source}, target: {has_target}, is_paired: {is_paired}"
        )
        return is_paired

    def _process_paired_batch(self, batch_idx: int, batch: dict):
        """Process a paired batch with sequential source and target model calls."""
        logger.debug(f"Processing paired batch {batch_idx}")

        # First pass: source texts
        source_tokens_list = []
        for i in range(batch["source_input_ids"].size(0)):
            input_ids = batch["source_input_ids"][i]
            attention_mask = batch["source_attention_mask"][i]
            actual_ids = input_ids[attention_mask == 1]
            source_tokens_list.append(
                self.tokenizer.convert_ids_to_tokens(actual_ids.tolist())
            )

        source_batch = {
            "input_ids": batch["source_input_ids"],
            "attention_mask": batch["source_attention_mask"],
            "tokens": source_tokens_list,
            # Include other metadata
            **{
                k: v
                for k, v in batch.items()
                if k
                not in [
                    "source_input_ids",
                    "source_attention_mask",
                    "target_input_ids",
                    "target_attention_mask",
                ]
            },
        }

        logger.debug(f"Source batch keys: {list(source_batch.keys())}")
        logger.debug(f"Source batch input_ids shape: {source_batch['input_ids'].shape}")

        self.model.update_hooks(batch_idx=batch_idx, batch=source_batch, is_source=True)
        try:
            logger.debug("Calling model with source batch")
            self.model(source_batch)
        except hooks.custom_exceptions.TargetModuleReached:
            pass

        # Second pass: target texts
        target_tokens_list = []
        for i in range(batch["target_input_ids"].size(0)):
            input_ids = batch["target_input_ids"][i]
            attention_mask = batch["target_attention_mask"][i]
            actual_ids = input_ids[attention_mask == 1]
            target_tokens_list.append(
                self.tokenizer.convert_ids_to_tokens(actual_ids.tolist())
            )

        target_batch = {
            "input_ids": batch["target_input_ids"],
            "attention_mask": batch["target_attention_mask"],
            "tokens": target_tokens_list,
            # Include other metadata
            **{
                k: v
                for k, v in batch.items()
                if k
                not in [
                    "source_input_ids",
                    "source_attention_mask",
                    "target_input_ids",
                    "target_attention_mask",
                ]
            },
        }

        logger.debug(f"Target batch keys: {list(target_batch.keys())}")
        logger.debug(f"Target batch input_ids shape: {target_batch['input_ids'].shape}")

        self.model.update_hooks(batch_idx=batch_idx, batch=target_batch, is_target=True)
        try:
            logger.debug("Calling model with target batch")
            self.model(target_batch)
        except hooks.custom_exceptions.TargetModuleReached:
            pass


class InterventionsManager:
    """Manages the learning and application of interventions on a generative model.

    This class handles loading model responses, configuring intervention parameters,
    and learning intervention strategies for specific modules within the model.

    Interventions are techniques used to modify the behavior of a model by injecting biases
    or constraints during inference. They can be used for tasks like domain adaptation,
    style transfer, or controlling the output of a model.

    Attributes:
        responses_path (Path): The path to the directory where model responses are stored.
        cfg (DictConfig): A configuration object containing parameters for the interventions manager.

    """

    def __init__(self, responses_path: Path, cfg: DictConfig):
        """Initialize the InterventionsManager.

        Args:
            responses_path (Path): The path to the directory containing model response data.
            cfg (DictConfig): A configuration object defining intervention parameters and settings.

        Returns:
            None
        """
        self.responses_path = responses_path
        self.cfg = cfg

        self.output_path = self.get_output_path(cfg)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Determine the output directory for storing intervention data.

        Args:
            cfg (DictConfig): The configuration object containing intervention parameters.

        Returns:
            Path: The path to the directory where intervention results will be saved.
        """
        model_name = Path(cfg.model_params.model_path).name
        sorted_keys = list(sorted(cfg.intervention_params.keys()))
        name = "-".join(
            [
                str(cfg.intervention_params[k])
                for k in sorted_keys
                if k
                not in [
                    "hook_params",
                    "state_path",
                    "executor_params",
                    "learn_intervention",
                    "experiment_id",
                ]
            ]
        )
        return Path(cfg.cache_dir) / cfg.tag / model_name / name

    def _learn_intervention(self, module_name: str):
        """Learn an intervention strategy for a specific module.

        This method loads relevant model responses, applies the intervention to those responses,
        and learns the parameters of the intervention strategy based on the modified data.

        Args:
            module_name (str): The name of the module within the language model for which the intervention will be learned.

        Returns:
            None
        """
        # Load responses for a given module
        data_subset = self.responses_loader.load_data_subset(
            {
                "module_names": [module_name],
                "pooling_op": [self.cfg.intervention_params.pooling_op],
                "subset": list(self.cfg.task_params.src_subsets)
                + list(self.cfg.task_params.dst_subsets),
            },
            num_workers=0,
        )
        data_subset = ResponsesLoader.label_src_dst_subsets(
            data_subset,
            src_subset=self.cfg.task_params.src_subsets,
            dst_subset=self.cfg.task_params.dst_subsets,
            balanced=True,
            seed=self.cfg.seed,
        )
        z = torch.tensor(data_subset["responses"])
        y = torch.tensor(data_subset["label"]).to(torch.bool)

        hook = create_or_load_hook(
            hook_name=self.cfg.intervention_params.name,
            module_name=module_name,
            base_hook_params=self.cfg.intervention_params.hook_params,
        )

        # Estimate hook parameters
        hook.fit(
            responses=z,
            labels=y,
            **self.cfg.intervention_params.hook_params,
        )

        # Save hook
        self.output_path.mkdir(exist_ok=True, parents=True)
        hook.save_state_dict(state_path=self.output_path / (module_name + ".statedict"))

    def _learn_paired_intervention(self, module_name: str):
        """Learn an intervention strategy for paired data.

        This method loads paired responses and learns interventions based on
        feature differences between source and target pairs, rather than
        using random src/dst labeling.

        Args:
            module_name: The name of the module for which the intervention will be learned.
        """
        logger.info(f"Learning paired intervention for module: {module_name}")

        # Load paired data
        data_subset = self.responses_loader.load_data_subset(
            {
                "module_names": [module_name],
                "pooling_op": [self.cfg.intervention_params.get("pooling_op", "mean")],
            },
            num_workers=0,
        )

        # For paired data, pass the list of dictionaries directly to the hook
        # The SAEFeatureSpaceHook.fit() method expects and handles this format
        paired_responses = data_subset[
            "responses"
        ]  # List of dicts with source/target features
        # Create dummy labels since they're not used for paired data learning
        y = torch.ones(len(paired_responses), dtype=torch.bool)

        logger.info(
            f"Loaded {len(paired_responses)} paired samples for intervention learning"
        )

        hook = create_or_load_hook(
            hook_name=self.cfg.intervention_params.name,
            module_name=module_name,
            base_hook_params=self.cfg.intervention_params.hook_params,
        )

        # Estimate hook parameters
        hook.fit(
            responses=paired_responses,
            labels=y,
            **self.cfg.intervention_params.hook_params,
        )

        # Save hook
        self.output_path.mkdir(exist_ok=True, parents=True)
        hook.save_state_dict(state_path=self.output_path / (module_name + ".statedict"))

    def learn_intervention_all(self) -> None:
        """Learn interventions on all modules specified in the configuration.

        Args:
            None

        Returns:
            None
        """
        logger.info("Learning all interventions at once.")

        # Determine if this is paired intervention learning based on dataset type
        is_paired = self.cfg.task_params.dataset == "tabular-paired"

        if is_paired:
            logger.info("Using paired intervention learning approach")
            # Use PairedResponsesLoader for paired data
            self.responses_loader = PairedResponsesLoader(
                root=self.responses_path,
                columns=["responses", "id", "module_names"],  # + self.cfg.load_fields,
            )
            # Get module names from paired data
            module_names = self.responses_loader.available_modules
            # Filter by requested patterns
            filtered_module_names = []
            for pattern in self.cfg.model_params.module_names:
                for module_name in module_names:
                    if pattern in module_name:
                        filtered_module_names.append(module_name)

            # Learn paired interventions
            for module_name in filtered_module_names:
                self._learn_paired_intervention(module_name=module_name)
        else:
            # Standard intervention learning
            self.responses_loader = ResponsesLoader(
                root=self.responses_path,
                from_folders=[
                    f"*/*/{self.cfg.intervention_params.get('pooling_op', 'mean')}",
                ],
                columns=["responses", "id", "subset"],  # + self.cfg.load_fields,
            )
            # Get only the module names requested via args
            module_names = self.responses_loader.get_attribute_values(
                "module_names", filter_patterns=self.cfg.model_params.module_names
            )
            for module_name in module_names:
                self._learn_intervention(module_name=module_name)

    def learn_intervention_all_incremental(self, responses_manager: "ResponsesManager"):
        """Learn interventions incrementally for all specified modules.

        This method allows for learning interventions on a module-by-module basis,
        leveraging previously learned interventions from previous modules when learning a new one.

        Supports both paired and non-paired datasets:
        - Paired (tabular-paired): Uses PairedResponsesLoader + _learn_paired_intervention()
        - Non-paired (json-subsets, tabular_to_json): Uses ResponsesLoader + _learn_intervention()

        Args:
            responses_manager (ResponsesManager): A manager to handle and load responses from files.

        Returns:
            None
        """
        logger.info("Learning interventions incrementally")

        # Detect paired vs non-paired data
        is_paired = self.cfg.task_params.dataset == "tabular-paired"
        if is_paired:
            logger.info("Using paired intervention learning approach (incremental)")

        # Get only the module names requested via args
        module_names = responses_manager.get_module_names(
            module_names=self.cfg.model_params.module_names
        )
        # Here we load intervention hooks for all those layers already computed (used_module_names).
        used_module_names: t.List[str] = []
        for module_name in module_names:
            intervention_hooks = []
            for used in used_module_names:
                state_path = self.output_path / f"{used}.statedict"
                # These hooks will intervene on the model, that's why we need hook_* args.
                hook = create_or_load_hook(
                    hook_name=self.cfg.intervention_params.name,
                    module_name=used,
                    base_hook_params=self.cfg.intervention_params.hook_params,
                    state_path=state_path,
                )
                intervention_hooks.append(hook)
            logger.info(
                f"Loaded {len(intervention_hooks)} hooks: [{''.join(used_module_names[:1])}:{''.join(used_module_names[-1:])}]"
            )
            # Register all the hooks.
            logger.info(f"New inference on {module_name}.")

            # 1. Save responses for current module
            responses_manager.compute_responses(
                [module_name], extra_hooks=intervention_hooks
            )

            if is_paired:
                self.responses_loader = PairedResponsesLoader(
                    root=self.responses_path,
                    columns=["responses", "id", "module_names"],
                )
                self._learn_paired_intervention(module_name)
            else:
                self.responses_loader = ResponsesLoader(
                    root=self.responses_path,
                    from_folders=[
                        f"*/*/{self.cfg.intervention_params.get('pooling_op', 'mean')}",
                    ],
                    columns=["responses", "id", "subset"],
                )
                self._learn_intervention(module_name)

            # 3. Finally add the current module name in the used ones
            used_module_names.append(module_name)


@hydra.main(config_path="../configs", version_base="1.3")
def main(cfg: DictConfig) -> t.Optional[InterventionsManager]:
    return learn_intervention(cfg)


def learn_intervention(cfg: DictConfig) -> t.Optional[InterventionsManager]:
    """Learn interventions on a model from a given configuration.

    Args:
        cfg (DictConfig): The configuration parameters for the intervention learning process.

    Returns:
        InterventionsManager: An instance of the InterventionsManager class if any interventions were learned, otherwise None.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Set random seed
    utils.seed_all(cfg.seed)

    # Check if intervention requires learning
    requires_learning = cfg.interventions.intervention_params.get(
        "learn_intervention", True
    )

    print(f"requires_learning = {requires_learning}")

    if not requires_learning:
        print(
            f"Intervention '{cfg.interventions.intervention_params.name}' does not require learning. Skipping response collection and training."
        )
        return None

    if cfg.interventions.intervention_params.incremental == "incr":
        cfg.responses.intervention_params.hook_params.raise_exception = True
        logger.warning(
            "Responses hook will raise exception to speedup incremental training."
        )
    responses_manager = ResponsesManager(cfg.responses)
    interventions_manager = InterventionsManager(
        ResponsesManager.get_output_path(cfg.responses), cfg.interventions
    )

    if cfg.interventions.intervention_params.incremental == "atonce":
        if cfg.compute_responses:
            responses_manager.compute_responses()
        interventions_manager.learn_intervention_all()
    elif cfg.interventions.intervention_params.incremental == "incr":
        if not cfg.compute_responses:
            raise RuntimeError(
                "Cannot learn incremental interventions with cfg.compute_responses=false."
            )
        interventions_manager.learn_intervention_all_incremental(responses_manager)
    else:
        raise ValueError(f"Unknown incremental mode {cfg.interventions.incremental}")
    return interventions_manager


if __name__ == "__main__":
    main()
