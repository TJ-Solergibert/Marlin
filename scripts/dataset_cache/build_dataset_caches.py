import os

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import get_blend_from_list
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronCorePretrainingSampler,
    MegatronPretrainingRandomSampler,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.distributed import init_process_group
from torch.utils.data import DataLoader


def build_train_valid_test_datasets(cfg, tokenizer):
    # (bool: get_attention_mask_from_fusioneod_mask_loss: bool, reset_attention_mask: bool, reset_position_ids: bool, seed: int, global_batch_size: int, max_train_steps: int, sequence_length: int, path_to_cache: str, data_prefix: List, no_seqlen_plus_one_input_tokens: bool = False):
    logging.info("Building GPT datasets.")
    global_batch_size = cfg.model.global_batch_size
    max_train_steps = cfg.trainer.max_steps
    # NOTE(tj.solergibert) Never run any validation or test loops
    eval_iters = 0
    test_iters = 0

    # TODO: @athitten make num of eval and test samples 1 always, after it works with non DictConfig data_prefix.
    train_valid_test_num_samples = [
        max_train_steps * global_batch_size,
        eval_iters * global_batch_size,
        test_iters * global_batch_size,
    ]

    # Function needed for mcore GPTDataset
    is_dataset_built_on_rank = lambda: True

    add_extra_token = not cfg.model.data.get("no_seqlen_plus_one_input_tokens", False)
    kwargs = {
        "random_seed": cfg.model.seed,
        "sequence_length": cfg.model.data.seq_length,
        "path_to_cache": cfg.model.data.index_mapping_dir,
        "tokenizer": tokenizer,
        "reset_position_ids": cfg.model.data.get("reset_position_ids", False),
        "reset_attention_mask": cfg.model.data.get("reset_attention_mask", False),
        "eod_mask_loss": cfg.model.data.get("eod_mask_loss", False),
        "create_attention_mask": not cfg.model.get(
            "get_attention_mask_from_fusion", True
        ),
        "mmap_bin_files": cfg.model.data.get("mmap_bin_files", True),
        "drop_last_partial_validation_sequence": cfg.model.data.get(
            "validation_drop_last", True
        ),
        "num_dataset_builder_threads": cfg.model.data.get(
            "num_dataset_builder_threads", 1
        ),
        "renormalize_blend_weights": cfg.model.data.get(
            "renormalize_blend_weights", False
        ),
        "add_extra_token_to_sequence": add_extra_token,
        "goldfish_loss": cfg.model.data.get("goldfish_loss", False),
        "goldfish_k": cfg.model.data.get("goldfish_k", 4),
        "goldfish_h": cfg.model.data.get("goldfish_h", 13),
    }

    kwargs["blend"] = get_blend_from_list(cfg.model.data.data_prefix)
    kwargs["split"] = "100,0,0"  # NOTE(tj.solergibert) Use ALL data for training

    dataset_config = GPTDatasetConfig(**kwargs)
    dataset_config.mock = False

    _train_ds, _validation_ds, _test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_valid_test_num_samples,
        is_dataset_built_on_rank,
        dataset_config,
    ).build()

    if _train_ds is not None:
        logging.info(f"Length of train dataset: {len(_train_ds)}")
    if _validation_ds is not None:
        logging.info(f"Length of val dataset: {len(_validation_ds)}")
    if _test_ds is not None:
        logging.info(f"Length of test dataset: {len(_test_ds)}")
    logging.info("Finished building GPT datasets.")

    return _train_ds, _validation_ds, _test_ds


def build_pretraining_data_loader(
    cfg,
    dataset,
    consumed_samples,
    dataset_type=None,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
):
    """Build dataloader given an input dataset."""

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
    # Megatron sampler
    if (
        hasattr(cfg.model.data, "dataloader_type")
        and cfg.model.data.dataloader_type is not None
    ):
        if cfg.model.data.dataloader_type == "single":
            batch_sampler = MegatronCorePretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=cfg.model.micro_batch_size,
                data_parallel_rank=int(os.environ.get("RANK", 0)),
                data_parallel_size=int(os.environ.get("WORLD_SIZE", 1)),
                drop_last=drop_last,
                global_batch_size=cfg.model.global_batch_size,
                rampup_batch_size=cfg.model.get("rampup_batch_size", None),
                pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
            )
        elif cfg.model.data.dataloader_type == "cyclic":
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=cfg.model.micro_batch_size,
                data_parallel_rank=int(os.environ.get("RANK", 0)),
                data_parallel_size=int(os.environ.get("WORLD_SIZE", 1)),
                drop_last=drop_last,
            )
        else:
            raise ValueError(
                'cfg.model.data.dataloader_type must be "single" or "cyclic"'
            )
    else:
        raise ValueError(
            'cfg.model.data.dataloader_type not found. Must be "single" or "cyclic"'
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.model.data.num_workers > 0 else False,
    )


def setup_training_data(cfg, train_ds):
    consumed_samples = 0
    logging.info(
        f"Setting up train dataloader with len(len(self._train_ds)): {len(train_ds)} and consumed samples: {consumed_samples}"
    )
    _train_dl = build_pretraining_data_loader(cfg, train_ds, consumed_samples)
    return _train_dl


@hydra_runner(config_path="configs", config_name="Llama3-base")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    init_process_group(
        backend="gloo",
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        rank=int(os.environ.get("RANK", 0)),
    )

    tokenizer = get_nmt_tokenizer(
        library=cfg.model.tokenizer.library,
        model_name=cfg.model.tokenizer.get("type", None),
        use_fast=cfg.model.tokenizer.get("use_fast", False),
        delimiter=cfg.model.tokenizer.get("delimiter", None),
        special_tokens=cfg.model.tokenizer.get("special_tokens", None),
        trust_remote_code=cfg.model.tokenizer.get("trust_remote_code", False),
        legacy=False,
        chat_template=getattr(cfg.model.tokenizer, "chat_template", None),
    )
    (
        train_ds,
        _,
        _,
    ) = build_train_valid_test_datasets(cfg, tokenizer)
    _ = setup_training_data(cfg, train_ds)
    logging.info(f"Prepared dataset cache in: {cfg.model.data.index_mapping_dir}")


if __name__ == "__main__":
    main()
