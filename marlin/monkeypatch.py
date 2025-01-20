import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_current_global_batch_size
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.utils import drain_embedding_wgrad_compute
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.parts.utils_funcs import get_last_rank


def _training_step(self, dataloader_iter):
    """
    We pass the dataloader iterator function to the micro-batch scheduler.
    The input batch to each micro-batch is fetched using the dataloader function
    in the micro-batch fwd function.
    """
    # Initialize userbuffer communicators.
    if self.initialize_ub:
        self.initialize_ub_func()

    # Reset learning rate
    if self.if_init_step and self.reset_lr:
        num_groups = len(self._optimizer.param_groups)
        for group in range(num_groups):
            self._optimizer.param_groups[group]["lr"] = (
                0.0 if self.cfg.optim.sched.warmup_steps > 0 else self.cfg.optim.lr
            )
        self._optimizer.param_groups[0]["reset_lr"] = {
            "num_steps": self.trainer.global_step,
            "reset_lr_steps": True if self.reset_lr_steps else False,
            "if_init_step": self.if_init_step,
        }
        self.if_init_step = False

    if self.rampup_batch_size:
        current_global_batch_size = get_current_global_batch_size()
        # do validation and save the checkpoint when gbs is changed
        if self.prev_global_batch_size != current_global_batch_size and self.prev_global_batch_size:
            self.trainer.should_stop = True

    # zero out the mcore grad buf
    if self.use_mcore_dist_optim:
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()

    # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
    self._optimizer.zero_grad()

    if self.with_distributed_adam and not self.use_mcore_dist_optim:
        # hack to enable overlapping param sync and forward compute
        # note: the distributed optimizer monkey-patches each
        # parameter's __getattribute__ function so that it can
        # launch parameter all-gathers the first time the
        # parameter is accessed after the optimizer step. However,
        # PyTorch directly passes embedding parameters into a C++,
        # bypassing this process. A quick-and-dirty hack is to
        # manually interact with the parameter.
        modules = self.model if isinstance(self.model, list) else [self.model]
        for module in modules:
            if isinstance(module, (Float16Module, MCoreFloat16Module)):
                module = module.module
            if not self.mcore_gpt:
                module = module.language_model

            if hasattr(module, "embedding"):
                for param in module.embedding.parameters():
                    param.data_ptr()

    if self.cfg.get("pipeline_model_parallel_size", 1) > 1 and parallel_state.is_pipeline_last_stage(
        ignore_virtual=True
    ):
        if (
            self.cfg.get("defer_embedding_wgrad_compute", False) and self.mcore_gpt and not self.use_mcore_dist_optim
        ):  # Silently ignore the optimization if MCORE is not used
            module_list = self.get_model_module_list()
            if len(module_list) > 1:
                embedding_module = module_list[-1]
            else:
                embedding_module = module_list[0]

            embedding_module.embedding_activation_buffer.clear()
            assert (
                len(embedding_module.embedding_activation_buffer) == 0
            ), "When you defer wgrads, this buffer should not hold stray activations"

    loss_mean = self.training_step_fwd_bwd_step_call(dataloader_iter, forward_only=False)

    if self.cfg.get("fp8", False):
        self.prev_step_training = self.training

    # Optimization: Defer the embedding GEMM Wgrads of the last PP stage to pipeline flush waiting time
    if self.cfg.get("pipeline_model_parallel_size", 1) > 1 and parallel_state.is_pipeline_last_stage(
        ignore_virtual=True
    ):
        if (
            self.cfg.get("defer_embedding_wgrad_compute", False) and self.mcore_gpt and not self.use_mcore_dist_optim
        ):  # Silently ignore the optimization if MCORE is not used
            module_list = self.get_model_module_list()
            if len(module_list) > 1:
                embedding_module = module_list[-1]
            else:
                embedding_module = module_list[0]

            embedding_activation_buffer = embedding_module.embedding_activation_buffer
            grad_output_buffer = embedding_module.grad_output_buffer
            if self.cfg.get("share_embeddings_and_output_weights", True):
                weight = embedding_module.shared_embedding_or_output_weight()
            else:
                weight = embedding_module.output_layer.weight

            drain_embedding_wgrad_compute(
                embedding_module.config, embedding_activation_buffer, grad_output_buffer, weight
            )

    # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
    if self.cfg.get("tensor_model_parallel_size", 1) > 1 and self.cfg.get("sequence_parallel", False):
        # Mcore DistOpt handles this, so we don't have to
        if not self.use_mcore_dist_optim:
            self.megatron_timer_start("allreduce_sequence_parallel_gradients", log_level=1)
            self.allreduce_sequence_parallel_gradients()
            self.megatron_timer_stop("allreduce_sequence_parallel_gradients")

    self.megatron_timer_start("gradient_allreduce", log_level=1)
    if self.use_fsdp:
        # Reduce the gradients omitted from FSDP-sharding
        self.allreduce_fsdp_sharding_omitted_gradients()
    elif self.with_distributed_adam:
        if not self.use_mcore_dist_optim:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        # else: Mcore distributed optim calls finalize_model_grads to finish grad sync
    elif self.megatron_amp_O2:
        # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
        if (
            self.cfg.get("pipeline_model_parallel_size", 1) > 1
            or self.cfg.get("sequence_parallel", False)
            or not self.cfg.get("async_grad_allreduce", True)
        ):
            # main grads are stored in the MainParamsOptimizer wrapper
            self._optimizer.allreduce_main_grads()
    else:
        # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
        # so we all-reduce gradients after the pipeline
        self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)
    self.megatron_timer_stop("gradient_allreduce")

    if (
        not self.use_mcore_dist_optim
        and self.cfg.get("pipeline_model_parallel_size", 1) > 1
        and self.cfg.get("share_embeddings_and_output_weights", True)
    ):
        self.megatron_timer_start("allreduce_first_last_embeddings", log_level=1)
        # when using pipeline parallelism the first and last stage must keep embeddings in sync
        self.allreduce_first_last_embeddings()
        self.megatron_timer_stop("allreduce_first_last_embeddings")

    if self.log_memory_usage:
        max_memory_reserved = torch.cuda.max_memory_reserved()
        memory_allocated = torch.cuda.memory_allocated()
        self.log(
            "peak_memory_usage",
            max_memory_reserved,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        self.log(
            "memory_allocated",
            memory_allocated,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

    ## logging
    if self.log_train_loss:
        # When using pipeline parallelism, loss is calculated only in the last pipeline stage and
        # it should be casted to other pipeline stages for logging.
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if torch.distributed.get_rank() == get_last_rank():
                torch.distributed.send(loss_mean, 0)
            elif torch.distributed.get_rank() == 0:
                torch.distributed.recv(loss_mean, get_last_rank())
        self.log("reduced_train_loss", loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)

        # (@adithyare) we need to check for the _scaler attribute to enable pp>1 for adapter training
        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log("loss_scale", loss_scale, batch_size=1)

    lr = self._optimizer.param_groups[0]["lr"]
    self.log("lr", lr, rank_zero_only=True, batch_size=1)
    self.log(
        "global_step",
        self.trainer.global_step,
        prog_bar=True,
        rank_zero_only=True,
        batch_size=1,
    )

    consumed_samples = self._compute_consumed_samples_after_training_step()
    # TODO: make sure compute_consumed_samples works for pipeline parallelism
    self.log(
        "consumed_samples",
        consumed_samples,
        prog_bar=True,
        rank_zero_only=True,
        batch_size=1,
    )

    # NOTE(tj.solergibert) Log consumed tokens
    ###################################################################
    self.log(
        "consumed_tokens",
        int(consumed_samples * self.cfg.get("encoder_seq_length", 512)),
        prog_bar=True,
        rank_zero_only=True,
        batch_size=1,
    )
    ###################################################################
    if self.rampup_batch_size:
        self.prev_global_batch_size = current_global_batch_size
        self.prev_consumed_samples = consumed_samples
        num_microbatch_calculator.update(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size
        self.log("global_batch_size", current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.if_first_step = 1

    return loss_mean


def monkeypatch_training_step():
    # NOTE(tj.solergibert) Monkeypatch to log consumed_tokens.
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
        MegatronGPTModel,
    )

    MegatronGPTModel.training_step = _training_step


import math
from typing import List

from nemo.collections.common.parts.utils import extend_instance
from nemo.utils import logging


class _EmbeddingScalingMixin(torch.nn.Module):
    """
    A mixin class for scaling embeddings in Megatron GPT.
    The scaling is applied only if the configuration (accessible via `self.config`)
    includes `apply_embedding_scaling` set to True.
    """

    def forward(self, **kwargs):
        """
        Forward pass that scales the output embeddings from the `forward` method of
        the superclass by the square root of the hidden size specified in the configuration.
        """
        embeddings = super().forward(**kwargs)
        return embeddings * torch.tensor(self.config.hidden_size**0.5, dtype=embeddings.dtype)


def _drop_layers(model, layers_to_drop: List[int]):
    def noop_forward_patch(
        hidden_states,
        attention_mask,
        context_mask=None,
        context=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        return hidden_states.clone(), context

    num_layers = len(model.decoder.layers)
    for layer_id in layers_to_drop:
        assert layer_id > 0 and layer_id <= num_layers, f"Layers to drop should be in range (1, {num_layers})"
        logging.info(f"Patching layer {layer_id} to noop-layer in forward pass")
        model.decoder.layers[layer_id - 1].forward = noop_forward_patch


def _apply_rope_scaling(
    freqs,
    scale_factor: int = 8,
    low_freq_factor: int = 1,
    high_freq_factor: int = 4,
    old_context_len: int = 8192,
):
    # Apply scaling for RoPE frequencies
    logging.info(
        f"Apply rope scaling with scale_factor={scale_factor}, low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, old_context_len={old_context_len}."
    )

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def _mcore_model_customize(cfg, model):
    if cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
        extend_instance(model.embedding, _EmbeddingScalingMixin)
    if cfg.get("scale_positional_embedding", False):
        model.rotary_pos_emb.inv_freq = _apply_rope_scaling(
            model.rotary_pos_emb.inv_freq,
            scale_factor=cfg.get("scale_factor", 8),
            low_freq_factor=cfg.get("low_freq_factor", 1),
            high_freq_factor=cfg.get("high_freq_factor", 4),
            old_context_len=cfg.get("old_context_len", 8192),
        )
    if cfg.get("mcore_customization_config", {}).get("final_logit_softcapping", 0):
        from nemo.collections.nlp.models.language_modeling.megatron.gemma2.gemma2_modules import (
            Gemma2OutputLayer,
        )

        extend_instance(model.output_layer, Gemma2OutputLayer)
    if cfg.get("drop_layers"):
        assert cfg.get("skip_train", False), "Dropping layers allowed only for validation runs (forward pass)"
        _drop_layers(model, cfg.get("drop_layers"))


def monkeypatch_llama_ropes():
    # NOTE(tj.solergibert) Added in https://github.com/NVIDIA/NeMo/pull/11807
    from nemo.collections.nlp.models.language_modeling import megatron_gpt_model

    megatron_gpt_model.mcore_model_customize = _mcore_model_customize


def _setup_validation_data(self, cfg):
    if hasattr(self, "_validation_ds"):
        if self._validation_ds is not None and len(self._validation_ds) > 0:
            consumed_samples = 0
            logging.info(
                f"Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}"
            )

            drop_last = True
            if not self.validation_drop_last:
                logging.info("Drop last in validation dataset is set to False")
                drop_last = False
            pad_samples_to_global_batch_size = False
            if self.cfg.data.get("pad_samples_to_global_batch_size", False):
                logging.info("pad_samples_to_global_batch_size set to True")
                pad_samples_to_global_batch_size = True

            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, "validation", drop_last, pad_samples_to_global_batch_size
            )
        else:
            self._validation_dl = None


def monkeypatch_validation_test_dataloaders():
    # NOTE(tj.solergibert) Remove validation dataloader
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
        MegatronGPTModel,
    )

    MegatronGPTModel.setup_validation_data = _setup_validation_data
