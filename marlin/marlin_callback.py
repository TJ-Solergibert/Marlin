import os
import subprocess

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizerCallback
from nemo.utils.get_rank import is_global_rank_zero

############################ MarlinCallback ############################
# 1. Received SIGUSR2 --> Store checkpoint + exit gracefully
# 2. On save checkpoint --> Schedule security copy + Schedule evaluation pipeline
# 3. Check save checkpoint trigger
# 4. Check exit trigger. If found we also cancel the next scheduled jobs (With the slurm job name variable)


class MarlinCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logging_dir = cfg.exp_manager.explicit_log_dir
        self.save_trigger = os.path.join(self.logging_dir, "triggers", "save")
        self.exit_trigger = os.path.join(self.logging_dir, "triggers", "exit")

        self.trigger_save_checkpoint = False
        self.trigger_exit = False

        logging.info(
            f"[{self.__class__.__name__}] Save trigger setup! run `touch {self.save_trigger}` to save a checkpoint"
        )
        logging.info(
            f"[{self.__class__.__name__}] Exit trigger setup! run `touch  {self.exit_trigger}` to stop training"
        )

    def on_train_batch_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        if trainer._logger_connector.should_update_logs:
            if trainer.received_auto_save_signal:
                self.trigger_save_checkpoint = True
                self.trigger_exit = True

            if os.path.isfile(self.save_trigger):
                logging.info(f"[{self.__class__.__name__}] Save trigger detected!")
                self.trigger_save_checkpoint = True

            if os.path.isfile(self.exit_trigger):
                logging.info(f"[{self.__class__.__name__}] Exit trigger detected!")
                self.trigger_exit = True

            if self.trigger_save_checkpoint:
                self.custom_save_checkpoint(trainer)
                self.trigger_save_checkpoint = False
                # If save trigger file detected --> Reset trigger. `received_auto_save_signal=True` triggers a save BUT doesn't creates any save_trigger file
                if not trainer.received_auto_save_signal and is_global_rank_zero():
                    os.remove(self.save_trigger)

            if self.trigger_exit:
                trainer.should_stop = True
                # If exit trigger file detected --> Reset trigger & Cancel pending jobs in the queue
                self.trigger_exit = False  # Bit useless
                if not trainer.received_auto_save_signal and is_global_rank_zero():
                    os.remove(self.exit_trigger)
                    logging.info(f"[{self.__class__.__name__}] Cancelling all pending jobs")
                    subprocess.check_output("scancel --jobname $SLURM_JOB_NAME", shell=True)

    def on_train_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        if trainer.global_step > int(trainer.max_steps * 0.97) and is_global_rank_zero():
            logging.info(f"[{self.__class__.__name__}] Approaching training end, cancelling all pending jobs")
            subprocess.check_output("scancel --jobname $SLURM_JOB_NAME", shell=True)

    def custom_save_checkpoint(self, trainer: "pl.Trainer"):
        logging.info(f"[{self.__class__.__name__}] Triggering checkpoint save!")
        monitor_candidates = trainer.checkpoint_callback._monitor_candidates(trainer)
        trainer.checkpoint_callback._save_topk_checkpoint(trainer, monitor_candidates)

        if self.trigger_exit and (self.cfg.get("exp_manager", {}) or {}).get("checkpoint_callback_params", {}).get(
            "async_save", False
        ):
            logging.info(f"[{self.__class__.__name__}] Finalizing checkpoint before exit...")
            async_finalizer_callback = [c for c in trainer.callbacks if isinstance(c, AsyncFinalizerCallback)].pop()
            async_finalizer_callback.on_train_end(trainer)

    def on_save_checkpoint(self, trainer: "pl.Trainer", *args, **kwargs) -> None:  # "pl.Trainer"
        logging.info(f"[{self.__class__.__name__}] We are in {self.__class__.__name__}.on_save_checkpoint method")
        # TODO(tj.solergibert) Submit eval
        # TODO(tj.solergibert) Submit checkpoint copy
