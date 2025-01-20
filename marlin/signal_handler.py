import signal
from types import FrameType
from typing import Any, Callable, Union

import lightning.pytorch as pl
from lightning.pytorch.trainer.connectors.signal_connector import (
    _HandlersCompose,
    _SignalConnector,
)
from nemo.utils import logging

_SIGNAL_TRIGGER = signal.SIGUSR2

# copied from signal.pyi
_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]


class MarlinSignalHandler(_SignalConnector):
    # _SignalConnector: https://github.com/Lightning-AI/pytorch-lightning/blob/c45c3c92c059661c20443a0a19399f7442db7f90/src/lightning/pytorch/trainer/connectors/signal_connector.py#L36
    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        self.trainer.received_auto_save_signal = False

    def register_signal_handlers(self) -> None:
        # This method is called in trainer.fit() https://github.com/Lightning-AI/pytorch-lightning/blob/c45c3c92c059661c20443a0a19399f7442db7f90/src/lightning/pytorch/trainer/trainer.py#L977
        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()

        sigusr_handlers: list[_HANDLER] = []
        sigterm_handlers: list[_HANDLER] = [self._sigterm_notifier_fn]

        logging.info(f"[{self.__class__.__name__}] Marlin Auto-checkpoint saver enabled. Setting signal handlers.")
        sigusr_handlers.append(self._slurm_sigusr_handler_fn)
        sigterm_handlers.append(self._sigterm_handler_fn)

        sigusr = _SIGNAL_TRIGGER
        assert sigusr is not None, f"No signal specified, remove {self.__class__.__name__} _signal_connector"
        if sigusr_handlers and not self._has_already_handler(sigusr):
            self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))

        # We have our own handler, but include existing ones too
        if self._has_already_handler(signal.SIGTERM):
            sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
        self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        logging.info(f"[{self.__class__.__name__}] Handling Auto-checkpoint saver signal: {signum}")
        self.trainer.received_auto_save_signal = True
