"""
Base trainer: epoch loop, checkpointing and resuming.

Changes with respect to the original STARK BaseTrainer (behaviour of a single uninterrupted run is unchanged):
  - A complete "latest" checkpoint (network, optimizer, LR scheduler, RNG states, statistics) is written atomically
    after EVERY epoch, so a run interrupted at any point (e.g. a Colab session ending) resumes from the last finished
    epoch. Numbered checkpoints (network weights only, ~1/3 of the size; enough for evaluation) are additionally kept
    every `keep_every` epochs and at the end.
  - The weights used to initialise a run (e.g. stage-1 weights for stage 2) are loaded ONLY when the run starts from
    scratch. The original loaded them after resuming as well, which overwrote the already trained classification head
    of a resumed stage-2 run with the untrained one from the stage-1 checkpoint.
  - Training settings are stored as a plain dictionary (no pickled settings objects).
"""
import os
import random

import numpy as np
import torch

from lib.train.admin import multigpu


class BaseTrainer:
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.settings = settings
        self.epoch = 0
        self.stats = {}
        self.device = getattr(settings, "device", None) or torch.device("cuda:0")
        self.actor.to(self.device)
        self._checkpoint_dir = settings.checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    @property
    def net(self):
        return self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

    def train(self, max_epochs, init_fn=None):
        """Trains until `max_epochs`. Resumes from the latest checkpoint if there is one; otherwise calls
        `init_fn(net)` (if given) to initialise the network, e.g. with stage-1 weights."""
        if self.load_checkpoint():
            print(f"Resumed from epoch {self.epoch}.")
        elif init_fn is not None:
            init_fn(self.net)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            keep = (epoch == max_epochs) or (self.settings.keep_every and epoch % self.settings.keep_every == 0)
            self.save_checkpoint(keep_numbered=keep)
        print("Finished training!")

    def train_epoch(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ checkpoints
    def _state(self):
        return {
            "epoch": self.epoch,
            "net_type": type(self.net).__name__,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "stats": self.stats,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
            "train_config": getattr(self.settings, "train_config", None),
        }

    def _atomic_save(self, state, path):
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def save_checkpoint(self, keep_numbered=False):
        state = self._state()
        self._atomic_save(state, os.path.join(self._checkpoint_dir, "latest.pth.tar"))
        if keep_numbered:
            name = "{}_ep{:04d}.pth.tar".format(state["net_type"], self.epoch)
            weights = {k: state[k] for k in ("epoch", "net_type", "net", "train_config")}
            self._atomic_save(weights, os.path.join(self._checkpoint_dir, name))

    def load_checkpoint(self) -> bool:
        path = os.path.join(self._checkpoint_dir, "latest.pth.tar")
        if not os.path.isfile(path):
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["net_type"] == type(self.net).__name__, "Network is not of the correct type."
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.stats = ckpt.get("stats", self.stats)
        self.epoch = ckpt["epoch"]
        rng = ckpt.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        return True
