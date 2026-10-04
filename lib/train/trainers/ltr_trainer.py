"""
LTR trainer: one training (and, every `epoch_interval` epochs, one validation) pass per epoch.

Changes with respect to the original STARK LTRTrainer:
  - Gradient accumulation: the loader yields micro-batches of `settings.batchsize` samples; the gradients of
    `settings.accum_steps` micro-batches are accumulated (each loss scaled by 1/accum_steps) before one optimizer step.
    The gradient is therefore the mean over `batchsize * accum_steps` samples, exactly like the original multi-GPU
    training (8 GPUs x 16 samples, gradients averaged by DistributedDataParallel), and gradient clipping is applied
    to that accumulated gradient. STARK freezes all BatchNorm layers of the backbone, so splitting a batch into
    micro-batches does not change the forward pass either.
  - Per epoch the number of optimizer steps equals the original one (samples_per_epoch // effective batch);
    leftover micro-batches that would not complete a step are not processed.
  - Progress output with samples/s and an ETA; per-epoch statistics are appended to `history.csv`.
"""
import csv
import os
import time
from collections import OrderedDict

import torch

from lib.train.admin import AverageMeter, StatValue
from lib.train.trainers import BaseTrainer

try:
    from lib.train.admin import TensorboardWriter
except Exception:  # pragma: no cover - tensorboard is optional
    TensorboardWriter = None


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})
        self.accum_steps = int(getattr(settings, "accum_steps", 1))
        self.tensorboard_writer = None
        if TensorboardWriter is not None and getattr(settings, "tensorboard_dir", None):
            try:
                self.tensorboard_writer = TensorboardWriter(settings.tensorboard_dir, [l.name for l in loaders])
            except Exception as e:  # noqa: BLE001
                print(f"TensorBoard disabled: {e}")

    # ------------------------------------------------------------------ one pass over a loader
    def cycle_dataset(self, loader):
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        micro_batches = len(loader)
        if loader.training:
            steps = micro_batches // self.accum_steps
            micro_batches = steps * self.accum_steps
        else:
            steps = micro_batches
        if micro_batches == 0:
            raise RuntimeError(f"'{loader.name}': samples per epoch is smaller than one optimizer step")

        start = time.time()
        samples = 0
        step = 0
        self.optimizer.zero_grad()
        for i, data in enumerate(loader, 1):
            if i > micro_batches:
                break
            data = data.to(self.device)
            data["epoch"] = self.epoch
            data["settings"] = self.settings
            loss, stats = self.actor(data)

            if loader.training:
                (loss / self.accum_steps).backward()
                if i % self.accum_steps == 0:
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
            else:
                step = i

            batch_size = data["template_images"].shape[loader.stack_dim]
            samples += batch_size
            self._update_stats(stats, batch_size, loader)
            boundary = (not loader.training) or i % self.accum_steps == 0
            if boundary and (step == 1 or step % self.settings.print_interval == 0 or step == steps):
                self._print_stats(loader, step, steps, samples, start)

    def _print_stats(self, loader, step, steps, samples, start):
        elapsed = time.time() - start
        rate = samples / max(elapsed, 1e-9)
        epoch_time = elapsed / max(step, 1) * steps
        remaining = epoch_time - elapsed
        msg = (f"[{loader.name}: epoch {self.epoch}/{self.settings.max_epochs}, step {step}/{steps}] "
               f"{rate:.1f} samples/s, epoch ETA {remaining / 60:.1f} min")
        if loader.training:
            total = remaining + (self.settings.max_epochs - self.epoch) * epoch_time
            msg += f", training ETA {total / 3600:.1f} h"
        for name, val in self.stats[loader.name].items():
            if hasattr(val, "avg"):
                msg += f" | {name}: {val.avg:.5f}"
        print(msg, flush=True)
        with open(self.settings.log_file, "a") as f:
            f.write(msg + "\n")

    def train_epoch(self):
        epoch_start = time.time()
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)
        self._write_history(time.time() - epoch_start)
        self._stats_new_epoch()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    # ------------------------------------------------------------------ statistics
    def _update_stats(self, new_stats, batch_size, loader):
        if self.stats.get(loader.name) is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})
        for name, val in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _write_history(self, seconds):
        row = {"epoch": self.epoch, "seconds": round(seconds, 1),
               "lr": self.optimizer.param_groups[0]["lr"]}
        for loader in self.loaders:
            for name, val in (self.stats.get(loader.name) or {}).items():
                if hasattr(val, "avg") and val.count > 0:
                    row[f"{loader.name}/{name}"] = val.avg
        path = self.settings.history_file
        rows = []
        if os.path.isfile(path):
            with open(path, newline="") as f:
                # rows of epochs that are repeated after resuming from an earlier checkpoint are replaced
                rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < self.epoch]
        rows.append(row)
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]  # e.g. the first validation epoch adds columns
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training and self.lr_scheduler is not None:
                for i, lr in enumerate(self.lr_scheduler.get_last_lr()):
                    var = f"LearningRate/group{i}"
                    self.stats[loader.name].setdefault(var, StatValue())
                    self.stats[loader.name][var].update(lr)
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, "new_epoch"):
                    stat_value.new_epoch()
