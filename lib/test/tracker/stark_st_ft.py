"""
STARK-ST + video-specific fine-tuning of the classification head (cls_head, a 3-layer MLP) at inference time.

Modes (params.ft_mode):
  - 'init'   : fine-tune only on the first frame (with the ground-truth box).
  - 'online' : first frame + every frame a template update takes its template from (with the tracker's own
               predicted box; with update_mode='max' this is the selected frame of the interval).
  (In 'none' mode STARK_ST is used directly instead of this class.)

Sample types (params.ft_samples):
  - 'pos'    : positive sample only (label 1).
  - 'posneg' : positive + a negative sample (label 0) from a region of the same frame that does not contain
               the target. The negative generation is provisional; see ft_sampling.negative_search_box.

Flow (each fine-tuning session):
  1. Determine the positive box (optionally with the ST2 jitter) and, if needed, the negative box.
  2. For each box, crop the search region, pass it through backbone + transformer (no_grad) and take the
     decoder output hs (1, 1, 1, 256).
  3. Train only cls_head for `epochs` steps with BCEWithLogitsLoss, AdamW and gradient clipping.
     (With 1-2 samples, 1 epoch = 1 optimisation step.)
cls_head is reset to the base weights at the start of every video, so nothing leaks between videos.
"""
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from lib.test.tracker.ft_sampling import jitter_box, negative_search_box
from lib.test.tracker.stark_st import STARK_ST
from lib.utils.merge import merge_template_search
from lib.utils.processing_utils import sample_target

FT_LOG_HEADER = "frame,session,epoch,loss,pos_prob,neg_prob,n_pos,n_neg,neg_coverage\n"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STARK_ST_FT(STARK_ST):
    FT_MODES = ('init', 'online')
    FT_SAMPLES = ('pos', 'posneg')

    def __init__(self, params):
        super().__init__(params)
        if params.ft_mode not in self.FT_MODES:
            raise ValueError(f"invalid ft_mode={params.ft_mode!r}, must be one of {self.FT_MODES}")
        if params.ft_samples not in self.FT_SAMPLES:
            raise ValueError(f"invalid ft_samples={params.ft_samples!r}, must be one of {self.FT_SAMPLES}")

        self.ft_mode = params.ft_mode
        self.ft_samples = params.ft_samples
        self.ft_lr = params.ft_lr
        self.ft_weight_decay = params.ft_weight_decay
        self.ft_grad_clip_norm = params.ft_grad_clip_norm
        self.ft_epochs_init = params.ft_epochs_init
        self.ft_epochs_online = params.ft_epochs_online
        self.ft_pos_jitter = params.ft_pos_jitter
        self.ft_center_jitter = params.ft_center_jitter
        self.ft_scale_jitter = params.ft_scale_jitter
        self.max_ft_updates = params.max_ft_updates  # -1 = unlimited
        self.seed = params.seed
        self.log_dir = params.log_dir  # None = no loss log is written

        self.base_cls_weights = deepcopy(self.network.cls_head.state_dict())
        self.criterion = nn.BCEWithLogitsLoss()
        self.rng = np.random.RandomState(self.seed)
        self.ft_update_count = 0
        self.ft_frames = []

    # ------------------------------------------------------------------ helpers
    def _log_path(self):
        return None if self.log_dir is None else os.path.join(self.log_dir, "finetune_loss.txt")

    def _get_hs(self, image, bbox):
        """Passes the search region centred on bbox through the pipeline and returns the decoder output (hs)."""
        x_patch_arr, _, x_amask_arr = sample_target(image, bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            seq_dict = merge_template_search(self.z_dict_list + [x_dict])
            _, _, output_embed = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=False,
                                                                  run_cls_head=False)
        return output_embed.detach()  # (1, 1, 1, 256); gradients only flow through cls_head

    def _finetune_on_frame(self, image, bbox, epochs, session, frame):
        img_h, img_w = image.shape[:2]
        if self.ft_pos_jitter:
            pos_box = jitter_box(bbox, self.rng, self.ft_center_jitter, self.ft_scale_jitter)
        else:
            pos_box = list(bbox)
        hs_list, labels = [self._get_hs(image, pos_box)], [1.0]

        neg_coverage = float('nan')
        if self.ft_samples == 'posneg':
            neg_box, neg_coverage = negative_search_box(bbox, img_w, img_h, self.params.search_factor)
            hs_list.append(self._get_hs(image, neg_box))
            labels.append(0.0)

        self._run_finetune(hs_list, labels, epochs, session, neg_coverage, frame)

    def _run_finetune(self, hs_list, labels, epochs, session, neg_coverage, frame):
        hs_batch = torch.cat(hs_list, dim=1)  # (1, N, 1, 256)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=hs_batch.device)
        optimizer = torch.optim.AdamW(self.network.cls_head.parameters(), lr=self.ft_lr,
                                      weight_decay=self.ft_weight_decay)
        n_pos = int((labels_t == 1).sum().item())
        n_neg = int((labels_t == 0).sum().item())
        rows = []

        self.network.cls_head.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            logits = self.network.cls_head(hs_batch)[-1].view(-1)  # (N,)
            loss = self.criterion(logits, labels_t)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pos_prob = probs[labels_t == 1].mean().item() if n_pos else float('nan')
                neg_prob = probs[labels_t == 0].mean().item() if n_neg else float('nan')
            rows.append(f"{frame},{session},{epoch},{loss.item():.8e},{pos_prob:.8f},{neg_prob:.8f},"
                        f"{n_pos},{n_neg},{neg_coverage:.4f}\n")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.cls_head.parameters(), max_norm=self.ft_grad_clip_norm)
            optimizer.step()
        self.network.cls_head.eval()

        path = self._log_path()
        if path is not None:
            with open(path, "a") as f:
                f.writelines(rows)

    # ------------------------------------------------------------------ main methods
    def initialize(self, image, info: dict):
        seed_everything(self.seed)
        self.rng = np.random.RandomState(self.seed)
        super().initialize(image, info)
        self.network.cls_head.load_state_dict(self.base_cls_weights)
        self.ft_update_count = 0
        self.ft_frames = []

        path = self._log_path()
        if path is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(FT_LOG_HEADER)

        if self.ft_epochs_init > 0:
            self._finetune_on_frame(image, info['init_bbox'], self.ft_epochs_init, session='init', frame=0)

    def track(self, image, info: dict = None):
        out = super().track(image, info)
        out['ft_updated'] = False
        if (self.ft_mode == 'online' and out['template_updated'] and self.ft_epochs_online > 0
                and (self.max_ft_updates < 0 or self.ft_update_count < self.max_ft_updates)):
            # The frame the new template was taken from is treated as reliable; the box is the tracker's own prediction.
            src_frame, _, src_image, src_box = self.template_source
            self._finetune_on_frame(src_image, src_box, self.ft_epochs_online, session='online', frame=src_frame)
            self.ft_update_count += 1
            self.ft_frames.append(src_frame)
            out['ft_updated'] = True
        return out
