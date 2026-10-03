"""
STARK-ST tracker (inference).

Differences from the original STARK code (with default parameters the behaviour is identical):
  - The template update interval, confidence threshold and maximum number of updates come from
    `params` (originally they were read from the YAML based on the dataset name).
  - track() also returns a `template_updated` flag, so subclasses (fine-tuning) do not have to
    re-evaluate the update condition themselves.
  - Frame indices of the performed template updates are kept in `self.update_frames`.
"""
from copy import deepcopy

import torch

from lib.models.stark import build_starkst
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.checkpoint import load_network_weights
from lib.utils.merge import merge_template_search
from lib.utils.processing_utils import sample_target


class STARK_ST(BaseTracker):
    def __init__(self, params):
        super().__init__(params)
        network = build_starkst(params.cfg)
        network.load_state_dict(load_network_weights(params.checkpoint), strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.frame_id = 0

        # Template update settings
        self.update_intervals = list(params.update_intervals)
        self.update_conf_thr = params.update_conf_thr
        self.max_template_updates = params.max_template_updates  # -1 = unlimited
        self.num_extra_template = len(self.update_intervals)

        self.z_dict1 = {}
        self.z_dict_list = []
        self.template_update_count = 0
        self.update_frames = []

    def initialize(self, image, info: dict):
        self.z_dict_list = []
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # The first element is the (static) template of the first frame, the others are dynamic templates
        self.z_dict_list.append(self.z_dict1)
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        self.state = info['init_bbox']
        self.frame_id = 0
        self.template_update_count = 0
        self.update_frames = []

    def _update_allowed(self):
        return self.max_template_updates < 0 or self.template_update_count < self.max_template_updates

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Mean of the boxes of all queries (original STARK)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h)
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

        # Template update: interval reached AND confidence above threshold AND limit not exceeded
        template_updated = False
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > self.update_conf_thr and self._update_allowed():
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[idx + 1] = z_dict_t
                self.template_update_count += 1
                template_updated = True
        if template_updated:
            self.update_frames.append(self.frame_id)

        return {"target_bbox": self.state,
                "conf_score": conf_score,
                "template_updated": template_updated}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
