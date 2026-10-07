"""
STARK-ST tracker (inference).

Differences from the original STARK code (with default parameters the behaviour is identical):
  - The template update interval, confidence threshold and maximum number of updates come from
    `params` (originally they were read from the YAML based on the dataset name).
  - track() also returns a `template_updated` flag, so subclasses (fine-tuning) do not have to
    re-evaluate the update condition themselves.
  - Frame indices of the performed template updates are kept in `self.update_frames`.
  - An alternative update rule (params.update_mode = 'max', see _update_max).
"""
from copy import deepcopy

import torch

from lib.models.stark import build_starkst
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import box_iou, box_xywh_to_xyxy, clip_box
from lib.utils.checkpoint import load_network_weights
from lib.utils.merge import merge_template_search
from lib.utils.processing_utils import sample_target

UPDATE_MODES = ('stark', 'max')


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
        self.update_mode = getattr(params, 'update_mode', 'stark')
        if self.update_mode not in UPDATE_MODES:
            raise ValueError(f"invalid update_mode={self.update_mode!r}, must be one of {UPDATE_MODES}")
        self.update_intervals = list(params.update_intervals)
        self.update_conf_thr = params.update_conf_thr
        self.update_iou_thr = getattr(params, 'update_iou_thr', 0.5)  # 'max' only
        self.max_template_updates = params.max_template_updates  # -1 = unlimited
        self.num_extra_template = len(self.update_intervals)

        self.z_dict1 = {}
        self.z_dict_list = []
        self.template_update_count = 0
        self.update_frames = []
        self.interval_best = {}       # 'max': best candidate of the current interval, per update interval
        self.template_source = None   # (frame, conf, image, box) the last updated template was taken from

    def initialize(self, image, info: dict):
        self.z_dict_list = []
        self.z_dict1 = self._template(image, info['init_bbox'])
        # The first element is the (static) template of the first frame, the others are dynamic templates
        self.z_dict_list.append(self.z_dict1)
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        self.state = info['init_bbox']
        self.frame_id = 0
        self.template_update_count = 0
        self.update_frames = []
        self.interval_best = {}
        self.template_source = None

    def _template(self, image, box):
        z_patch_arr, _, z_amask_arr = sample_target(image, box, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            return self.network.forward_backbone(template)

    def _update_allowed(self):
        return self.max_template_updates < 0 or self.template_update_count < self.max_template_updates

    def _update_stark(self, image, conf_score):
        """Original STARK: every update_interval-th frame becomes the template if conf > update_conf_thr."""
        updated = False
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > self.update_conf_thr and self._update_allowed():
                self.z_dict_list[idx + 1] = self._template(image, self.state)
                self.template_update_count += 1
                self.template_source = (self.frame_id, conf_score, image, list(self.state))
                updated = True
        return updated

    def _update_max(self, image, conf_score, prev_state):
        """'max': the frames are split into intervals of update_interval frames; at the end of an interval the
        candidate with the highest confidence becomes the template. Candidate: conf > update_conf_thr and
        IoU(box, box of the previous candidate of the interval) >= update_iou_thr; at the start of an interval the
        reference is the box of the frame before. The first interval (frames 1..N) is skipped, so the first update
        is at frame 2N. Without a candidate the template is not changed."""
        updated = False
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id <= update_i:
                continue
            if (self.frame_id - 1) % update_i == 0:  # first frame of an interval
                self.interval_best[idx] = {'ref_box': list(prev_state), 'conf': -1.0, 'z_dict': None}
            best = self.interval_best[idx]
            iou, _ = box_iou(box_xywh_to_xyxy(torch.tensor([best['ref_box']], dtype=torch.float32)),
                             box_xywh_to_xyxy(torch.tensor([self.state], dtype=torch.float32)))
            if conf_score > self.update_conf_thr and iou.item() >= self.update_iou_thr:
                best['ref_box'] = list(self.state)
                if conf_score > best['conf']:
                    best.update(conf=conf_score, z_dict=self._template(image, self.state),
                                source=(self.frame_id, conf_score, image, list(self.state)))
            if self.frame_id % update_i == 0 and best['z_dict'] is not None and self._update_allowed():
                self.z_dict_list[idx + 1] = best['z_dict']
                self.template_update_count += 1
                self.template_source = best['source']
                updated = True
        return updated

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
        prev_state = self.state
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

        if self.update_mode == 'max':
            template_updated = self._update_max(image, conf_score, prev_state)
        else:
            template_updated = self._update_stark(image, conf_score)
        out = {"target_bbox": self.state,
               "conf_score": conf_score,
               "template_updated": template_updated}
        if template_updated:
            self.update_frames.append(self.frame_id)
            out["template_frame"], out["template_conf"] = self.template_source[:2]
        return out

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
