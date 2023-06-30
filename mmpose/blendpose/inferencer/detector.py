# Copyright (c) wilson.xu. All rights reserved.
import numpy as np
import mmcv
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class Detector(object):
    def __init__(self,
                 config,
                 weight_path,
                 bbox_thr=0.4,
                 nms_thr=0.4,
                 det_cat_id=0,
                 draw_bbox=False,
                 device='cuda'):
        assert has_mmdet, 'Please install mmdet to run the detector.'

        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.det_cat_id = det_cat_id
        self.draw_bbox = draw_bbox
        # build detector
        self.model = init_detector(config, weight_path, device=device)
        self.model.cfg = adapt_mmdet_pipeline(self.model.cfg)

    def __call__(self, img, canvas=None, **kwargs):
        # predict human bboxes
        det_result = inference_detector(self.model, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                       pred_instance.scores > self.bbox_thr)]
        bboxes = bboxes[nms(bboxes, self.nms_thr), :4]
        # draw bboxes
        if self.draw_bbox and canvas is not None:
            canvas = mmcv.imshow_bboxes(canvas, bboxes, 'green', show=False)
        return bboxes, canvas
