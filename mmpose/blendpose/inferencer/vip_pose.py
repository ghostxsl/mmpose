# Copyright (c) wilson.xu. All rights reserved.
import cv2
import numpy as np
from PIL import Image

import mmcv
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples

from mmpose.blendpose.template_adapter import TemplatePoseAdapter
from mmpose.blendpose.openpose import Hand, openposeHandDetect
from mmpose.blendpose.inferencer.detector import Detector
from mmpose.blendpose.utils import (mmpose2openpose,
                                    draw_bodypose,
                                    hand_detect, draw_handpose,
                                    get_bbox_intersection)


class VIPPoseInferencer(object):
    def __init__(self,
                 det_cfg,
                 det_pth,
                 bodypose_cfg=None,
                 bodypose_pth=None,
                 handpose_cfg=None,
                 handpose_pth=None,
                 facepose_cfg=None,
                 facepose_pth=None,
                 wholebodypose_cfg=None,
                 wholebodypose_pth=None,
                 template_dir=None,
                 is_hand_intersection=False,
                 body_kpt_thr=0.3,
                 hand_kpt_thr=0.3,
                 face_kpt_thr=0.3,
                 bbox_thr=0.4,
                 nms_thr=0.4,
                 det_cat_id=0,
                 draw_bbox=False,
                 device='cuda'):
        if bodypose_cfg is None and wholebodypose_cfg is None:
            raise Exception(f"`bodypose_cfg` and `wholebodypose_cfg`"
                            f" cannot be both `None`.")
        self.init_body = bodypose_cfg is not None
        self.init_hand = handpose_cfg is not None
        self.init_face = facepose_cfg is not None
        # body: [0:17], foot: [17:23], face: [23:91], hand: [91:133],
        self.init_wholebody = wholebodypose_cfg is not None
        self.handpose_cfg = handpose_cfg
        self.template_dir = template_dir
        self.is_hand_intersection = is_hand_intersection
        self.body_kpt_thr = body_kpt_thr
        self.hand_kpt_thr = hand_kpt_thr
        self.face_kpt_thr = face_kpt_thr
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.det_cat_id = det_cat_id
        self.draw_bbox = draw_bbox

        # build object detector
        self.detector = Detector(det_cfg, det_pth, bbox_thr,
                                 nms_thr, det_cat_id, draw_bbox, device)

        # build whole body pose estimator
        if self.init_wholebody:
            self.wholebodypose_estimator = init_pose_estimator(
                wholebodypose_cfg,
                wholebodypose_pth,
                device=device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # build body pose estimator
        if self.init_body:
            self.bodypose_estimator = init_pose_estimator(
                bodypose_cfg,
                bodypose_pth,
                device=device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # build hand pose estimator
        if self.init_hand:
            if handpose_cfg == "openpose":
                self.handpose_estimator = Hand(handpose_pth, device=device)
            else:
                self.handpose_estimator = init_pose_estimator(
                    handpose_cfg,
                    handpose_pth,
                    device=device,
                    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
        elif template_dir is not None:
            self.template_adapter = TemplatePoseAdapter(template_dir)

        # build face pose estimator
        if self.init_face:
            self.facepose_estimator = init_pose_estimator(
                facepose_cfg,
                facepose_pth,
                device=device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    # Modified by Wilson.xu
    @staticmethod
    def face_detect(kpts,
                    kpt_valid,
                    img_shape,
                    min_face_bbox=20,
                    ratio_ear_distance=1.3):
        """
            nose 0
            right: eye 14, ear 16
            left: eye 15, ear 17
        Args:
            kpts (ndarray: float): shape[n, 18, 2], (x, y)
            kpt_valid (ndarray: int): shape[n, 18], keypoint score > threshold, (1 or 0)
            img_shape (tuple|list): (h, w)
            min_hand_bbox (int): Default: 20


        Returns (ndarray: float): shape[n, 4], (x1, y1, x2, y2)
        """
        h, w = img_shape
        detect_result = []
        for face, is_valid in zip(kpts, kpt_valid):
            if not is_valid[0]:
                continue

            x, y = face[0]
            length = 0.0
            # right eye
            if is_valid[14]:
                x0, y0 = face[14]
                d = max(abs(x - x0), abs(y - y0))
                length = max(length, 2.0 * d * ratio_ear_distance)
            # right ear
            if is_valid[16]:
                x0, y0 = face[16]
                d = max(abs(x - x0), abs(y - y0))
                length = max(length, d * ratio_ear_distance)
            # left eye
            if is_valid[15]:
                x0, y0 = face[15]
                d = max(abs(x - x0), abs(y - y0))
                length = max(length, 2.0 * d * ratio_ear_distance)
            # left ear
            if is_valid[17]:
                x0, y0 = face[17]
                d = max(abs(x - x0), abs(y - y0))
                length = max(length, d * ratio_ear_distance)
            x1, y1 = x - length, y - length
            x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
            x2, y2 = x + length, y + length
            x2, y2 = np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1)

            if x2 - x1 >= min_face_bbox and y2 - y1 >= min_face_bbox:
                detect_result.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)])

        return np.asarray(detect_result)

    @staticmethod
    def draw_facepose(canvas, kpts, kpt_valid):
        H, W, _ = canvas.shape
        kpts[..., 0] = np.clip(kpts[..., 0], 0, W - 1)
        kpts[..., 1] = np.clip(kpts[..., 1], 0, H - 1)
        for face, face_valid in zip(kpts, kpt_valid):
            for kpt, is_valid in zip(face, face_valid):
                if is_valid:
                    cv2.circle(
                        canvas, (int(kpt[0]), int(kpt[1])),
                        3, (255, 255, 255), thickness=-1)
        return canvas

    def pack_results(self, kpts, kpt_scores, bboxes, h, w):
        # normalize keypoints
        return {
            "keypoints": np.concatenate(
                [kpts.astype('float32') / np.array([w - 1, h - 1]),
                 kpt_scores[..., None]], axis=-1),
            "bboxes": bboxes.astype(
                'float32') / np.array([w - 1, h - 1, w - 1, h - 1])
        }

    def pred_body_pose(self, img, bboxes, bodypose_estimator):
        # predict body/wholebody keypoints
        pose_results = inference_topdown(bodypose_estimator, img, bboxes)
        return merge_data_samples(pose_results).get('pred_instances', None)

    def pred_hand_pose(self, img, canvas, body_kpts, body_kpt_scores, body_bboxes):
        H, W, _ = img.shape
        results = None
        if self.handpose_cfg == "openpose":
            # transfer results to openpose format
            candidate, subset = [], []
            for kpts, kpt_scores in zip(body_kpts, body_kpt_scores):
                idx = 0
                total_scroe = 0
                single_set = []
                for kpt, score in zip(kpts, kpt_scores):
                    if score > self.body_kpt_thr:
                        kpt = np.append(kpt, score)
                        kpt = np.append(kpt, idx)
                        candidate.append(kpt)
                        single_set.append(idx)
                        total_scroe += score
                        idx += 1
                    else:
                        single_set.append(-1)
                single_set += [total_scroe, idx]
                subset.append(np.asarray(single_set))
            if candidate:
                candidate = np.stack(candidate, axis=0)
                subset = np.stack(subset, axis=0)
                # predict hand bboxes
                hand_bboxes = openposeHandDetect(candidate, subset, img)
                if hand_bboxes:
                    hand_bboxes = [[a[0], a[1], a[0] + a[2], a[1] + a[2]] for a in hand_bboxes]
                    hand_bboxes = np.asarray(hand_bboxes)
                    if self.draw_bbox and canvas is not None:
                        canvas = mmcv.imshow_bboxes(canvas, hand_bboxes, 'green', show=False)
                    # predict hand keypoints
                    hand_kpts = []
                    hand_kpt_scores = []
                    for x1, y1, x2, y2 in hand_bboxes:
                        peaks = self.handpose_estimator(img[y1:y2, x1:x2, ::-1]).astype(np.float32)
                        scores = np.where(peaks.sum(-1) < 0, 0, 1)
                        hand_kpts.append(peaks + np.array([x1, y1]))
                        hand_kpt_scores.append(scores)
                    results = (np.stack(hand_kpts), np.stack(hand_kpt_scores), hand_bboxes)
        else:
            # predict hand bboxes
            hand_bboxes = hand_detect(body_kpts, body_kpt_scores > self.body_kpt_thr, (H, W))
            if self.is_hand_intersection:
                hand_bboxes = get_bbox_intersection(hand_bboxes, body_bboxes[:, None])
            hand_bboxes = hand_bboxes.reshape([-1, 4])
            if self.draw_bbox and canvas is not None:
                canvas = mmcv.imshow_bboxes(canvas, hand_bboxes, 'green', show=False)
            # predict hand keypoints
            pose_results = inference_topdown(self.handpose_estimator, img,
                                             hand_bboxes)
            pred_hand = merge_data_samples(pose_results).get('pred_instances', None)
            if pred_hand:
                hand_bboxes = pred_hand.bboxes
                hand_kpts = pred_hand.keypoints
                hand_kpt_scores = pred_hand.keypoint_scores
                # Filter outliers
                in_bbox_x = np.logical_and((hand_kpts[..., 0] - hand_bboxes[:, 0:1]) > 0,
                                           (hand_bboxes[:, 2:3] - hand_kpts[..., 0]) > 0)
                in_bbox_y = np.logical_and((hand_kpts[..., 1] - hand_bboxes[:, 1:2]) > 0,
                                           (hand_bboxes[:, 3:4] - hand_kpts[..., 1]) > 0)
                hand_kpt_scores = np.where(
                    np.logical_and(in_bbox_x, in_bbox_y), hand_kpt_scores, 0)
                results = (hand_kpts, hand_kpt_scores, hand_bboxes)
        return results, canvas

    def pred_face_pose(self, img, body_kpts, body_kpt_valid, canvas=None):
        H, W, _ = img.shape
        # predict hand bboxes
        face_bboxes = self.face_detect(body_kpts, body_kpt_valid, (H, W))
        if self.draw_bbox and canvas is not None:
            canvas = mmcv.imshow_bboxes(canvas, face_bboxes, 'green', show=False)
        # predict hand keypoints
        pose_results = inference_topdown(self.facepose_estimator, img,
                                         face_bboxes)
        return merge_data_samples(pose_results).get('pred_instances', None), canvas

    def __call__(self, img, canvas=None, **kwargs):
        """
        Args:
            img (ndarray):
            canvas (ndarray|None):
        Returns: canvas
        """
        if isinstance(img, (Image.Image, np.ndarray)):
            img = np.array(img)
        else:
            raise Exception(f"Unsupported type: {type(img)}.")

        H, W, C = img.shape
        assert C == 3, "The input image can only be in RGB format."
        if canvas is None:
            canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        elif isinstance(canvas, (Image.Image, np.ndarray)):
            canvas = np.array(canvas)
        else:
            raise Exception(f"Unsupported type: {type(img)}.")

        draw_body = kwargs.get('body', True)
        draw_hand = kwargs.get('hand', True)
        draw_face = kwargs.get('face', True)
        return_results = kwargs.get('return_results', True)

        # 0. predict human bboxes
        human_bboxes, canvas = self.detector(img, canvas)
        # 1. predict body pose
        if self.init_wholebody:
            pred_wholebody = self.pred_body_pose(
                img, human_bboxes, self.wholebodypose_estimator)
        if self.init_body:
            pred_body = self.pred_body_pose(
                img, human_bboxes, self.bodypose_estimator)
        elif self.init_wholebody:
            pred_body = pred_wholebody
        else:
            raise Exception(f"`bodypose_cfg` and `wholebodypose_cfg`"
                            f" cannot be both `None`.")

        body_res, hand_res, face_res = None, None, None
        if pred_body:
            # transfer `mmpose` to `openpose` format
            body_kpts, body_kpt_scores = mmpose2openpose(
                pred_body.keypoints[:, :17], pred_body.keypoint_scores[:, :17],
                self.body_kpt_thr)
            body_kpt_valid = body_kpt_scores > self.body_kpt_thr
            body_kpts[..., 0] = np.clip(body_kpts[..., 0], 0, W - 1)
            body_kpts[..., 1] = np.clip(body_kpts[..., 1], 0, H - 1)
            if return_results:
                body_res = self.pack_results(
                    body_kpts, body_kpt_scores, human_bboxes, H, W)
            if draw_body:
                # Draw body pose
                canvas = draw_bodypose(canvas, body_kpts, body_kpt_valid)

            # 2. predict hand pose or use template matching
            if self.init_wholebody:
                if pred_wholebody:
                    hand_kpts = pred_wholebody.keypoints[:, 91:]
                    hand_kpts = hand_kpts.reshape([-1, 21, 2])
                    hand_kpt_scores = pred_wholebody.keypoint_scores[:, 91:]
                    hand_kpt_scores = hand_kpt_scores.reshape([-1, 21])
                    if return_results:
                        hand_bboxes = np.concatenate([
                            hand_kpts.min(1), hand_kpts.max(1)
                        ], axis=-1)
                        hand_res = self.pack_results(
                            hand_kpts, hand_kpt_scores, hand_bboxes, H, W)
                    if draw_hand:
                        canvas = draw_handpose(
                            canvas, hand_kpts, hand_kpt_scores > self.hand_kpt_thr)
            elif self.init_hand:
                pred_hand, canvas = self.pred_hand_pose(
                    img, canvas, body_kpts, body_kpt_scores, human_bboxes)
                # draw hand pose
                if pred_hand:
                    hand_kpts, hand_kpt_scores, hand_bboxes = pred_hand
                    if return_results:
                        hand_res = self.pack_results(
                            hand_kpts, hand_kpt_scores, hand_bboxes, H, W)
                    if draw_hand:
                        canvas = draw_handpose(
                            canvas, hand_kpts, hand_kpt_scores > self.hand_kpt_thr)
            elif self.template_dir is not None:
                hand_kpts, hand_kpt_valid = self.template_adapter(canvas, body_kpts, body_kpt_valid)
                if draw_hand:
                    canvas = draw_handpose(canvas, hand_kpts, hand_kpt_valid)

            # 3. predict face pose
            if self.init_face:
                pred_face, canvas = self.pred_face_pose(img, body_kpts, body_kpt_valid, canvas)
                # Draw face pose
                if pred_face and draw_face:
                    canvas = self.draw_facepose(
                        canvas, pred_face.keypoints,
                        pred_face.keypoint_scores > self.face_kpt_thr)

        if return_results:
            return canvas, {'body': body_res, 'hand': hand_res, 'face': face_res}
        else:
            return canvas
