# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) wilson.xu. All rights reserved.
import os
import math
from argparse import ArgumentParser

import cv2
from PIL import Image
import matplotlib
import numpy as np

import mmcv
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

from mmpose.openpose import Hand, openposeHandDetect, openposeDrawHandPose

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class RTMPoseInfer(object):
    def __init__(self,
                 det_cfg,
                 det_pth,
                 bodypose_cfg,
                 bodypose_pth,
                 handpose_cfg,
                 handpose_pth,
                 facepose_cfg,
                 facepose_pth,
                 kpt_thr=0.4,
                 bbox_thr=0.4,
                 nms_thr=0.4,
                 det_cat_id=0,
                 draw_bbox=False,
                 device='cuda'):
        self.kpt_thr = kpt_thr
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.det_cat_id = det_cat_id
        self.draw_bbox = draw_bbox
        # build detector
        self.detector = init_detector(det_cfg, det_pth, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        # build body pose estimator
        self.bodypose_estimator = init_pose_estimator(
            bodypose_cfg,
            bodypose_pth,
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
        # build hand pose estimator
        if handpose_cfg is not None:
            self.handpose_estimator = init_pose_estimator(
                handpose_cfg,
                handpose_pth,
                device=device,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
        else:
            self.handpose_estimator = Hand(handpose_pth, device=device)
        # build face pose estimator
        self.facepose_estimator = init_pose_estimator(
            facepose_cfg,
            facepose_pth,
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    # Reference https://github.com/lllyasviel/ControlNet/blob/main/annotator/openpose/util.py#L94
    # Modified by Wilson.xu
    @staticmethod
    def hand_detect(kpts,
                    kpt_valid,
                    img_shape,
                    min_hand_bbox=20,
                    ratio_wrist_elbow=0.333,
                    ratio_distance=1.1):
        """
            right: shoulder 2, elbow 3, wrist 4
            left: shoulder 5, elbow 6, wrist 7
        Args:
            kpts (ndarray: float): shape[n, 18, 2], (x, y)
            kpt_valid (ndarray: int): shape[n, 18], keypoint score > threshold, (1 or 0)
            img_shape (tuple|list): (h, w)
            min_hand_bbox (int): Default: 20
            ratio_wrist_elbow (float): Default: 0.33
            ratio_distance (float): Default: 1.3

        Returns (ndarray: float): shape[n, 4], (x1, y1, x2, y2)
        """

        def comput_hand_bbox(wrist, elbow, h, w, shoulder=None):
            center = wrist + ratio_wrist_elbow * (wrist - elbow)
            distance_wrist_elbow = np.sqrt(np.square(wrist - elbow).sum())
            length = ratio_distance * distance_wrist_elbow
            if shoulder is not None:
                distance_elbow_shoulder = 0.9 * np.sqrt(
                    np.square(elbow - shoulder).sum())
                length = max(length, ratio_distance * distance_elbow_shoulder)
            x1, y1 = center - length / 2
            x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
            x2, y2 = center + length / 2
            x2, y2 = np.clip(x2, 0, w - 1), np.clip(y2, 0, h - 1)
            return [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ] if x2 - x1 >= min_hand_bbox and y2 - y1 >= min_hand_bbox else None

        h, w = img_shape
        detect_result = []
        for body, is_valid in zip(kpts, kpt_valid):
            # left hand
            if np.sum(is_valid[[6, 7]]) == 2:
                left_shoulder = body[5] if is_valid[5] == 1 else None
                out = comput_hand_bbox(body[7], body[6], h, w, left_shoulder)
                if out:
                    detect_result.append(out)
            # right hand
            if np.sum(is_valid[[3, 4]]) == 2:
                right_shoulder = body[2] if is_valid[2] == 1 else None
                out = comput_hand_bbox(body[4], body[3], h, w, right_shoulder)
                if out:
                    detect_result.append(out)
        return np.asarray(detect_result)

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
    def mmpose2openpose(body_kpts, body_kpt_scores, kpt_thr=0.4):
        # transfer mmpose to openpose
        keypoints_info = np.concatenate(
            (body_kpts, body_kpt_scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:,
        2] = np.logical_and(keypoints_info[:, 5, 2] > kpt_thr,
                            keypoints_info[:, 6, 2] > kpt_thr).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        return keypoints_info[..., :2], keypoints_info[..., 2]

    @staticmethod
    def draw_bodypose(canvas, kpts, kpt_valid, stickwidth=4, radius=4):
        links = [[1, 2], [1, 5],
                 [2, 3], [3, 4],
                 [5, 6], [6, 7],
                 [1, 8], [8, 9], [9, 10],
                 [1, 11], [11, 12], [12, 13],
                 [1, 0], [0, 14], [14, 16],
                 [0, 15], [15, 17]]

        colors = [[255, 0, 0], [255, 85, 0],
                  [255, 170, 0], [255, 255, 0], [170, 255, 0],
                  [85, 255, 0], [0, 255, 0], [0, 255, 85],
                  [0, 255, 170], [0, 255, 255], [0, 170, 255],
                  [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        # draw links
        for link, color in zip(links, colors):
            for kpt, is_valid in zip(kpts, kpt_valid):
                if np.sum(is_valid[link]) == 2:
                    kpt_XY = kpt[link]
                    mean_XY = np.mean(kpt_XY, axis=0)
                    diff_XY = kpt_XY[0] - kpt_XY[1]
                    length = np.sqrt(np.square(diff_XY).sum())
                    angle = math.degrees(math.atan2(diff_XY[1], diff_XY[0]))
                    polygon = cv2.ellipse2Poly((int(mean_XY[0]), int(mean_XY[1])),
                                               (int(length / 2), stickwidth),
                                               int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(canvas, polygon, color)

        # canvas = (canvas * 0.6).astype(np.uint8)
        # draw points
        for i, color in enumerate(colors):
            for kpt, is_valid in zip(kpts, kpt_valid):
                if is_valid[i]:
                    cv2.circle(
                        canvas, (int(kpt[i, 0]), int(kpt[i, 1])),
                        radius, color, thickness=-1)
        return canvas

    @staticmethod
    def draw_handpose(canvas, kpts, kpt_valid, radius=4, show_number=False):
        links = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]
        H, W, _ = canvas.shape
        kpts[..., 0] = np.clip(kpts[..., 0], 0, W - 1)
        kpts[..., 1] = np.clip(kpts[..., 1], 0, H - 1)
        for kpt, is_valid in zip(kpts, kpt_valid):
            kpt = kpt.astype(np.int32)
            # draw links
            for i, link in enumerate(links):
                if np.sum(is_valid[link]) == 2:
                    x1, y1 = kpt[link[0]]
                    x2, y2 = kpt[link[1]]
                    cv2.line(
                        canvas, (x1, y1), (x2, y2),
                        matplotlib.colors.hsv_to_rgb(
                            [i / len(links), 1.0, 1.0]) * 255,
                        thickness=2)
            # draw points
            for i, (x, y) in enumerate(kpt):
                if is_valid[i]:
                    cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
                    if show_number:
                        cv2.putText(
                            canvas,
                            str(i), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 0, 0),
                            lineType=cv2.LINE_AA)
        return canvas

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

    def pred_body_pose(self, img, canvas=None):
        # predict human bboxes
        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                       pred_instance.scores > self.bbox_thr)]
        bboxes = bboxes[nms(bboxes, self.nms_thr), :4]
        if self.draw_bbox and canvas is not None:
            canvas = mmcv.imshow_bboxes(canvas, bboxes, 'green', show=False)
        # predict body keypoints
        pose_results = inference_topdown(self.bodypose_estimator, img, bboxes)
        return merge_data_samples(pose_results).get('pred_instances', None), canvas

    def pred_and_draw_hand_pose(self, img, canvas, body_kpts, body_kpt_scores, mode="mmpose"):
        assert mode in ["mmpose", "openpose"]
        H, W, _ = img.shape
        if mode == "mmpose":
            # predict hand bboxes
            hand_bboxes = self.hand_detect(body_kpts, body_kpt_scores > self.kpt_thr, (H, W))
            if self.draw_bbox and canvas is not None:
                canvas = mmcv.imshow_bboxes(canvas, hand_bboxes, 'green', show=False)
            # predict hand keypoints
            pose_results = inference_topdown(self.handpose_estimator, img,
                                             hand_bboxes)
            pred_hand = merge_data_samples(pose_results).get('pred_instances', None)

            if pred_hand:
                # Filter outliers
                hand_bboxes = pred_hand.bboxes
                hand_kpts = pred_hand.keypoints
                hand_kpt_valid = pred_hand.keypoint_scores > self.kpt_thr
                in_bbox_x = np.logical_and((hand_kpts[..., 0] - hand_bboxes[:, 0:1]) > 0,
                                           (hand_bboxes[:, 2:3] - hand_kpts[..., 0]) > 0)
                in_bbox_y = np.logical_and((hand_kpts[..., 1] - hand_bboxes[:, 1:2]) > 0,
                                           (hand_bboxes[:, 3:4] - hand_kpts[..., 1]) > 0)
                hand_kpt_valid = np.logical_and(hand_kpt_valid,
                                                np.logical_and(in_bbox_x, in_bbox_y))
                # draw hand pose
                canvas = self.draw_handpose(canvas, hand_kpts, hand_kpt_valid)
            return canvas
        elif mode == "openpose":
            # transfer results to openpose format
            candidate, subset = [], []
            for kpts, kpt_scores in zip(body_kpts, body_kpt_scores):
                idx = 0
                total_scroe = 0
                single_set = []
                for kpt, score in zip(kpts, kpt_scores):
                    if score > self.kpt_thr:
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
                hands_list = openposeHandDetect(candidate, subset, img)
                if hands_list:
                    if self.draw_bbox and canvas is not None:
                        hand_bboxes = [[a[0], a[1], a[0] + a[2], a[1] + a[2]] for a in hands_list]
                        hand_bboxes = np.asarray(hand_bboxes)
                        canvas = mmcv.imshow_bboxes(canvas, hand_bboxes, 'green', show=False)
                    # predict hand keypoints
                    hands = []
                    for x, y, w, is_left in hands_list:
                        peaks = self.handpose_estimator(img[y:y + w, x:x + w, ::-1]).astype(np.float32)
                        if peaks.ndim == 2 and peaks.shape[1] == 2:
                            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(img.shape[1])
                            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(img.shape[0])
                            hands.append(peaks.tolist())
                    canvas = openposeDrawHandPose(canvas, hands)
            return canvas
        else:
            raise Exception(f"Worng mode: {mode}! The correct "
                            "mode is in [`mmpose`, `openpose`]")

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

    def __call__(self, img, canvas=None, mode="openpose", **kwargs):
        """
        Args:
            img (ndarray):
            canvas (ndarray|None):
            mode (str): Default: `mmpose`
        Returns: canvas
        """
        H, W, C = img.shape
        assert C == 3
        assert mode in ["mmpose", "openpose"]
        if canvas is None:
            canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        # 1. predict body pose
        pred_body, canvas = self.pred_body_pose(img, canvas)
        if pred_body:
            # transfer `mmpose` to `openpose` format
            body_kpts, body_kpt_scores = self.mmpose2openpose(
                pred_body.keypoints, pred_body.keypoint_scores, self.kpt_thr)
            body_kpt_valid = body_kpt_scores > self.kpt_thr
            body_kpts[..., 0] = np.clip(body_kpts[..., 0], 0, W - 1)
            body_kpts[..., 1] = np.clip(body_kpts[..., 1], 0, H - 1)
            # draw body pose
            canvas = self.draw_bodypose(canvas, body_kpts, body_kpt_valid)

            # 2. predict hand pose
            canvas = self.pred_and_draw_hand_pose(img, canvas, body_kpts, body_kpt_scores, mode)

            # 3. predict face pose
            pred_face, canvas = self.pred_face_pose(img, body_kpts, body_kpt_valid, canvas)
            # draw face pose
            if pred_face:
                canvas = self.draw_facepose(
                    canvas, pred_face.keypoints,
                    pred_face.keypoint_scores > self.kpt_thr)

        return canvas


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('body_config', help='Config file for pose')
    parser.add_argument('body_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('hand_config', help='Config file for pose')
    parser.add_argument('hand_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('face_config', help='Config file for pose')
    parser.add_argument('face_checkpoint', help='Checkpoint file for pose')

    parser.add_argument(
        '--img_file', type=str, default=None, help='Image file')
    parser.add_argument(
        '--img_dir', type=str, default=None, help='Image dir')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='output',
        help='root of the output img file. '
             'Default not saving the visualization images.')

    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.4,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.4,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw_bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--mode',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='hand pose mode')
    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    inferencer = RTMPoseInfer(
        args.det_config, args.det_checkpoint,
        args.body_config, args.body_checkpoint,
        args.hand_config if args.hand_config != "openpose" else None, args.hand_checkpoint,
        args.face_config, args.face_checkpoint,
        kpt_thr=args.kpt_thr,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        det_cat_id=args.det_cat_id,
        draw_bbox=args.draw_bbox,
        device=args.device)

    mkdir_or_exist(args.out_dir)
    if args.img_dir:
        for line in os.listdir(args.img_dir):
            if os.path.splitext(line)[-1].lower() in ['.jpg', '.png']:
                # inference
                img_file = os.path.join(args.img_dir, line)
                img = np.asarray(Image.open(img_file).convert("RGB"))
                canvas = inferencer(img.copy(), img.copy(), mode=args.mode)
                Image.fromarray(canvas).save(os.path.join(args.out_dir, os.path.basename(img_file)))
                print(f"{os.path.basename(img_file)} has been saved")
    elif args.img_file:
        img = np.asarray(Image.open(args.img_file).convert("RGB"))
        canvas = inferencer(img.copy(), img.copy(), mode=args.mode)
        Image.fromarray(canvas).save(os.path.join(args.out_dir, os.path.basename(args.img_file)))
        print(f"{args.img_file} has been saved")
    else:
        raise Exception('Please enter the correct `img_file` or `img_dir`!')


if __name__ == '__main__':
    main()
    print("Done!")
