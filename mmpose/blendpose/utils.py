# Copyright (c) wilson.xu. All rights reserved.
import math
import cv2
import matplotlib
import numpy as np
import pickle


def pkl_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out


def mmpose2openpose(body_kpts, body_kpt_scores, kpt_thr=0.4):
    """
    Convert `mmpose` format to `openpose` format.
    Args:
        body_kpts (ndarray): shape[n, 17, 2]
        body_kpt_scores (ndarray): shape [n, 17]
        kpt_thr (float): Default 0.4

    Returns: `openpose_kpts`: shape[n, 18, 2],
             `openpose_kpt_scores`: shape[n, 18]
    """
    keypoints_info = np.concatenate(
        (body_kpts, body_kpt_scores[..., None]), axis=-1)
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    # neck score when visualizing pred
    neck[:, 2] = np.logical_and(keypoints_info[:, 5, 2] > kpt_thr,
                                keypoints_info[:, 6, 2] > kpt_thr).astype(int)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    return keypoints_info[..., :2], keypoints_info[..., 2]


def draw_bodypose(canvas, kpts, kpt_valid, stickwidth=4,
                  radius=4, alpha=0.6, show_number=False):
    """
    Args:
        canvas (ndarray): shape[H, W, 3]
        kpts (ndarray): shape[n, 18, 2]
        kpt_valid (ndarray: bool): shape[n, 18]
        stickwidth (int): Default 4.
        radius (int): Default 4.
        alpha (float): Default 0.6.

    Returns: `canvas`: shape[H, W, 3]
    """
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
                cv2.fillConvexPoly(canvas, polygon, [int(float(c) * alpha) for c in color])

    # draw points
    for i, color in enumerate(colors):
        for kpt, is_valid in zip(kpts, kpt_valid):
            if is_valid[i]:
                cv2.circle(
                    canvas, (int(kpt[i, 0]), int(kpt[i, 1])),
                    radius, color, thickness=-1)
                if show_number:
                    cv2.putText(
                        canvas,
                        str(i), (int(kpt[i, 0]), int(kpt[i, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0),
                        lineType=cv2.LINE_AA)
    return canvas


# Reference https://github.com/lllyasviel/ControlNet/blob/main/annotator/openpose/util.py#L94
# Modified by Wilson.xu
def hand_detect(kpts,
                kpt_valid,
                img_shape,
                min_hand_bbox=20,
                ratio_wrist_elbow=0.333,
                ratio_distance=1.2):
    """
        right: shoulder 2, elbow 3, wrist 4
        left: shoulder 5, elbow 6, wrist 7
    Args:
        kpts (ndarray: float): shape[n, 18, 2], (x, y)
        kpt_valid (ndarray: bool): shape[n, 18], keypoint score > threshold, (1 or 0)
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
        ] if x2 - x1 >= min_hand_bbox and y2 - y1 >= min_hand_bbox else [0, 0, 0, 0]

    h, w = img_shape
    detect_result = []
    for body, is_valid in zip(kpts, kpt_valid):
        single_result = []
        # left hand
        if np.sum(is_valid[[6, 7]]) == 2:
            left_shoulder = body[5] if is_valid[5] == 1 else None
            out = comput_hand_bbox(body[7], body[6], h, w, left_shoulder)
            single_result.append(out)
        else:
            single_result.append([0, 0, 0, 0])
        # right hand
        if np.sum(is_valid[[3, 4]]) == 2:
            right_shoulder = body[2] if is_valid[2] == 1 else None
            out = comput_hand_bbox(body[4], body[3], h, w, right_shoulder)
            single_result.append(out)
        else:
            single_result.append([0, 0, 0, 0])
        detect_result.append(np.asarray(single_result))
    return np.stack(detect_result)


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


def get_bbox_intersection(bbox1, bbox2):
    """
    Calculate the intersection bbox of `bbox1` and `bbox2`
    Args:
        bbox1 (ndarray): shape[..., 4], `x1y1x2y2` format.
        bbox2 (ndarray): shape[..., 4], `x1y1x2y2` format.

    Returns: `inter_bbox`: shape[..., 4], `x1y1x2y2` format.
    """
    x1 = np.maximum(bbox1[..., 0:1], bbox2[..., 0:1])
    y1 = np.maximum(bbox1[..., 1:2], bbox2[..., 1:2])
    x2 = np.minimum(bbox1[..., 2:3], bbox2[..., 2:3])
    y2 = np.minimum(bbox1[..., 3:4], bbox2[..., 3:4])
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter_areas = w * h
    inter_bbox = np.concatenate([x1, y1, x2, y2], axis=-1)
    return np.where(inter_areas > 0, inter_bbox, 0)


bodypose_dict = {
        0:
        dict(name='nose', id=0, color=[255, 0, 85], type='upper', swap=''),
        1:
        dict(name='neck', id=1, color=[255, 0, 0], type='upper', swap=''),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[255, 85, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 170, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='right_wrist',
            id=4,
            color=[255, 255, 0],
            type='upper',
            swap='left_wrist'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[170, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_elbow',
            id=6,
            color=[85, 255, 0],
            type='upper',
            swap='right_elbow'),
        7:
        dict(
            name='left_wrist',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        8:
        dict(
            name='right_hip',
            id=8,
            color=[255, 0, 170],
            type='lower',
            swap='left_hip'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[255, 0, 255],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='right_ankle',
            id=10,
            color=[170, 0, 255],
            type='lower',
            swap='left_ankle'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[85, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='left_knee',
            id=12,
            color=[0, 0, 255],
            type='lower',
            swap='right_knee'),
        13:
        dict(
            name='left_ankle',
            id=13,
            color=[0, 85, 255],
            type='lower',
            swap='right_ankle'),
        14:
        dict(
            name='right_eye',
            id=14,
            color=[0, 255, 170],
            type='upper',
            swap='left_eye'),
        15:
        dict(
            name='left_eye',
            id=15,
            color=[0, 255, 255],
            type='upper',
            swap='right_eye'),
        16:
        dict(
            name='right_ear',
            id=16,
            color=[0, 170, 255],
            type='upper',
            swap='left_ear'),
        17:
        dict(
            name='left_ear',
            id=17,
            color=[0, 170, 255],
            type='upper',
            swap='right_ear'),
    }
