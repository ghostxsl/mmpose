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


def draw_bodypose(canvas, kpts, kpt_valid, is_origin=False,
                  stickwidth=4, radius=4, alpha=0.6, show_number=False):
    """
    Args:
        canvas (ndarray): shape[H, W, 3]
        kpts (ndarray): shape[n, 18, 2]
        kpt_valid (ndarray: bool): shape[n, 18]
        is_origin (bool): If `canvas` is not the original image,
                        it needs to be multiplied by `alpha`. Default: False.
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
                cv2.fillConvexPoly(canvas, polygon, color)

    if not is_origin:
        canvas = (canvas.astype('float32') * alpha).astype('uint8')
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
        # right hand
        if np.sum(is_valid[[3, 4]]) == 2:
            right_shoulder = body[2] if is_valid[2] == 1 else None
            out = comput_hand_bbox(body[4], body[3], h, w, right_shoulder)
            single_result.append(out)
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


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(
            weights_name.split('.')[1:])]
    return transfered_model_weights


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [0,]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def openposeHandDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[
                5, 6, 7
            ]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            distanceElbowShoulder = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])
    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left
    '''
    return detect_result


def openposeDrawHandPose(canvas, all_hand_peaks, eps=1e-3):
    H, W, _ = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas, (x1, y1), (x2, y2),
                    matplotlib.colors.hsv_to_rgb(
                        [ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


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
