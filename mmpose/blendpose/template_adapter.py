# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import exists, join, splitext, expanduser
import random
import numpy as np
from mmpose.blendpose.utils import pkl_load


class TemplatePoseAdapter(object):
    def __init__(self, template_dir):
        template_dir = expanduser(template_dir)
        self.template = self.create_template_dict(template_dir)

    @staticmethod
    def create_template_dict(template_dir):
        src_dir = join(template_dir, "src")
        pose_dir = join(template_dir, "pose")
        lists = [a for a in os.listdir(src_dir) if splitext(
            a)[-1].lower() in ['.jpg', '.png']]
        out_dict = dict()
        for i, name in enumerate(lists):
            prefix = splitext(name)[0]
            if exists(join(pose_dir, prefix + '_pose.pkl')):
                out_dict[i] = pkl_load(join(pose_dir, prefix + '_pose.pkl'))
                out_dict[i]['name'] = name
        print(f"Load {len(out_dict)} pose templates from {template_dir}")
        return out_dict

    def compute_scale_factor(self, src_body, template_body):
        # left: elbow 6, wrist 7
        # right: elbow 3, wrist 4
        left_elbow_wrist_src = np.sqrt(
            np.square(src_body[:, 7] - src_body[:, 6]).sum(-1))
        left_elbow_wrist_template = np.sqrt(
            np.square(template_body[0:1, 7] - template_body[0:1, 6]).sum(-1))
        left_scale_factor = left_elbow_wrist_src / left_elbow_wrist_template

        right_elbow_wrist_src = np.sqrt(
            np.square(src_body[:, 4] - src_body[:, 3]).sum(-1))
        right_elbow_wrist_template = np.sqrt(
            np.square(template_body[0:1, 4] - template_body[0:1, 3]).sum(-1))
        right_scale_factor = right_elbow_wrist_src / right_elbow_wrist_template
        return np.stack([left_scale_factor, right_scale_factor], axis=-1)

    def __call__(self, canvas, body_kpts, body_kpt_valid, **kwargs):
        # random select template hand pose
        template = random.choice(self.template)

        H, W, _ = canvas.shape
        # 1. scale hand pose
        template_body = template['body']['keypoints'][..., :2] * np.array([W - 1, H - 1])
        template_hand = template['hand']['keypoints'][..., :2] * np.array([W - 1, H - 1])
        # compute scaling factor
        scale_factor = self.compute_scale_factor(body_kpts, template_body)
        num_hands, num_pts, _ = template_hand.shape
        # left and right hand
        left_hands = []
        right_hands = []
        for s in scale_factor:
            left_hands.append(points_scale(template_hand[0], s[0], s[0]))
            right_hands.append(points_scale(template_hand[1], s[1], s[1]))

        # 2. translate hand pose
        for i, src_body in enumerate(body_kpts):
            # left wrist: 7
            left_hands[i] = points_translation(left_hands[i], left_hands[i][0], src_body[7])
            # right wrist: 4
            right_hands[i] = points_translation(right_hands[i], right_hands[i][0], src_body[4])

        # 3. rotate hand pose
        for i, src_body in enumerate(body_kpts):
            # left: elbow 6, wrist 7
            vx_left = left_hands[i][9] - left_hands[i][0]
            vy_left = src_body[7] - src_body[6]
            left_hands[i] = points_rotate(left_hands[i], vx_left, vy_left, src_body[7])
            # right: elbow 3, wrist 4
            vx_right = right_hands[i][9] - right_hands[i][0]
            vy_right = src_body[4] - src_body[3]
            right_hands[i] = points_rotate(right_hands[i], vx_right, vy_right, src_body[4])

        hand_kpts = np.concatenate([np.stack(left_hands), np.stack(right_hands)])
        hand_kpt_valid = np.ones(hand_kpts.shape[:2])
        hand_kpt_valid = hand_kpt_valid.reshape([-1, 2, 21])
        # left wrist: 7
        hand_kpt_valid = np.where(body_kpt_valid[:, 7:8][:, None], hand_kpt_valid, 0)
        # right wrist: 4
        hand_kpt_valid = np.where(body_kpt_valid[:, 4:5][:, None], hand_kpt_valid, 0)
        return hand_kpts, hand_kpt_valid.reshape([-1, 21])


def points_translation(points, px, py):
    """
    translate `points` from `px` to `py`.
    M * P: | 1, 0, delta_x |   | x |
           | 0, 1, delta_y | * | y |
                               | 1 |
    Args:
        points (ndarray: float): shape[n, 2]
        px (list|tuple): coordinate(x ,y)
        py (list|tuple): coordinate(x ,y)

    Returns: `out_points`(ndarray: float): shape[n, 2]
    """
    n, _ = points.shape
    M = np.zeros([2, 3])
    M[0, 0], M[1, 1] = 1, 1
    M[0, 2] = py[0] - px[0]
    M[1, 2] = py[1] - px[1]
    points = np.concatenate([points, np.ones([n, 1])], axis=-1).T
    return np.matmul(M, points).T


def points_horizontal_flip(points, W):
    """
    horizontal flip `points`.
    M * P: |-1, 0, W |   | x |
           | 0, 1, 0 | * | y |
                         | 1 |
    Args:
        points (ndarray: float): shape[n, 2]
        W (int): The width of the image.

    Returns: `out_points`(ndarray: float): shape[n, 2]
    """
    n, _ = points.shape
    M = np.zeros([2, 3])
    M[0, 0], M[1, 1] = -1, 1
    M[0, 2] = W
    points = np.concatenate([points, np.ones([n, 1])], axis=-1).T
    return np.matmul(M, points).T


def points_scale(points, fx, fy):
    """
    Scale `points` according to `fx` and `fy`.
    M * P: |fx,  0,  W|   | x |
           | 0, fy,  0| * | y |
                          | 1 |
    Args:
        points (ndarray: float): shape[n, 2]
        fx (float): Scale factor in the x-direction.
        fy (float): Scale factor in the y-direction.

    Returns: `out_points`(ndarray: float): shape[n, 2]
    """
    n, _ = points.shape
    M = np.zeros([2, 3])
    M[0, 0], M[1, 1] = fx, fy
    points = np.concatenate([points, np.ones([n, 1])], axis=-1).T
    return np.matmul(M, points).T


def points_rotate(points, vx, vy, center=(0, 0)):
    """
    rotate `points` from `vector_x` to `vector_y`.
    the center of rotation is `center`.
    M * P: | cos_a, sin_a,  (1 - cos_a) * cx - cy * sin_a|   | x |
           |-sin_a, cos_a,  (1 - cos_a) * cy + cx * sin_a| * | y |
                                                             | 1 |
    Args:
        points (ndarray: float): shape[n, 2]
        vx (list|tuple): coordinate(x ,y)
        vy (list|tuple): coordinate(x ,y)
        center (list|tuple): coordinate(x ,y)

    Returns: `out_points`(ndarray: float): shape[n, 2]
    """
    def get_clock_angle(vx, vy):
        # clockwise is negative, counterclockwise is positive.
        norm_ = np.linalg.norm(vx) * np.linalg.norm(vy)
        # radians
        rho_ = np.arcsin(np.cross(vx, vy) / norm_)
        theta = np.arccos(np.dot(vx, vy) / norm_)
        return theta if rho_ < 0 else -theta

    n, _ = points.shape
    M = np.zeros([2, 3])
    theta = get_clock_angle(vx, vy)
    # radians
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    M[0, 0], M[1, 1] = cos_theta, cos_theta
    M[0, 1], M[1, 0] = sin_theta, -sin_theta
    M[0, 2] = (1 - cos_theta) * center[0] - center[1] * sin_theta
    M[1, 2] = (1 - cos_theta) * center[1] + center[0] * sin_theta
    # M_1 = cv2.getRotationMatrix2D(center, np.rad2deg(theta), 1.0)
    points = np.concatenate([points, np.ones([n, 1])], axis=-1).T
    return np.matmul(M, points).T
