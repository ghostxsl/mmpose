# Copyright (c) wilson.xu. All rights reserved.
import os

import numpy as np
import torch

from .body import Body
from .face import Face
from .hand import Hand
from .utils import (load_file_from_url, model_path, openposeDrawBodyPose,
                    openposeDrawFacePose, openposeDrawHandPose,
                    openposeFaceDetect, openposeHandDetect, save_weights_dir)


class OpenposeDetector(object):

    def __init__(self,
                 use_hand=False,
                 use_face=False,
                 body_pth=None,
                 hand_pth=None,
                 face_pth=None,
                 return_results=False,
                 device='cuda'):
        self.use_hand = use_hand
        self.use_face = use_face
        self.return_results = return_results
        self.device = device

        body_pth = body_pth or os.path.join(save_weights_dir,
                                            'body_pose_model.pth')
        hand_pth = hand_pth or os.path.join(save_weights_dir,
                                            'hand_pose_model.pth')
        face_pth = face_pth or os.path.join(save_weights_dir, 'facenet.pth')

        if not os.path.exists(body_pth):
            body_pth = load_file_from_url(
                model_path['body'], model_dir=save_weights_dir)
        if not os.path.exists(hand_pth) and use_hand:
            hand_pth = load_file_from_url(
                model_path['hand'], model_dir=save_weights_dir)
        if not os.path.exists(face_pth) and use_face:
            face_pth = load_file_from_url(
                model_path['face'], model_dir=save_weights_dir)

        self.body_estimation = Body(body_pth, device=device)
        self.hand_estimation = Hand(
            hand_pth, device=device) if use_hand else None
        self.face_estimation = Face(
            face_pth, device=device) if use_face else None

    def __call__(self, oriImg, canvas=None, hand=False, face=False):
        H, W, C = oriImg.shape
        assert C == 3, 'The input image can only be in RGB format.'
        if canvas is None:
            canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        body_res, hand_res, face_res = None, None, None
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            # 1. Body
            candidate, subset = self.body_estimation(oriImg)
            canvas = openposeDrawBodyPose(canvas, candidate, subset)
            if self.return_results:
                # save body results
                body_res = {
                    'candidate': candidate,
                    'subset': subset
                }
            # 2. Hand
            if hand and self.use_hand:
                hands = []
                hands_list = openposeHandDetect(candidate, subset, oriImg)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y + w, x:x +
                                                        w]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1,
                                               peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1,
                                               peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                canvas = openposeDrawHandPose(canvas, hands)
                if self.return_results:
                    # save hand results
                    hand_res = {
                        'bboxes': hands_list,
                        'keypoints': hands
                    }
            # 3. Face
            if face and self.use_face:
                faces = []
                faces_list = openposeFaceDetect(candidate, subset, oriImg)
                for x, y, w in faces_list:
                    peaks = self.face_estimation(oriImg[y:y + w, x:x + w])
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1,
                                               peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1,
                                               peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())
                canvas = openposeDrawFacePose(canvas, faces)
                if self.return_results:
                    # save face results
                    face_res = {
                        'bboxes': faces_list,
                        'keypoints': faces
                    }
        if self.return_results:
            return canvas, {'body': body_res, 'hand': hand_res, 'face': face_res}
        else:
            return canvas
