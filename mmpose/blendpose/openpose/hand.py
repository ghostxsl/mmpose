# Copyright (c) wilson.xu. All rights reserved.
import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label

from . import utils
from .model import HandPoseModel


class Hand(object):

    def __init__(self, model_path, device='cuda'):
        self.device = device
        model = HandPoseModel()
        model_dict = utils.transfer(model, torch.load(model_path))
        model.load_state_dict(model_dict)
        self.model = model.to(device)
        self.model.eval()
        print(f'Loads checkpoint by local backend from path: {model_path}')

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        padValue = 128
        threshold = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(
                oriImg, (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = utils.padRightDownCorner(
                imageToTest, stride, padValue)
            im = np.transpose(
                np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float().to(self.device)
            with torch.no_grad():
                output = self.model(data).cpu().numpy()

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(output),
                                   (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(
                heatmap, (0, 0),
                fx=stride,
                fy=stride,
                interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] -
                              pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(
                heatmap, (oriImg.shape[1], oriImg.shape[0]),
                interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(
                one_heatmap > threshold, dtype=np.uint8)
            # All below threshold
            if np.sum(binary) == 0:
                all_peaks.append([-1, -1])
                continue
            label_img, label_numbers = label(
                binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([
                np.sum(map_ori[label_img == i])
                for i in range(1, label_numbers + 1)
            ]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = utils.npmax(map_ori)
            all_peaks.append([x, y])
        return np.asarray(all_peaks)
