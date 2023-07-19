# Copyright (c) wilson.xu. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from .model import FaceNet
from .utils import smart_resize

params = {
    'gaussian_sigma':
    2.5,
    'inference_img_size':
    736,  # 368, 736, 1312
    'heatmap_peak_thresh':
    0.1,
    'crop_scale':
    1.5,
    'line_indices':
    [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
     [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
     [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25],
     [25, 26], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34],
     [34, 35], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
     [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42], [48, 49],
     [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56],
     [56, 57], [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63],
     [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]],
}


class Face(object):
    """The OpenPose face landmark detector model.

    Args:
        inference_size: set the size of the inference image size, suggested:
            368, 736, 1312, default 736
        gaussian_sigma: blur the heatmaps, default 2.5
        heatmap_peak_thresh: return landmark if over threshold, default 0.1
    """

    def __init__(self,
                 model_path,
                 inference_size=None,
                 gaussian_sigma=None,
                 heatmap_peak_thresh=None,
                 device='cuda'):
        self.device = device
        self.inference_size = inference_size or params['inference_img_size']
        self.sigma = gaussian_sigma or params['gaussian_sigma']
        self.threshold = heatmap_peak_thresh or params['heatmap_peak_thresh']
        self.model = FaceNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
        print(f'Loads checkpoint by local backend from path: {model_path}')

    def __call__(self, face_img):
        H, W, C = face_img.shape

        w_size = 384
        x_data = torch.from_numpy(smart_resize(
            face_img, (w_size, w_size))).permute([2, 0, 1]) / 256.0 - 0.5
        x_data = x_data.to(self.device)

        with torch.no_grad():
            hs = self.model(x_data[None, ...])
            heatmaps = F.interpolate(
                hs[-1], (H, W), mode='bilinear',
                align_corners=True).cpu().numpy()[0]
        return self.compute_peaks_from_heatmaps(heatmaps)

    def compute_peaks_from_heatmaps(self, heatmaps):
        all_peaks = []
        for part in range(heatmaps.shape[0]):
            map_ori = heatmaps[part].copy()
            binary = np.ascontiguousarray(map_ori > 0.05, dtype=np.uint8)

            if np.sum(binary) == 0:
                continue

            positions = np.where(binary > 0.5)
            intensities = map_ori[positions]
            mi = np.argmax(intensities)
            y, x = positions[0][mi], positions[1][mi]
            all_peaks.append([x, y])

        return np.asarray(all_peaks)
