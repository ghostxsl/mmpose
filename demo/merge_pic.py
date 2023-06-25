# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
from infer_rtmpose_blend import mkdir_or_exist
from PIL import Image

rtmpose_dir = '/Users/wilson.xu/PR/mmpose/demo/mmpose_0620/man'
openpose_dir = '/Users/wilson.xu/PR/mmpose/demo/openpose_0620/man'
out_dir = 'merge_res/man'

# rtmpose_dir = "/Users/wilson.xu/PR/mmpose/demo/mmpose_0620/woman"
# openpose_dir = "/Users/wilson.xu/PR/mmpose/demo/openpose_0620/woman"
# out_dir = "merge_res/woman"

mkdir_or_exist(out_dir)

for name in os.listdir(rtmpose_dir):
    if os.path.splitext(name)[-1].lower() in ['.jpg', '.png']:
        rtm = np.array(
            Image.open(os.path.join(rtmpose_dir, name)).convert('RGB'))
        open = np.array(
            Image.open(os.path.join(openpose_dir, name)).convert('RGB'))
        out = np.concatenate([rtm, open], axis=1)
        Image.fromarray(out).save(os.path.join(out_dir, name))
        print(f'{name} saved!')
print('Done!')
