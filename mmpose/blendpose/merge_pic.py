# Copyright (c) wilson.xu. All rights reserved.
import os

import numpy as np
from mmengine import mkdir_or_exist
from PIL import Image

rtmpose_dir = '/Users/wilson.xu/PR/mmpose/output_rtmpose/woman'
openpose_dir = '/Users/wilson.xu/PR/demo/openpose_0620/woman'
out_dir = 'merge_res/woman'

mkdir_or_exist(out_dir)

for name in os.listdir(rtmpose_dir):
    if os.path.splitext(name)[-1].lower() in ['.jpg', '.png']:
        if not os.path.exists(os.path.join(openpose_dir, name)):
            continue
        rtm = np.array(
            Image.open(os.path.join(rtmpose_dir, name)).convert('RGB'))
        open = np.array(
            Image.open(os.path.join(openpose_dir, name)).convert('RGB'))
        out = np.concatenate([rtm, open], axis=1)
        Image.fromarray(out).save(os.path.join(out_dir, name))
        print(f'{name} saved!')
print('Done!')

# import cv2
# import numpy as np
# from PIL import Image
#
# img = np.array(
#     Image.open("/Users/wilson.xu/Downloads/testdataset_0612/man/image24.png").convert("RGB"))
#
# vis = cv2.Canny(img, 100, 200)
# Image.fromarray(vis).save("canny.jpg")
#
# print("DOne!")
