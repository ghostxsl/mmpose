# 环境依赖
```commandline
pip install mmengine
pip install mmcv
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
git clone http://gitlab.tools.vipshop.com/wilson.xu/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

# 使用 RTMPoseInfer
以下config路径均是相对于./mmpose
```
rtmpose_infer = RTMPoseInfer(
    "mmpose/demo/mmdetection_cfg/rtmdet_l_8xb32-300e_coco.py/rtmdet_l_8xb32-300e_coco.py",
    "path_to_weights/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-210e_body8-256x192.py",
    "path_to_weights/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
    None, # 若使用openpose请指定 None；若使用rtmpose请指定config: mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py
    "path_to_weights/ControlNet/annotator/ckpts/hand_pose_model.pth",
    "mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py",
    "path_to_weights/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-62026ef2_20230228.pth",
)
```
