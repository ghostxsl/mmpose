# 环境依赖
```commandline
pip install mmengine
pip install mmcv
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
git clone https://github.com/ghostxsl/mmpose.git
cd mmpose
git checkout blend_rtmpose
pip install -r requirements.txt
pip install -v -e .
```

# 使用 VIPPoseInferencer
以下config路径均是相对于./mmpose
```
from mmpose.blendpose.inferencer import VIPPoseInferencer

pose_inferencer = VIPPoseInferencer(
    det_cfg="mmpose/demo/mmdetection_cfg/rtmdet_l_8xb32-300e_coco.py/rtmdet_l_8xb32-300e_coco.py",
    det_pth="path_to_weights/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    bodypose_cfg="mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-210e_body8-256x192.py",
    bodypose_pth="path_to_weights/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
    use_hand=True, # 是否使用hand pose estimator
    use_face=True, # 是否使用face pose estimator
    template_dir="path_to_template_dir", # 若使用手部pose template，请指定template目录
    handpose_cfg="openpose", # 若使用openpose请指定"openpose"；若使用rtmpose请指定config路径: mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py
    handpose_pth="path_to_weights/ControlNet/annotator/ckpts/hand_pose_model.pth",
    facepose_cfg="mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py",
    facepose_pth="path_to_weights/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-62026ef2_20230228.pth",
)
```
