# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) wilson.xu. All rights reserved.
import os
from argparse import ArgumentParser

from PIL import Image
import numpy as np

from mmengine import mkdir_or_exist
from mmpose.blendpose.inferencer import VIPPoseInferencer
from mmpose.blendpose.utils import pkl_save


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')

    parser.add_argument('--body_cfg', default=None, help='Config file for pose')
    parser.add_argument('--body_pth', default=None, help='Checkpoint file for pose')
    parser.add_argument('--wholebody_cfg', default=None, help='Config file for pose')
    parser.add_argument('--wholebody_pth', default=None, help='Checkpoint file for pose')
    parser.add_argument('--hand_cfg', default=None, help='Config file for pose')
    parser.add_argument('--hand_pth', default=None, help='Checkpoint file for pose')
    parser.add_argument('--face_cfg', default=None, help='Config file for pose')
    parser.add_argument('--face_pth', default=None, help='Checkpoint file for pose')

    parser.add_argument('--template_dir', default=None, help='Template dir')
    parser.add_argument(
        '--return_results', action='store_true', help='Return results')

    parser.add_argument(
        '--img_file', type=str, default=None, help='Image file')
    parser.add_argument(
        '--img_dir', type=str, default=None, help='Image dir')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='output',
        help='root of the output img file. '
             'Default not saving the visualization images.')

    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.4,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.4,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--body_kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--hand_kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--face_kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw_bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--draw_img', action='store_true', help='Draw results on the image')
    parser.add_argument(
        '--is_hand_intersection', action='store_true', help='is_hand_intersection')
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index')
    parser.add_argument(
        '--end',
        type=int,
        default=-1,
        help='Ending Index')

    args = parser.parse_args()
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    inferencer = VIPPoseInferencer(
        args.det_config, args.det_checkpoint,
        bodypose_cfg=args.body_cfg,
        bodypose_pth=args.body_pth,
        handpose_cfg=args.hand_cfg,
        handpose_pth=args.hand_pth,
        facepose_cfg=args.face_cfg,
        facepose_pth=args.face_pth,
        wholebodypose_cfg=args.wholebody_cfg,
        wholebodypose_pth=args.wholebody_pth,
        template_dir=args.template_dir,
        is_hand_intersection=args.is_hand_intersection,
        body_kpt_thr=args.body_kpt_thr,
        hand_kpt_thr=args.hand_kpt_thr,
        face_kpt_thr=args.face_kpt_thr,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        det_cat_id=args.det_cat_id,
        draw_bbox=args.draw_bbox,
        device=args.device)

    mkdir_or_exist(args.out_dir)
    if args.img_dir:
        if args.return_results:
            mkdir_or_exist(os.path.join(args.out_dir, "pose"))
            mkdir_or_exist(os.path.join(args.out_dir, "res"))
        img_list = os.listdir(args.img_dir)
        start = args.start
        end = args.end if args.end > 0 else len(img_list)
        img_list = img_list[start:end]
        print(f"infer index: {start} => {end}")
        for i, line in enumerate(img_list):
            if os.path.splitext(line)[-1].lower() in ['.jpg', '.jpeg', '.png']:
                # inference
                img_file = os.path.join(args.img_dir, line)
                img = np.asarray(Image.open(img_file).convert("RGB"))
                if args.return_results:
                    canvas, results = inferencer(img.copy(),
                                                 img.copy() if args.draw_img else None,
                                                 return_results=True)
                    pkl_save(results,
                             os.path.join(args.out_dir, "pose", os.path.splitext(line)[0] + "_pose.pkl"))
                    save_name = os.path.join(args.out_dir, "res", line)
                else:
                    canvas = inferencer(img.copy(),
                                        img.copy() if args.draw_img else None)
                    save_name = os.path.join(args.out_dir, line)
                Image.fromarray(canvas).save(save_name)
                print(f"{i + start}/{len(img_list) + start}: {save_name} has been saved")
    elif args.img_file:
        img = np.asarray(Image.open(args.img_file).convert("RGB"))
        name = os.path.basename(args.img_file)
        if args.return_results:
            canvas, results = inferencer(img.copy(),
                                         img.copy() if args.draw_img else None,
                                         return_results=True)
            pkl_save(results,
                     os.path.join(args.out_dir, name.split('.')[0] + "_pose.pkl"))
        else:
            canvas = inferencer(img.copy(),
                                img.copy() if args.draw_img else None)
        Image.fromarray(canvas).save(os.path.join(args.out_dir, name))
        print(f"{os.path.join(args.out_dir, name)} has been saved")
    else:
        raise Exception('Please enter the correct `img_file` or `img_dir`!')


if __name__ == '__main__':
    main()
    print("Done!")
