# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) wilson.xu. All rights reserved.
import os
from argparse import ArgumentParser

from PIL import Image
import numpy as np

from mmengine import mkdir_or_exist
from mmpose.blendpose.inferencer import VIPPoseInferencer


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('body_config', help='Config file for pose')
    parser.add_argument('body_checkpoint', help='Checkpoint file for pose')

    parser.add_argument('--template_dir', default=None, help='Template dir')
    parser.add_argument('--hand_cfg', default=None, help='Config file for pose')
    parser.add_argument('--hand_pth', default=None, help='Checkpoint file for pose')
    parser.add_argument('--face_cfg', default=None, help='Config file for pose')
    parser.add_argument('--face_pth', default=None, help='Checkpoint file for pose')
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
        '--kpt_thr',
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

    args = parser.parse_args()
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    inferencer = VIPPoseInferencer(
        args.det_config, args.det_checkpoint,
        args.body_config, args.body_checkpoint,
        False if args.hand_cfg is None else True,
        False if args.face_cfg is None else True,
        args.template_dir,
        args.return_results,
        handpose_cfg=args.hand_cfg,
        handpose_pth=args.hand_pth,
        facepose_cfg=args.face_cfg,
        facepose_pth=args.face_pth,
        kpt_thr=args.kpt_thr,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        det_cat_id=args.det_cat_id,
        draw_bbox=args.draw_bbox,
        device=args.device)

    mkdir_or_exist(args.out_dir)
    if args.img_dir:
        for line in os.listdir(args.img_dir):
            if os.path.splitext(line)[-1].lower() in ['.jpg', '.png']:
                # inference
                img_file = os.path.join(args.img_dir, line)
                img = np.asarray(Image.open(img_file).convert("RGB"))
                if args.return_results:
                    canvas, results = inferencer(img.copy(), img.copy())
                    body, hand, face = results
                    if body is not None:
                        np.save(os.path.join(args.out_dir, line.split('.')[0] + "_body"), body)
                    if hand is not None:
                        np.save(os.path.join(args.out_dir, line.split('.')[0] + "_hand"), hand)
                    if face is not None:
                        np.save(os.path.join(args.out_dir, line.split('.')[0] + "_face"), face)
                else:
                    canvas = inferencer(img.copy(), img.copy())
                Image.fromarray(canvas).save(os.path.join(args.out_dir, line))
                print(f"{os.path.join(args.out_dir, line)} has been saved")
    elif args.img_file:
        img = np.asarray(Image.open(args.img_file).convert("RGB"))
        name = os.path.basename(args.img_file)
        if args.return_results:
            canvas, results = inferencer(img.copy(), img.copy())
            body, hand, face = results
            if body is not None:
                np.save(os.path.join(args.out_dir, name.split('.')[0] + "_body"), body)
            if hand is not None:
                np.save(os.path.join(args.out_dir, name.split('.')[0] + "_hand"), hand)
            if face is not None:
                np.save(os.path.join(args.out_dir, name.split('.')[0] + "_face"), face)
        else:
            canvas = inferencer(img.copy(), img.copy())
        Image.fromarray(canvas).save(os.path.join(args.out_dir, name))
        print(f"{os.path.join(args.out_dir, name)} has been saved")
    else:
        raise Exception('Please enter the correct `img_file` or `img_dir`!')


if __name__ == '__main__':
    main()
    print("Done!")
