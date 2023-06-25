# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import numpy as np
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--img_file', default=None, help='Image file')
    parser.add_argument('--img_dir', default=None, help='Image dir')

    parser.add_argument('--out_file', default=None, help='Path to output file')
    parser.add_argument(
        '--out_dir', default='output', help='Dir to output files')

    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt_thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def infer_one_image(model,
                    visualizer,
                    img_path,
                    out_file,
                    kpt_thr=0.3,
                    draw_heatmap=False,
                    show_kpt_idx=False,
                    skeleton_style='mmpose',
                    show=False):
    # inference a single image
    batch_results = inference_topdown(model, img_path)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(img_path, channel_order='rgb')
    # img = np.zeros_like(img)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=False,
        kpt_thr=kpt_thr,
        draw_heatmap=draw_heatmap,
        show_kpt_idx=show_kpt_idx,
        skeleton_style=skeleton_style,
        show=show,
        out_file=out_file)


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    if args.img_dir:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        img_list = os.listdir(args.img_dir)
        for line in img_list:
            if os.path.splitext(line)[1] in ['.jpg', '.png']:
                out_file = os.path.join(args.out_dir, line)
                line = os.path.join(args.img_dir, line)
                infer_one_image(
                    model,
                    visualizer,
                    line,
                    out_file,
                    kpt_thr=args.kpt_thr,
                    draw_heatmap=args.draw_heatmap,
                    show_kpt_idx=args.show_kpt_idx,
                    skeleton_style=args.skeleton_style,
                    show=args.show)
    elif args.img_file:
        infer_one_image(
            model,
            visualizer,
            args.img_file,
            args.out_file,
            kpt_thr=args.kpt_thr,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show)
    else:
        raise Exception('Please enter the correct image path or folder path!')


if __name__ == '__main__':
    main()
