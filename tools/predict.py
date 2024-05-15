import argparse
import sys
import os.path as osp
import warnings
import mmcv
import torch
import cv2
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0. Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose predict keypoints')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--image', help='image file path')
    parser.add_argument('--out', default=None, help='output result file')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={}, help='override some settings in the used config')
    args = parser.parse_args()
    return args


def visualize_keypoints(image_path, keypoints, bbox, output_path):
    image = cv2.imread(image_path)

    # Draw bounding box
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw keypoints
    for keypoint in keypoints:
        x, y, conf = keypoint
        if conf > 0.5:  # Only draw keypoints with confidence > 0.5
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imwrite(output_path, image)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # load model
    model = init_pose_model(cfg, args.checkpoint, device='cuda:{}'.format(args.gpu_id))

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model.eval()

    # Perform inference
    result, _ = inference_top_down_pose_model(model, args.image, format='xyxy')

    # Visualize keypoints and bounding box
    keypoints = result[0]['keypoints']
    bbox = result[0]['bbox']
    output_image_path = args.out if args.out else 'output_image.jpg'
    visualize_keypoints(args.image, keypoints, bbox, output_image_path)

    # Save or print results
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(result, args.out)
    else:
        print(result)


if __name__ == '__main__':
    main()