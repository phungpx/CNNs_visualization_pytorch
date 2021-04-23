import os
import sys
sys.path.append(os.environ['PWD'])

import cv2
import random
import argparse
from pathlib import Path

import utils


def _resize(image, max_dim=800):
    h, w = image.shape[:2]
    if (h > w) and (h > max_dim):
        image = cv2.resize(image, dsize=(int(max_dim * w / h), max_dim))
    elif (w > h) and (w > max_dim):
        image = cv2.resize(image, dsize=(max_dim, int(max_dim * h / w)))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='path to image.')
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    if args.pattern:
        image_paths = list(Path(args.image_path).glob(args.pattern))
    else:
        image_paths = [Path(args.image_path)]

    if args.random:
        random.shuffle(image_paths)

    config = utils.load_yaml('./modules/gradCAM/config.yaml')
    visualizer = utils.create_instance(config['gradCAM'])
    # visualizer = utils.create_instance(config['gradCAM_pretrained'])

    for i, image_path in enumerate(image_paths):
        if i == 10:
            break
        image = cv2.imread(str(image_path))
        _, heatmap = visualizer(image)
        cv2.imshow('grad CAM', _resize(heatmap))
        cv2.waitKey()
        cv2.destroyAllWindows()
