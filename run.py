import cv2
import argparse

from pathlib import Path
from natsort import natsorted


import os
import sys
sys.path.append(os.environ['PWD'])

import utils


def _resize(image, max_dimension=500):
    h, w = image.shape[:2]
    if (h > w) and (h > max_dimension):
        image = cv2.resize(image, dsize=(int(max_dimension * w / h), max_dimension))
    elif (w > h) and (w > max_dimension):
        image = cv2.resize(image, dsize=(max_dimension, int(max_dimension * h / w)))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image dir.')
    parser.add_argument('--config-path', default='./modules/CAM/config.yaml')
    parser.add_argument('--module-name', default='cifar10')
    parser.add_argument('--output-dir', help='path to save image')
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    parser.add_argument('--show-image', action='store_true')
    parser.add_argument('--start-index', default=1)
    args = parser.parse_args()

    image_paths = list(Path(args.image_path).glob(args.pattern)) if args.pattern else [Path(args.image_path)]
    image_paths = natsorted(image_paths, key=lambda x: x.stem)

    output_dir = Path(args.output_dir) if args.output_dir else Path('output/')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    config = utils.load_yaml(args.config_path)
    visualizer = utils.create_instance(config[args.module_name])

    for idx, image_path in enumerate(image_paths[int(args.start_index) - 1:], int(args.start_index)):
        print('-' * 50)
        print(f'{idx} / {len(image_paths)} - {image_path}')

        image = cv2.imread(str(image_path))

        heatmap, class_name, class_score = visualizer(image)

        # cv2.putText(img=heatmap, text=f'{class_name}_{class_score: .2f}', origin=)

        cv2.imwrite(str(output_dir.joinpath(image_path.name)), heatmap)

        if args.show_image:
            cv2.imshow(f'{class_name}_{class_score: .2f}', _resize(heatmap))
            cv2.waitKey()
            cv2.destroyAllWindows()
