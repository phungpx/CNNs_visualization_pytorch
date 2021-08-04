import cv2
import argparse

from pathlib import Path
from natsort import natsorted

import os
import sys

sys.path.append(os.environ['PWD'])

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image dir.')
    parser.add_argument('--output-dir', help='path to save image')
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    parser.add_argument('--show-image', action='store_true')
    parser.add_argument('--start-index', default=1)
    args = parser.parse_args()

    image_paths = list(Path(args.image_path).glob(args.pattern)) if args.pattern else [Path(args.image_path)]
    image_paths = natsorted(image_paths, key=lambda x: x.stem)

    output_dir = Path(args.output_dir) if args.output_dir else Path('output/flask/')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    config = utils.load_yaml('modules/flask/config.yaml')
    predictor = utils.create_instance(config['cifar_10'])

    for idx, image_path in enumerate(image_paths[int(args.start_index) - 1:], int(args.start_index)):
        print('-' * 50)
        print(f'{idx} / {len(image_paths)} - {image_path}')

        images = [cv2.imread(str(image_path))]

        outputs = predictor(images=images)
        for i, (image, (class_name, class_score)) in enumerate(zip(images, outputs)):
            cv2.putText(img=image,
                        text=f'{class_name} ({class_score * 100:.2f}%)',
                        org=(0, int(0.06 * image.shape[0])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=max(image.shape) / 1000,
                        color=(255, 255, 255),
                        thickness=max(image.shape) // 400)

            if i == 0:
                cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
            else:
                cv2.imwrite(str(output_dir.joinpath(f'{image_path.stem}_{i}{image_path.suffix}')), image)

            if args.show_image:
                cv2.imshow(image_path.name, image)
                cv2.waitKey()
                cv2.destroyAllWindows()
