import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classes_file", type=str, help="classes file", required=True)
parser.add_argument("--dataset_folder", type=str, help="Folder where the data is", required=True)
parser.add_argument('--relative_labels', type=str, help='path to csv file with labels', required=True)
parser.add_argument('--weights', type=str, help='Previous weights', required=False, default=None)
parser.add_argument('--augmentation', action='store_true', help='augmentation', required=False, default=False)
args = parser.parse_args()
print(args)

import sys
sys.path.append('..')

from trainer import Trainer
from Config.augmentation_options import preset_1


if __name__ == '__main__':
    DATASET_NAME = 'mask-faces'

    tr = Trainer(
            input_shape=(160, 160, 3),
            classes_file=args.classes_file,
            image_width=640,  # The original image width
            image_height=480,   # The original image height
            image_folder=args.dataset_folder
    )

    dataset_conf = {
                'relative_labels': args.relative_labels,
                'dataset_name': DATASET_NAME,
                'test_size': 0.1,
                'sequences': preset_1,  # check Config > augmentation_options.py
                'augmentation': args.augmentation,
    }

    anchors_conf = {
                'anchor_no': 9,
                'relative_labels':  args.relative_labels
    }

    tr.train(epochs=100, 
            batch_size=8, 
            learning_rate=1e-3, 
            dataset_name=DATASET_NAME, 
            merge_evaluation=False,
            min_overlaps=0.5,
            new_dataset_conf=dataset_conf,
            new_anchors_conf=anchors_conf,
            weights=args.weights)