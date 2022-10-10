import os
import numpy as np
import argparse
from core.dataset.SensatUrban_ultra import SensatUrban_ultra
from insaug.utils import mean_shift, compute_covariance_matrix, compute_variance


def get_parser():
    parser = argparse.ArgumentParser('SensatUrban Dataset Augment Preprocessing')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--save_path', type=str, default='./instance/', help='instance save path')
    parser.add_argument('--augment_classes', type=str, default='11', help='augment classes such as 11,12,13')

    args = parser.parse_args()
    return args


class GenInstance3D:
    def __init__(self, train_dataset, val_dataset, save_path="./instance/", augment_class=None):
        if augment_class is None:
            augment_class = [11]
        self.augment_class = augment_class
        if isinstance(self.augment_class, int):
            self.augment_class = [self.augment_class]
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_to_name = self.train_dataset.label_to_names

        # create save path
        self.save_path = save_path
        for label in self.augment_class:
            dir_path = os.path.join(self.save_path, self.label_to_name[label])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def gen_instance(self, split):
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            raise ValueError('split must be train or val')

        for i in range(len(dataset)):
            data = dataset[i]
            file_name = data["file_name"]
            cloud_idx = data["cloud_index"]
            selected_points = data['lidar'].F
            inverse_map = data['inverse_map'].F
            all_points = selected_points[inverse_map]
            all_labels = data['targets_mapped'].F
            sub_points = [all_points[all_labels == i] for i in self.augment_class]
            print(f"{i}/{len(dataset)} ", file_name)
            for class_name, points in zip(self.augment_class, sub_points):
                labels, cluster_centers, n_clusters = mean_shift(points)
                print(f"{class_name}: {labels.shape}, {cluster_centers}, {n_clusters}")
                for j in range(n_clusters):
                    ins_points = points[labels == j]
                    ins_points -= np.mean(ins_points, axis=0)
                    save_name = f"{cloud_idx}_{j}_{ins_points.shape[0]}.npy"
                    print("    saving {}, variance: {:.4f}".format(save_name, compute_variance(ins_points)))
                    np.save(os.path.join(self.save_path, self.label_to_name[class_name], save_name),
                            ins_points)


def main():
    args = get_parser()
    augment_classes = [int(i) for i in args.augment_classes.split(',')]
    dataset = SensatUrban_ultra(0.2, 0, args.data_root, bev_name='bev')
    gen_instance = GenInstance3D(dataset['train'], dataset['val'], args.save_path, augment_classes)
    gen_instance.gen_instance('train')
    gen_instance.gen_instance('val')


