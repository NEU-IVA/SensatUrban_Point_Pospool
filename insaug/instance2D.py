import os.path

import numpy as np
import cv2
import argparse
from tqdm import tqdm
from sklearn.cluster import MeanShift, estimate_bandwidth
from insaug.utils import mean_shift, compute_variance

label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                  4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                  9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}


def get_parser():
    parser = argparse.ArgumentParser('SensatUrban Dataset Augment Preprocessing')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--save_path', type=str, default='./instance/', help='instance save path')
    parser.add_argument('--augment_classes', type=str, default='11', help='augment classes such as 11,12,13')

    args = parser.parse_args()
    return args


def load_rgbd(img_path):
    depth_path = img_path[:-3].replace('img', 'depth')+'npy'
    img = cv2.imread(img_path)
    depth = np.load(depth_path)
    rgbd = np.concatenate((img, depth), axis=2)

    labels_path = img_path.replace('img', 'ann')
    labels = cv2.imread(labels_path)
    return rgbd, labels


def load_path(img_root):
    img_path_list = [os.path.join(img_root, filename) for filename in os.listdir(img_root)]
    return img_path_list


def main():
    args = get_parser()
    augment_classes = [int(i) for i in args.augment_classes.split(',')]
    save_path = args.save_path
    save_path = os.path.join(save_path, 'ins')
    for target_class in augment_classes:
        class_dir = os.path.join(save_path, label_to_names[target_class])
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
            print(f"mkdir {class_dir}")
    split = ['train', 'val']
    for s in split:
        img_root = os.path.join(args.data_root, 'img_dir', s)
        if not os.path.exists(img_root):
            raise ValueError
        img_path_list = load_path(img_root)
        for img_path in tqdm(img_path_list):
            rgbd, labels = load_rgbd(img_path)
            filename = os.path.splitext(os.path.basename(img_path))[0]
            for target_class in augment_classes:
                class_name = label_to_names[target_class]
                target_pixels = np.array((labels == target_class).nonzero())
                cluster_labels, _, n_clusters = mean_shift(target_pixels)
                print(f"{class_name}: n_clusters {n_clusters}")
                for j in range(n_clusters):
                    ins_pixels = target_pixels[cluster_labels == j]
                    ins_features = np.hstack(ins_pixels, rgbd[ins_pixels[:, 0], ins_pixels[:, 1]])
                    save_name = "{}_{:2d}_{:.4f}_{}.npy".format(filename, j, compute_variance(ins_pixels),
                                                                ins_features.shape[0])
                    print("    saving {}, variance: {:.4f}".format(save_name, compute_variance(ins_pixels)))
                    np.save(os.path.join(save_path, label_to_names[target_class], save_name), ins_pixels)


