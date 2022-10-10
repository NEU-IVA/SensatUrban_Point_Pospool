import os.path

import numpy as np
import cv2
import argparse
from sklearn.cluster import MeanShift, estimate_bandwidth
from insaug.utils import mean_shift, compute_variance


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
    img = np.array(img)
    depth = np.load(depth_path)
    rgbd = np.concatenate((img, depth), axis=2)
    return rgbd


def load_path(img_root):
    img_path_list = [os.path.join(img_root, filename) for filename in os.listdir(img_root)]

