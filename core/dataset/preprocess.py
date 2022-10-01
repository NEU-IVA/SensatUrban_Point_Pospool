import argparse
import glob
import threading
import time

from matplotlib import pyplot as plt

from core.utils.image_tools import complete2d
from helper_ply import read_ply, write_ply
import numpy as np
import os
from os.path import join
from torchpack.utils.config import configs as cfg
import cv2


class DatasetGenerator():
    def __init__(self, split, voxel_size, num_points):
        self.num_points = num_points
        self.mode = split
        self.dataset_path = "/home/zzh/datasets/data_release/"
        self.out_path = "/home/zzh/datasets/sensaturban/"
        self.voxel_size = voxel_size
        self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                               4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                               9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.bev_size = cfg.dataset.bev_size
        self.all_files = np.sort(glob.glob(join(self.dataset_path, 'train', '*.ply')))

        self.val_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']
        self.test_file_name = ['birmingham_block_2', 'birmingham_block_8',
                               'cambridge_block_15', 'cambridge_block_22',
                               'cambridge_block_16', 'cambridge_block_27']

        self.train_file_name = [i[38:-4] for i in self.all_files if i[38:-4] not in self.val_file_name]
        self.use_val = True  # whether use validation set or not

        self.num_per_class = np.zeros(self.num_classes)

        split = self.mode
        if split == 'train':
            self.file_names = self.train_file_name
        elif split == 'val':
            self.file_names = self.val_file_name
        else:
            pass

        # np.random.shuffle(self.file_names)
        self.data_path = os.path.join(self.out_path, split)
        self.ply_path = os.path.join(self.data_path, "ply")
        self.bev_path = os.path.join(self.data_path, "bev")
        self.alt_path = os.path.join(self.data_path, "alt")
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.ply_path, exist_ok=True)
        os.makedirs(self.bev_path, exist_ok=True)
        os.makedirs(self.alt_path, exist_ok=True)

    def genarate(self, index):

        cloud_idx = index
        t0 = time.time()
        ply_file = join(self.dataset_path, self.mode, '{:s}.ply'.format(self.file_names[cloud_idx]))
        data = read_ply(ply_file)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
        sub_labels = data['class']
        size = sub_colors.shape[0] * 4 * 7
        points_num = points.shape[0]
        # print('{:s} {:.1f} MB loaded in {:.1f}s'.format(ply_file.split('/')[-1], size * 1e-6, time.time() - t0))

        num_crop = int(size / 1000000)
        print("will crop ", num_crop)
        for i in range(num_crop):
            point_ind = np.random.randint(points_num)
            center_point = points[point_ind, :].reshape(1, -1)
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)
            queried_idx = np.where(
                (points[:, 0] > pick_point[0, 0] - self.bev_size) & (
                        points[:, 1] > pick_point[0, 1] - self.bev_size) &
                (points[:, 0] < pick_point[0, 0] + self.bev_size) & (
                        points[:, 1] < pick_point[0, 1] + self.bev_size)
            )
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = sub_colors[queried_idx]
            queried_pc_labels = sub_labels[queried_idx]
            bev, alt = self.to_BEV(np.hstack([queried_pc_xyz, queried_pc_colors]))
            # plt.imshow(bev)
            # plt.show()
            # exit()
            # print(queried_pc_labels.shape)
            # print(queried_pc_xyz.shape)
            cv2.imwrite(os.path.join(self.bev_path, self.file_names[cloud_idx] + "_" + str(i) + ".png"), bev)
            cv2.imwrite(os.path.join(self.alt_path, self.file_names[cloud_idx] + "_" + str(i) + ".png"), alt)

            write_ply(os.path.join(self.ply_path, self.file_names[cloud_idx] + "_" + str(i)),
                      np.hstack([queried_pc_xyz, queried_pc_labels.reshape((queried_pc_labels.shape[0], 1))]),
                      ['x', 'y', 'z', 'c'])

    def to_BEV(self, grid_data):
        gird_size = cfg.data.bev_size
        grid_size_scale = cfg.data.bev_scale  # size of projection0
        num = grid_data.shape[0]
        bev = np.zeros((gird_size, gird_size, 4))  # (grid_scale, grid_scale, 1)
        off = int(gird_size / 2)
        xs = (grid_data[:, 0] / grid_size_scale).astype(np.int32) + off
        ys = (grid_data[:, 1] / grid_size_scale).astype(np.int32) + off
        for i in range(num):
            # update points
            bev[xs[i], ys[i], 3] = grid_data[i, 2]
            # update RGB:
            bev[xs[i], ys[i], 2] = grid_data[i, 3]  # R
            bev[xs[i], ys[i], 1] = grid_data[i, 4]  # G
            bev[xs[i], ys[i], 0] = grid_data[i, 5]  # B
        rgb = bev[:, :, :3]
        alt = bev[:, :, 3]

        alt = complete2d(alt, 3)
        for c in range(3): rgb[:, :, c] = complete2d(rgb[:, :, c], 3)
        return rgb.astype(np.int32), alt


# class thread(threading.Thread):
#     def __init__(self, dataset, id):
#         super(thread, self).__init__()
#         self.dataset = dataset
#         self.id = id
#
#     def run(self):
#         self.dataset.genarate(self.id)
#         print("done ", self.id)


if __name__ == '__main__':
    cfg.load("../../configs/default.yaml", recursive=True)
    datasets = DatasetGenerator(split='train', num_points=cfg.dataset.num_points, voxel_size=cfg.dataset.voxel_size)
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int)
    args, opts = parser.parse_known_args()
    datasets.genarate(args.id)
    # ts = [thread(datasets, i) for i in range(33)]
    #
    # for i in ts:
    #     i.start()
