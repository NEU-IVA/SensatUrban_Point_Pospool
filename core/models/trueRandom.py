import glob
import time

import matplotlib.pyplot as plt

from core.utils.image_tools import complete2d
from core.utils.tool import DataProcessing as DP
from helper_ply import read_ply, write_ply
import numpy as np
import torch
import torch.utils.data
import os
from torch.utils.data import DataLoader
from os.path import join
import pickle
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.config import configs as cfg
from torchpack import distributed as dist
from torchpack.utils.config import configs


class Sensat_truerandom(dict):

    def __init__(self, voxel_size, num_points):
        super().__init__({
            'train': SensatUrban(split='train', voxel_size=voxel_size,
                                 num_points=num_points),
            'test': SensatUrban(split='val', voxel_size=voxel_size,
                                num_points=num_points)
        })


class SensatUrban():
    def __init__(self, split, voxel_size, num_points):

        self.num_points = num_points
        self.mode = split
        self.dataset_path = "/home/zzh/datasets/data_release/"
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
        self.loaded_idx = None
        self.lock = False

    def __getitem__(self, item):
        # print("item:", item)
        split = self.mode
        if split == 'train':
            file_names = self.train_file_name
        elif split == 'val':
            file_names = self.val_file_name
        else:
            pass
        np.random.shuffle(file_names)
        cloud_idx = item // cfg.dataset.num_crop

        if self.loaded_idx == None or cloud_idx != self.loaded_idx:
            while self.lock:
                print("wait lock")
            self.lock = True
            t0 = time.time()
            ply_file = join(self.dataset_path, split, '{:s}.ply'.format(file_names[cloud_idx]))
            data = read_ply(ply_file)
            self.points = np.vstack((data['x'], data['y'], data['z'])).T
            self.sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            self.sub_labels = data['class']
            size = self.sub_colors.shape[0] * 4 * 7
            self.points_num = self.points.shape[0]
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(ply_file.split('/')[-1], size * 1e-6, time.time() - t0))
            self.loaded_idx = cloud_idx
            self.lock = False
        else:
            pass

        while self.lock:
            print("wait lock")
        self.lock = True
        points = self.points
        sub_colors = self.sub_colors
        sub_labels = self.sub_labels
        points_num = self.points_num

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
        queried_pc_colors = sub_colors[queried_idx] / 255.0
        queried_pc_labels = sub_labels[queried_idx]

        # print("yeild", cloud_idx)
        if self.mode == 'train':
            # xyzs, rgbs, labels, query_idxs, bev, cloud_idx = self.spatially_regular_gen()
            xyzs = queried_pc_xyz.astype(np.float32)
            rgbs = queried_pc_colors.astype(np.float32)
            labels = queried_pc_labels
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])
            xyzs = np.dot(xyzs, rot_mat) * scale_factor
            xyzs = np.dot(xyzs, rot_mat)
        elif self.mode == 'val':
            xyzs = queried_pc_xyz.astype(np.float32)
            rgbs = queried_pc_colors.astype(np.float32)
            labels = queried_pc_labels
        else:
            pass

        voxels = np.round(xyzs / self.voxel_size).astype(np.int32)
        voxels -= voxels.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(voxels, return_index=True, return_inverse=True)

        if 'train' == self.mode:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)
        labels_selected = labels[inds].astype(np.int64)
        xyz_selected = xyzs[inds].astype(np.float32)
        voxel_selected = voxels[inds]
        bev = self.to_BEV(np.hstack([queried_pc_xyz, queried_pc_colors])).transpose(2, 1, 0)
        pc_num = np.array(xyz_selected.shape[0]).astype(np.int32)
        self.lock = False

        lidar_ST = SparseTensor(xyz_selected, voxel_selected)
        labels_ST = SparseTensor(labels_selected, voxel_selected)
        all_labels_ST = SparseTensor(labels, voxels)
        inverse_map = SparseTensor(inverse_map, voxels)

        return {
            'lidar': lidar_ST,
            'targets': labels_ST,
            'bev': bev,
            'targets_mapped': all_labels_ST,
            'inverse_map': inverse_map,
            'file_name': self.all_files[cloud_idx][0],
            'pc_num': pc_num
        }

    def __len__(self):
        if self.mode == 'train':
            return 4 * configs.workers_per_gpu * configs.dataset.num_crop
        elif self.mode == 'val':
            return 4 * configs.workers_per_gpu * configs.dataset.num_crop

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

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
            bev[xs[i], ys[i], 2] = grid_data[i, 5]  # R
            bev[xs[i], ys[i], 1] = grid_data[i, 4]  # G
            bev[xs[i], ys[i], 0] = grid_data[i, 3]  # B
        rgbd = bev[:, :, :]
        # for c in range(3): rgbd[:, :, c] = complete2d(rgbd[:, :, c], 3)
        return rgbd


if __name__ == '__main__':
    cfg.load("../../configs/default.yaml", recursive=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    datasets = Sensat_truerandom(num_points=configs.dataset.num_points, voxel_size=configs.dataset.voxel_size)
    dataflow = {}
    for split in datasets:
        sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=False)
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=1,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=datasets[split].collate_fn,
            prefetch_factor=4)

    # for i in range(1000):
    #     datasets['train'].__getitem__(i)
    #     print(i)
    i = 0
    for batch_idx, data in enumerate(dataflow['train']):
        # print(i)
        i = i + 1
        # exit(0)
