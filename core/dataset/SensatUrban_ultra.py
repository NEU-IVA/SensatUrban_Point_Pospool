import glob
import pickle

import cv2
import matplotlib.pyplot as plt
from helper_ply import read_ply, write_ply
import numpy as np
import torch
import torch.utils.data
import os
import pathlib
from tqdm import tqdm
from os.path import join
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.config import configs as cfg
from torchpack import distributed as dist
from torchpack.utils.config import configs
from sklearn.neighbors.kd_tree import KDTree


class SensatUrban_ultra(dict):

    def __init__(self, voxel_size, num_points, dataset_root, train_transform, test_trainsform, bev_size, bev_name='rgb'):
        super().__init__({
            'train': SensatUrban(split='train', voxel_size=voxel_size,
                                 num_points=num_points, dataset_root=dataset_root, bev_size=bev_size, transform=train_transform, bev_name=bev_name),
            'val': SensatUrban(split='val', voxel_size=voxel_size,
                                num_points=num_points, dataset_root=dataset_root, bev_size=bev_size, transform=test_trainsform, bev_name=bev_name)
        })


class SensatUrban(torch.utils.data.Dataset):

    def __init__(self, split, voxel_size, num_points, dataset_root, bev_size, transform=None, bev_name='rgb'):
        super(SensatUrban, self).__init__()
        self.num_points = num_points
        self.mode = split
        self.dataset_path = dataset_root
        self.voxel_size = voxel_size
        self.label_to_names = {0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
                               4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
                               9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.transform = transform
        self.bev_name = bev_name

        self.bev_size = bev_size
        self.train_files = np.sort(glob.glob(join(self.dataset_path, 'train', 'ply', '*.ply')))

        self.val_files = np.sort(glob.glob(join(self.dataset_path, 'val', 'ply', '*.ply')))

        self.test_files = ['birmingham_block_2', 'birmingham_block_8',
                           'cambridge_block_15', 'cambridge_block_22',
                           'cambridge_block_16', 'cambridge_block_27']

        if self.mode == 'train':
            self.files = self.train_files
        elif self.mode == 'val':
            self.files = self.val_files
        else:
            self.files = self.test_files  # TODO: update test data
            raise NotImplementedError

        self.gen_kdtree_pkl(self.files)
        # self.gen_kdtree_pkl(self.test_files)

        proportions_file = os.path.join(self.dataset_path, "{}_proportions.npy".format(self.mode))
        if not os.path.exists(proportions_file):
            self.proportions = self.generate_proportion(self.files)
            np.save(proportions_file, self.proportions)
            print("saving {} proportions".format(self.mode))
        else:
            self.proportions = np.load(proportions_file)

        self.use_val = True

        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def __getitem__(self, index):
        if self.mode == 'train':
            xyzs, labels, bev, alt, rgb, kdtree = self.spatially_regular_gen(self.train_files[index], True)

            # 方向矫正
            # theta = 1.5 * np.pi
            # rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
            #                     [-np.sin(theta),
            #                      np.cos(theta), 0], [0, 0, 1]])
            # xyzs = np.dot(xyzs, rot_mat)
        elif self.mode == 'val':
            xyzs, labels, bev, alt, rgb, kdtree = self.spatially_regular_gen(self.val_files[index], True)
        else:
            xyzs, labels, bev, alt, rgb, kdtree = self.spatially_regular_gen(self.val_files[index], True)
        
        voxels = np.round(xyzs / self.voxel_size).astype(np.int32)
        voxels -= voxels.min(0, keepdims=1)
        
        xyzs = xyzs.astype(np.float32)
        if self.transform is not None:
            xyzs = self.transform(xyzs)

        _, inds, inverse_map = sparse_quantize(voxels, voxel_size=self.voxel_size, return_index=True,
                                               return_inverse=True)

        mask = torch.ones(len(inds)).type(torch.int32)
        # if 'train' == self.mode:
        #     if len(inds) >= self.num_points:
        #         mask = torch.ones(self.num_points).type(torch.int32)
        #         inds = np.random.choice(inds, self.num_points, replace=False)
            # else:
            #     mask = torch.zeros(self.num_points).type(torch.int32)
            #     mask[:len(inds)] = 1
            #     padding_choice = np.random.choice(len(inds), self.num_points - len(inds))
            #     inds = np.hstack([inds, inds[padding_choice]])

        if len(inds) < 100:
            return self.__getitem__(index - 1)

        labels_selected = labels[inds].astype(np.int64)
        xyz_selected = xyzs[inds, :]
        voxel_selected = voxels[inds]
        rgb_selected = rgb[inds]

        lidar_ST = SparseTensor(xyz_selected, voxel_selected)
        labels_ST = SparseTensor(labels_selected, voxel_selected)
        rgb_ST = SparseTensor(rgb_selected, voxel_selected)
        all_labels_ST = SparseTensor(labels, voxels)
        inverse_map_ST = SparseTensor(inverse_map, voxels)
        inds_ST = SparseTensor(inds, voxels)
        mask_ST = SparseTensor(mask, voxels)

        pc_num = np.array(xyz_selected.shape[0]).astype(np.int32)
        # alt = alt.transpose(2, 1, 0)
        return {
            'lidar': lidar_ST,
            'targets': labels_ST,
            'rgb': rgb_ST,
            'bev': bev,
            'alt': alt,
            'targets_mapped': all_labels_ST,
            'inverse_map': inverse_map_ST,
            'pc_num': pc_num,
            'file_name': self.train_files[index],
            'mask': mask_ST,
            'cloud_index': index,
            'input_inds': inds_ST,
            'kdtree': kdtree
        }

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_files)
        if self.mode == 'val':
            return len(self.val_files)
        if self.mode == 'test':
            return len(self.test_files)

    def spatially_regular_gen(self, path, return_kdtree=False):
        filename = os.path.basename(path)[:-4]
        root_path = os.path.join(self.dataset_path, self.mode)
        bev_path = os.path.join(root_path, self.bev_name, filename + '.png')
        alt_path = os.path.join(root_path, 'alt', filename + '.npy')
        ply_path = path

        if return_kdtree:
            kdtree_path = os.path.join(root_path, 'kdt', filename + '.pkl')
            with open(kdtree_path, 'rb') as f:
                kdtree = pickle.load(f)

        ply_data = read_ply(ply_path)
        xyz = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'])).T
        rgb = np.vstack((ply_data['r'], ply_data['g'], ply_data['b'])).T / 255.0
        if self.mode == 'test':
            labels = np.array([])
        else:
            labels = ply_data['c']

        bev = cv2.imread(bev_path)
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
        alt = np.load(alt_path)
        alt = np.expand_dims(alt, axis=2)

        if return_kdtree:
            return xyz.astype(np.float32), labels.astype(np.uint8), bev.astype(np.uint8), alt.astype(np.float32), rgb.astype(np.float32), kdtree
        return xyz.astype(np.float32), labels.astype(np.uint8), bev.astype(np.uint8), alt.astype(np.float32), rgb.astype(np.float32)

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def generate_val_files(self, all_files, val_file_name):
        val_files = []
        for i in all_files:
            fileneame = os.path.basename(i)
            for j in val_file_name:
                if fileneame[:-4] == j:
                    val_files.append(i)

        train_files = [i for i in all_files if i not in val_files]
        return train_files, val_files

    def generate_proportion(self, files):
        proportions = np.zeros(self.num_classes, dtype=np.int32)
        print("generating proportions (be used to calculate the loss weight)")
        for filepath in tqdm(files):
            _, labels, *_ = self.spatially_regular_gen(filepath)
            for label in range(self.num_classes):
                proportions[label] += np.sum(labels == label)
        return proportions

    @staticmethod
    def gen_kdtree_pkl(files):
        kdtree_dir = os.path.split(files[0])[0].replace('ply', 'kdt')
        if not os.path.exists(kdtree_dir):
            os.mkdir(kdtree_dir)
            print("saving kdtree files in data root")
        else:
            print("loading kdtree files")
        for file in tqdm(files):
            kdtree_path = os.path.join(os.path.split(file)[0].replace('ply', 'kdt'), os.path.basename(file)[:-4] + ".pkl")
            if not os.path.exists(kdtree_path):
                ply_data = read_ply(file)
                xyz = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'])).T
                search_tree = KDTree(xyz)
                with open(kdtree_path, 'wb') as f:
                    pickle.dump(search_tree, f)


def to_BEV(grid_data):
    gird_size = 500
    grid_size_scale = 0.05  # size of projection0
    num = grid_data[0].shape[0]
    bev = np.zeros((gird_size, gird_size, 4))  # (grid_scale, grid_scale, 1)
    off = 0
    ys = (grid_data[0][:, 0] / grid_size_scale).astype(np.int32) + off
    xs = gird_size - ((grid_data[0][:, 1] / grid_size_scale).astype(np.int32) + off) - 1
    for i in range(num):
        # update points
        bev[xs[i], ys[i], 3] = grid_data[0][i, 2]
        # update RGB:
        bev[xs[i], ys[i], 0] = grid_data[1][i, 0]  # R
        bev[xs[i], ys[i], 1] = grid_data[1][i, 1]  # G
        bev[xs[i], ys[i], 2] = grid_data[1][i, 2]  # B
    rgb = bev[:, :, :3]
    alt = bev[:, :, 3]

    return rgb.astype(np.uint8), alt.astype(np.float32)


if __name__ == '__main__':
    cfg.load("../../configs/pospool.yaml", recursive=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    datasets = SensatUrban_ultra(num_points=configs.num_points, voxel_size=0.1,
                                 dataset_root=configs.root)
    dataflow = {}

    for split in datasets:
        sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=1,
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
            collate_fn=datasets[split].collate_fn, prefetch_factor=4)

    for batch_idx, data in enumerate(dataflow['train']):
        bev = data['bev']
        bev = bev.squeeze()
        xyz = data['lidar'].F
        alt = data['alt']

        if xyz.shape[0] < 100:
            print(xyz.shape, "error")
            print(data["file_name"])
            continue
        else:
            continue

        off = 0
        bind_idx = torch.floor(xyz[:, 0:2] / cfg.bev_scale).type(torch.int64) + off
        bind_idx = bind_idx[:, 0] * cfg.bev_size + cfg.bev_size - 1 - bind_idx[:, 1]

        bev = bev.permute(1, 0, 2)
        bev_ = torch.reshape(bev, (-1, 3))
        rgbs = torch.index_select(bev_, 0, bind_idx)

        project_bev, project_alt = to_BEV([xyz.cpu().numpy(), rgbs.cpu().numpy()])

        # write_ply(str(batch_idx) + "_new.ply", [xyz.cpu().numpy(), rgbs.cpu().numpy()], ['x', 'y', 'z', 'r', 'g', 'b'])

        plt.imshow(data['bev'][0, :, :, :].cpu().numpy())
        plt.title("orignal bev")
        plt.show()

        plt.imshow(data['alt'][0, :, :, :].cpu().numpy())
        plt.title("orignal alt")
        plt.show()

        plt.imshow(project_bev)
        plt.title("project_bev")
        plt.show()

        if batch_idx == 0:
            exit(0)
