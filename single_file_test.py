import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import sys
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import core.modules.pospool.utils.data_utils as d_utils
from core.modules.pospool.build import build_sensat_segmentation
from core.dataset.SensatUrban_ultra import SensatUrban_ultra
from core.modules.pospool.utils.util import AverageMeter, SensatUrban_metrics, SensatUrban_subset_metrics, \
    SensatUrban_voting_metrics, IoU_from_confusions
from core.modules.pospool.utils.lr_scheduler import get_scheduler
from core.modules.pospool.utils.logger import setup_logger
from core.modules.pospool.utils.config import config, update_config


def parse_option():
    parser = argparse.ArgumentParser('SensatUrban scene-segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    # parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')
    parser.add_argument('--knn_radius', type=int, default=0, help='knn radius, predict with sliding window, 0 means not use')

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='test_log', help='log dir [default: log]')

    # misc
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    # config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed
    config.knn_radius = args.knn_radius

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'SensatUrban', f'{ddir_name}_{int(time.time())}')

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    print(args)
    print(config)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def get_loader(config):
    # set the data loader
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range),
        d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries),
    ])

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
    ])

    datasets = SensatUrban_ultra(num_points=config.num_points, voxel_size=config.voxel_size,
                                 dataset_root=config.data_root, train_transform=train_transforms,
                                 test_trainsform=test_transforms, bev_size=config.bev_size,
                                 bev_name=config.bev_name)
    dataflow = {}

    for split in datasets:
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=datasets[split].collate_fn, prefetch_factor=4)

    return dataflow['train'], dataflow['val'], datasets


def main():
    args, config = parse_option()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir
    logger = setup_logger(output=config.log_dir, name="SensatUrban")

    *_, datasets = get_loader(config)
    model, criterion = build_sensat_segmentation(config, datasets['train'].proportions)

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.base_learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.base_learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported")

    err_results = datasets['train'][558:561]
    for idx, result in enumerate(err_results):
        end = time.time()
        print("****************INDEX ", idx+558)
        points = result['lidar'].F
        batch_map = result['lidar'].C[:, 3]
        mask = result['mask'].F.unsqueeze(0)
        features = torch.hstack((result['rgb'].F, points)).unsqueeze(0)
        points_labels = result['targets'].F.unsqueeze(0)
        points = points.unsqueeze(0)
        cloud_idx = result['cloud_idx'].F.unsqueeze(0)
        print("current clouds index: ", cloud_idx)
        search_tree = result['kdtree']
        bsz = result['lidar'].C[:, 3].max() + 1

        # forward
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)
        # features = features.transpose(2, 1).contiguous()

        pred = model(points, mask, features.transpose(2, 1).contiguous())
        loss = criterion(pred, points_labels, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("batch time: ", time.time() - end)

