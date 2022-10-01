import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from core.modules.pospool.utils.config import config, update_config
from core.modules.pospool.backbones.resnet import ResNet
from core.modules.pospool.heads import ClassifierResNet, MultiPartSegHeadResNet, SceneSegHeadResNet
from core.modules.pospool.losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy, \
    MaskedWeightCrossEntropy


def build_classification(config):
    model = ClassificationModel(config,
                                config.backbone, config.head, config.num_classes, config.input_features_dim,
                                config.radius, config.sampleDl, config.nsamples, config.npoints,
                                config.width, config.depth, config.bottleneck_ratio)
    criterion = LabelSmoothingCrossEntropyLoss()
    return model, criterion


def build_multi_part_segmentation(config):
    model = MultiPartSegmentationModel(config, config.backbone, config.head, config.num_classes, config.num_parts,
                                       config.input_features_dim,
                                       config.radius, config.sampleDl, config.nsamples, config.npoints,
                                       config.width, config.depth, config.bottleneck_ratio)
    criterion = MultiShapeCrossEntropy(config.num_classes)
    return model, criterion


def build_scene_segmentation(config):
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    criterion = MaskedCrossEntropy()
    return model, criterion

def build_sensat_segmentation(config, proportions):
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    if config.loss_weight:
        criterion = MaskedWeightCrossEntropy(proportions)
    else:
        criterion = MaskedCrossEntropy()
    return model, criterion

class ClassificationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(ClassificationModel, self).__init__()

        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Classification Model")

        if head == 'resnet_cls':
            self.classifier = ClassifierResNet(num_classes, width)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Classification Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.classifier(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class MultiPartSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, num_parts,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(MultiPartSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_part_seg':
            self.segmentation_head = MultiPartSegHeadResNet(num_classes, width, radius, nsamples, num_parts)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class SceneSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(SceneSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_scene_seg':
            self.segmentation_head = SceneSegHeadResNet(num_classes, width, radius, nsamples)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def parse_option():
    parser = argparse.ArgumentParser('S3DIS scene-segmentation training')
    parser.add_argument('--cfg', type=str, default="/home/zzh/projects/sensatUrban/configs/pospool.yaml",
                        required=False,
                        help='config file')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed

    config.local_rank = args.local_rank

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 's3dis', f'{ddir_name}_{int(time.time())}')

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


if __name__ == '__main__':
    opt, config = parse_option()
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)

    print("net have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
