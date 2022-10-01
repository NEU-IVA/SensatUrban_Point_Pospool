from typing import Callable

import numpy as np
import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Optimizer, Scheduler
import torchpack.distributed as dist

from core.dataset.sensatUrban_BEV import SensatUrban_BEV
from core.dataset.sensatUrban_crop import SensatUrban_crop
from core.dataset.sensatUrban_downsample import SensatUrban_reduced
from core.dataset.sensatUrban_nocolor import SensatUrban_nocolor
from core.dataset.sensatUrban_yes import SensatUrban_yes
from core.models.PVBNet_fusion import PVBNet_fusion
from core.experiments.PVNet import PVNet
from core.experiments.PVNet_fusion2 import PVNet_fusion2


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    batch_size *= dist.size()

    if dist.size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // dist.size()

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * ratio))


# def make_dataset():
#     return {
#         'train': sensatUrban(root=configs.dataset.root, num_points=configs.dataset.num_points,
#                              voxel_size=configs.dataset.voxel_size, split="train"),
#         'test': sensatUrban(root=configs.dataset.root, num_points=configs.dataset.num_points,
#                             voxel_size=configs.dataset.voxel_size, split="val"),
#     }

def make_dataset():
    if configs.dataset.name == 'reduced':
        model = SensatUrban_reduced(num_points=configs.dataset.num_points, voxel_size=configs.train.voxel_size)
    elif configs.dataset.name == 'yes':
        model = SensatUrban_yes(num_points=configs.dataset.num_points, voxel_size=configs.train.voxel_size)
    elif configs.dataset.name == 'bev':
        model = SensatUrban_BEV(num_points=configs.dataset.num_points, voxel_size=configs.train.voxel_size)
    elif configs.dataset.name == 'nocolor':
        model = SensatUrban_nocolor(num_points=configs.dataset.num_points, voxel_size=configs.train.voxel_size)
    elif configs.dataset.name == 'crop':
        model = SensatUrban_crop(num_points=configs.dataset.num_points, voxel_size=configs.train.pres,
                                 dataset_root=configs.dataset.root)
    else:
        raise NotImplementedError(configs.dataset.name)
    return model


def make_model():
    if configs.model.name == 'pvnet_fusion2':
        model = PVNet_fusion2(num_classes=configs.data.num_classes,
                              cr=1.0,
                              pres=configs.dataset.voxel_size,
                              vres=configs.dataset.voxel_size)
    elif configs.model.name == 'pvnet':
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = PVNet(num_classes=configs.data.num_classes,
                      cr=cr,
                      pres=configs.dataset.voxel_size,
                      vres=configs.dataset.voxel_size)
    elif configs.model.name == 'pvnet_fusion':
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = PVNet_fusion(num_classes=configs.data.num_classes,
                             cr=cr,
                             pres=configs.dataset.voxel_size,
                             vres=configs.dataset.voxel_size)
    elif configs.model.name == 'pvbnet_fusion':
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = PVBNet_fusion(num_classes=configs.data.num_classes,
                              cr=cr,
                              pres=configs.dataset.voxel_size,
                              vres=configs.dataset.voxel_size)
    elif configs.model.name == 'pvbnet_attention':
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = PVBNet_fusion(num_classes=configs.data.num_classes,
                              cr=cr,
                              pres=configs.dataset.voxel_size,
                              vres=configs.dataset.voxel_size)
    elif configs.model.name == 'spvcnn':
        from core.models.SPVCNN import SPVCNN
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    elif configs.model.name == 'spvcnn_add512layer':
        from core.models.SPVCNN_add512layer import SPVCNN_add512layer
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN_add512layer(num_classes=configs.data.num_classes,
                                   cr=cr,
                                   pres=configs.dataset.voxel_size,
                                   vres=configs.dataset.voxel_size)
    elif configs.model.name == 'spvcnn_orignal':
        from core.models.SPVCNN_orignal import SPVCNN_orignal
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN_orignal(num_classes=configs.data.num_classes,
                               cr=cr,
                               pres=configs.dataset.voxel_size,
                               vres=configs.dataset.voxel_size)
    elif configs.model.name == 'spvcnn_nopoint':
        from core.models.SPVCNN_nopoint import SPVCNN_nopoint
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN_nopoint(num_classes=configs.data.num_classes,
                               cr=cr,
                               pres=configs.dataset.voxel_size,
                               vres=configs.dataset.voxel_size)
    elif configs.model.name == 'spvcnn_add512andfusion':
        from core.models.SPVCNN_add512andfusion import SPVCNN_add512andfusion
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        model = SPVCNN_add512andfusion(num_classes=configs.data.num_classes,
                                       cr=cr,
                                       pres=configs.dataset.voxel_size,
                                       vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.optimizer.lr,
                                    momentum=configs.optimizer.momentum,
                                    weight_decay=configs.optimizer.weight_decay,
                                    nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.train.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from functools import partial

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=configs.train.num_epochs,
                              batch_size=configs.train.batch_size,
                              dataset_size=configs.data.training_size * configs.dataset.num_crop))
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
