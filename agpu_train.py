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
from torchpack import distributed as dist
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
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

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
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     datasets[split],
        #     num_replicas=dist.size(),
        #     rank=dist.rank(),
        #     shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=datasets[split].collate_fn,
            # prefetch_factor=4
        )

    return dataflow['train'], dataflow['val']


def load_checkpoint(config, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(config.log_dir, 'current.pth'))
    if epoch % config.save_freq == 0:
        torch.save(state, os.path.join(config.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, f'ckpt_epoch_{epoch}.pth')))


def main(config):
    train_loader, val_loader = get_loader(config)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    n_data = len(val_loader.dataset)
    logger.info(f"length of validation dataset: {n_data}")

    model, criterion = build_sensat_segmentation(config, train_loader.dataset.proportions)
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[dist.local_rank()],
    #                                                   find_unused_parameters=True
    #                                                   )
    criterion = criterion.cuda()

    config.base_learning_rate = config.base_learning_rate * config.batch_size / 8
    # config.base_learning_rate = config.base_learning_rate * dist.size() * config.batch_size / 8

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

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model, optimizer, scheduler)
        logger.info("==> checking loaded ckpt")
        validate('resume', val_loader, model, criterion, config, num_votes=2)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    # routine
    for epoch in range(config.start_epoch, config.epochs + 1):
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config)

        logger.info('epoch {}, total time {:.2f}, lr {:.5f}'.format(epoch,
                                                                    (time.time() - tic),
                                                                    optimizer.param_groups[0]['lr']))
        if epoch % config.val_freq == 0:
            validate(epoch, val_loader, model, criterion, config, num_votes=2)

        save_checkpoint(config, epoch, model, optimizer, scheduler)

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    validate('Last', val_loader, model, criterion, config, num_votes=20)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()
    logger.info(f'+-------------------------------------------------------------------+')
    logger.info(f'Train: [E {epoch}/{config.epochs}]')
    for idx, results in enumerate(train_loader):
        data_time.update(time.time() - end)
        print("step start========================================")
        points = results['lidar'].F
        batch_map = results['lidar'].C[:, 3]
        mask = results['mask'].F.unsqueeze(0)
        features = torch.hstack((results['rgb'].F, points)).unsqueeze(0)
        points_labels = results['targets'].F.unsqueeze(0)
        points = points.unsqueeze(0)
        cloud_idx = results['cloud_index']
        print("current clouds index: ", cloud_idx)
        search_tree = results['kdtree']
        bsz = results['lidar'].C[:, 3].max()+1

        # forward
        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)
        # features = features.transpose(2, 1).contiguous()
        # print(train_loader.dataset.files[cloud_idx])

        if config.knn_radius == 0:
            pred = model(points, mask, features.transpose(2, 1).contiguous())
            loss = criterion(pred, points_labels, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("batch time: ", time.time() - end)
            # update meters
            loss_meter.update(loss.item(), bsz)
        elif config.knn_radius > 0:
            print("start random sliding window---------------------------------")
            potential = np.zeros(points.shape[1], dtype=np.float32)
            idx = 1
            sub_end = time.time()
            while potential.min() == 0:
                idx += 1
                sub_points = torch.empty((0, config.knn_k, 3), dtype=torch.float32).cuda()
                sub_features = torch.empty((0, config.knn_k, config.in_features_dim), dtype=torch.float32).cuda()
                sub_mask = torch.empty((0, config.knn_k), dtype=torch.float32).cuda()
                sub_labels = torch.empty((0, config.knn_k), dtype=torch.float32).cuda()
                for ib in range(bsz):
                    print(points.shape)
                    print("====", batch_map.shape)
                    ib_points = points[0, batch_map == ib]
                    ib_potential = potential[batch_map == ib]
                    ib_features = features[0, batch_map == ib]
                    ib_labels = points_labels[0, batch_map == ib]
                    ib_search_tree = search_tree[ib]
                    ib_center = np.argmin(ib_potential)
                    ib_inds, dist = ib_search_tree.query_radius(ib_points[ib_center].cpu().numpy().reshape(1, -1),
                                                          return_distance=True,
                                                          r=config.knn_radius)
                    print("    choose {}/{}".format(len(ib_inds), config.knn_k))
                    ib_points = ib_points[ib_inds]
                    ib_features = ib_features[ib_inds]
                    ib_labels = ib_labels[ib_inds]
                    if len(ib_inds) < config.knn_k:
                        mask = np.zeros(config.knn_k, dtype=np.int32)
                        mask[:len(ib_inds)] = 1
                        padding = np.random.choice(len(ib_inds), config.knn_k - len(ib_inds), replace=True)
                        ib_inds = np.concatenate((ib_inds, padding))
                        ib_points = torch.concat((ib_points, ib_points[padding]))
                        ib_features = torch.concat((ib_features, ib_features[padding]))
                        ib_labels = torch.concat((ib_labels, ib_labels[padding]))
                    else:
                        mask = np.ones(config.knn_k, dtype=np.int32)
                        ib_inds = np.random.permutation(ib_inds)[:config.knn_k]
                        ib_points = ib_points[ib_inds]
                        ib_features = ib_features[ib_inds]
                        ib_labels = ib_labels[ib_inds]
                    sub_points = torch.concat((sub_points, ib_points.unsqueeze(0)))
                    sub_features = torch.concat((sub_features, ib_features.unsqueeze(0).transpose(2, 1).contiguous()))
                    sub_mask = torch.concat((sub_mask, torch.from_numpy(mask).unsqueeze(0)))
                    sub_labels = torch.concat((sub_labels, ib_labels.unsqueeze(0)))
                    potential[batch_map == ib][ib_inds] = np.square(1 - dist / np.square(config.knn_radius))
                sub_points = sub_points.cuda(non_blocking=True).contiguous()
                sub_features = sub_features.cuda(non_blocking=True).contiguous()
                sub_mask = sub_mask.cuda(non_blocking=True).contiguous()
                sub_labels = sub_labels.cuda(non_blocking=True).contiguous()
                pred = model(sub_points, sub_mask, sub_features)
                loss = criterion(pred, sub_labels, sub_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_meter.update(loss.item(), bsz)
                print(f"    rsw_idx{idx} sub_loss{loss.item()} time {time.time()-sub_end} ")
                sub_end = time.time()
            print("batch time: ", time.time() - end)
            print("end random sliding window-----------------------------------")
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if idx % config.print_freq == 0:
            logger.info(f'Train: [S {idx}/{len(train_loader)}]\t'
                        f'Batch Time {batch_time.val:.3f}\t'
                        f'Data Time {data_time.val:.3f}\t'
                        f'loss {loss_meter.val:.3f} (avg {loss_meter.avg:.3f})')
            # logger.info(f'[{cloud_label}]: {input_inds}')
    return loss_meter.avg


def validate(epoch, test_loader, model, criterion, config, num_votes=10):
    """
    One epoch validating
    """
    test_smooth = 0.95
    val_proportions = test_loader.dataset.proportions
    confusions = np.zeros((test_loader.dataset.num_classes, test_loader.dataset.num_classes), dtype=np.int32)
    sub_confusions = np.zeros((test_loader.dataset.num_classes, test_loader.dataset.num_classes), dtype=np.int32)
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        RT = d_utils.BatchPointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                                 z_range=config.z_angle_range)
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                                   std=config.noise_std, clip=config.noise_clip,
                                                   augment_symmetries=config.augment_symmetries)

        for current_step, results in enumerate(test_loader):
            points = results['lidar'].F
            mask = results['mask'].F
            features = torch.hstack((results['rgb'].F, points))
            inverse_map = results['inverse_map'].F
            batch_map = results['inverse_map'].C[:, 3]
            sub_batch_map = results['lidar'].C[:, 3]
            bsz = batch_map.max() + 1
            points_labels = results['targets'].F
            all_points_labels = results['targets_mapped'].F
            search_tree = results['kdtree']
            cloud_label = results['cloud_index']
            input_inds = results['input_inds']

            points = points.cuda(non_blocking=True).unsqueeze(0)
            mask = mask.cuda(non_blocking=True).unsqueeze(0)
            features = features.cuda(non_blocking=True).unsqueeze(0)
            points_labels = points_labels.cuda(non_blocking=True).unsqueeze(0)
            all_points_labels = all_points_labels.cuda(non_blocking=True)
            cloud_label = cloud_label.cuda(non_blocking=True)
            input_inds = input_inds.cuda(non_blocking=True)

            vote_logits = np.zeros((test_loader.dataset.num_classes, points_labels.shape[0]), dtype=np.float32)
            # augment for voting
            for v in range(num_votes):
                predictions = []
                targets = []
                if v > 0:
                    points = points.unsqueeze(0)
                    points = RT(points)
                    points = TS(points)
                    points = points.squeeze(0)
                    features = torch.concat(features[:, :3], points)

                # forward
                if config.knn_radius == 0:
                    pred = model(points, mask, features.transpose(2, 1).contiguous())
                else:
                    print("start random sliding window---------------------------------")
                    potential = np.zeros(points.shape[1], dtype=np.float32)
                    idx = 1
                    sub_end = time.time()
                    pred = np.zeros((points.shape[1], test_loader.dataset.num_classes))
                    while potential.min() == 0:
                        idx += 1
                        sub_points = torch.empty((0, config.knn_k, 3), dtype=torch.float32).cuda()
                        sub_features = torch.empty((0, config.knn_k, config.in_features_dim), dtype=torch.float32).cuda()
                        sub_mask = torch.empty((0, config.knn_k), dtype=torch.float32).cuda()
                        sub_labels = torch.empty((0, config.knn_k), dtype=torch.float32).cuda()
                        for ib in range(bsz):
                            print(points.shape)
                            print("====", batch_map.shape)
                            ib_points = points[0, sub_batch_map == ib]
                            ib_potential = potential[sub_batch_map == ib]
                            ib_features = features[0, sub_batch_map == ib]
                            ib_labels = points_labels[0, sub_batch_map == ib]
                            ib_all_labels = all_points_labels[batch_map == ib]
                            ib_search_tree = search_tree[ib]
                            ib_center = np.argmin(ib_potential)
                            ib_inds, dist = ib_search_tree.query_radius(ib_points[ib_center].cpu().numpy().reshape(1, -1),
                                                                  return_distance=True,
                                                                  r=config.knn_radius)
                            print("    choose {}/{}".format(len(ib_inds), config.knn_k))
                            ib_points = ib_points[ib_inds]
                            ib_features = ib_features[ib_inds]
                            ib_labels = ib_labels[ib_inds]
                            if len(ib_inds) < config.knn_k:
                                mask = np.zeros(config.knn_k, dtype=np.int32)
                                mask[:len(ib_inds)] = 1
                                padding = np.random.choice(len(ib_inds), config.knn_k - len(ib_inds), replace=True)
                                ib_inds = np.concatenate((ib_inds, padding))
                                ib_points = torch.concat((ib_points, ib_points[padding]))
                                ib_features = torch.concat((ib_features, ib_features[padding]))
                                ib_labels = torch.concat((ib_labels, ib_labels[padding]))
                            else:
                                mask = np.ones(config.knn_k, dtype=np.int32)
                                ib_inds = np.random.permutation(ib_inds)[:config.knn_k]
                                ib_points = ib_points[ib_inds]
                                ib_features = ib_features[ib_inds]
                                ib_labels = ib_labels[ib_inds]
                            sub_points = torch.concat((sub_points, ib_points.unsqueeze(0)))
                            sub_features = torch.concat((sub_features, ib_features.unsqueeze(0).transpose(2, 1).contiguous()))
                            sub_mask = torch.concat((sub_mask, torch.from_numpy(mask).unsqueeze(0)))
                            sub_labels = torch.concat((sub_labels, ib_labels.unsqueeze(0)))
                            potential[batch_map == ib][ib_inds] = np.square(1 - dist / np.square(config.knn_radius))
                        sub_points = sub_points.cuda(non_blocking=True).contiguous()
                        sub_features = sub_features.cuda(non_blocking=True).contiguous()
                        sub_mask = sub_mask.cuda(non_blocking=True).contiguous()
                        sub_labels = sub_labels.cuda(non_blocking=True).contiguous()
                        sub_pred = model(sub_points, sub_mask, sub_features)
                        for ib in range(bsz):
                            pred[sub_batch_map == ib] += sub_pred[ib]
                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), bsz)

                # collect
                pred = pred[:, mask]
                inds = input_inds[mask]
                vote_logits = test_smooth * vote_logits[:, inds] + (1 - test_smooth) * pred

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if current_step % config.print_freq == 0:
                    logger.info(
                        f'Test: [{current_step}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            IoUs, mIoU = SensatUrban_subset_metrics(test_loader.dataset, vote_logits, points_labels)

            logger.info(f'E{epoch} V{v} * sub_mIoU{mIoU:.3%}')
            logger.info(f'E{epoch} V{v} * sub_IoUs{IoUs}')

            proj_probs = vote_logits
            for ib in range(batch_map.max()+1):
                ib_proj = inverse_map[batch_map == ib]
                ib_logits = vote_logits[sub_batch_map == ib]
                ib_labels = points_labels[sub_batch_map == ib]
                sub_confusions += confusion_matrix(ib_labels, ib_logits, test_loader.dataset.num_classes)
                ib_logits = ib_logits[ib_proj]
                ib_labels = ib_labels[ib_proj]
                confusions += confusion_matrix(ib_labels, ib_logits, test_loader.dataset.num_classes)
        sub_IoUs, sub_mIoU = IoU_from_confusions(sub_confusions)

        logger.info(f'E{epoch} V{num_votes} * sub_mIoU {sub_mIoU:.3%}')
        logger.info(f'E{epoch} V{num_votes}  * sub_IoUs {sub_IoUs}')

        IoUs, mIoU = IoU_from_confusions(confusions)

        logger.info(f'E{epoch} V{num_votes} * mIoU {mIoU:.3%}')
        logger.info(f'E{epoch} V{num_votes}  * IoUs {IoUs}')
    return mIoU


if __name__ == "__main__":
    # dist.init()
    # torch.cuda.set_device(dist.local_rank())

    opt, config = parse_option()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir

    logger = setup_logger(output=config.log_dir, name="SensatUrban")
    path = os.path.join(config.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system('cp %s %s' % (opt.cfg, config.log_dir))
    logger.info("Full config saved to {}".format(path))
    main(config)
