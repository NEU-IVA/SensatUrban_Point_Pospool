import traceback
from typing import Any, Callable, Dict

import numpy as np
import torch
import wandb
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['SensatUrbanTrainer']


class SensatUrbanTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 wandb=False,
                 amp_enabled: bool = False) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb = wandb
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key:
                _inputs[key] = value.cuda()

        # SparseTensor(唯一点, 唯一体素)
        inputs = {'lidar': _inputs['lidar'], 'bev': _inputs['bev'], 'alt': _inputs['alt'],
                  'pc_num': feed_dict['pc_num'].numpy().tolist()}
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        # print(inputs['pc_num'], inputs['bev'].shape, inputs['alt'].shape, feed_dict['file_name'])
        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(inputs)

            if outputs.requires_grad:
                loss = self.criterion(outputs, targets)

        if outputs.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.wandb:
                if self.global_step % 100 == 0 or self.global_step == 1:
                    histograms = {}
                    for tag, value in self.model.named_parameters():
                        if 'resnet' in tag:
                            continue
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if value.grad != None:
                            try:
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            except Exception as e:
                                print(tag)
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": self.epoch_num,
                        **histograms
                    })
                else:
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": self.epoch_num
                    })

            self.scheduler.step()
        else:
            # loss = self.criterion(outputs, targets)
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (inputs['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)

                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)

            if self.wandb:
                if self.global_step % 100 == 0:
                    wandb.log({
                        "val_loss": loss.item(),
                        "epoch": self.epoch_num
                    })
                else:
                    wandb.log({
                        "val_loss": loss.item(),
                        "epoch": self.epoch_num
                    })

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
