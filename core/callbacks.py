from typing import Any, Dict, Union

import numpy as np
import torch
import wandb
from torchpack import distributed as dist
from torchpack.callbacks import SummaryWriter
from torchpack.callbacks.callback import Callback

__all__ = ['MeanIoU']

from torchpack.utils.typing import Trainer


class WandbWriter(SummaryWriter):
    '''
    Upload sumaries to Wandb server
    '''

    def __init__(self):
        pass

    def _set_trainer(self, trainer: Trainer) -> None:
        import wandb
        self.writer = wandb

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.writer.log({name: scalar})

    def _add_image(self, name: str, tensor: np.ndarray) -> None:
        self.writer.log({name: wandb.Image(tensor)})

    def _add_pointcloud(self, name: str, tensor: np.ndarray) -> None:
        self.writer.log({name: wandb.Object3D(tensor)})

    def _after_train(self) -> None:
        pass


class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        # self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]

        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum((targets == i)
                                               & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(outputs == i).item()

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)

        iou_data = [[i, ious[i]] for i in range(len(ious))]
        table = wandb.Table(data=iou_data, columns=['label', 'value'])
        wandb.log({
            "miou": miou,
            "iou": wandb.plot.bar(table, 'label', 'value', title="ious")
        })

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            print(ious)
            print(miou)
