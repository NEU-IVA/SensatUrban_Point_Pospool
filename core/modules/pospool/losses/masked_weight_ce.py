import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedWeightCrossEntropy(nn.Module):
    def __init__(self, proportions):
        super(MaskedWeightCrossEntropy, self).__init__()
        self.weights = 1 / (proportions / proportions.sum() + 2e-3)
        self.weights = torch.from_numpy(self.weights).float().cuda(non_blocking=True)

    def forward(self, logit, target, mask):
        loss = F.cross_entropy(logit, target, weight=self.weights, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()
