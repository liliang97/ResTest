import torch
import  torch.nn.functional as F
from    torch import nn


class LabelSmoothCELoss(nn.Module):
    def __init__(self,smoothing = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, label, smoothing=0.2):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss