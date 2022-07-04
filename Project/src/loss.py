import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_focal_loss(inputs:torch.Tensor, gamma:float):
    p = torch.exp(-inputs)
    loss = (1-p) ** gamma * inputs
    return loss.mean()

# focal loss object
class FocalLossLDAM(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 0.1):
        super(FocalLossLDAM, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.gamma = gamma
        self.weight = weight

    def forward(self, input : torch.Tensor, target : torch.Tensor)->torch.Tensor:
        return compute_focal_loss(F.cross_entropy(input, target, reduction = 'mean', weight = self.weight), self.gamma)
