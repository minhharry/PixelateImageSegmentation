import torch
import torch.nn as nn
from torchinfo import summary

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1
    
    def forward(self, pred, target):
        size = target.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        intersection = (pred_flat*target_flat).sum()
        return 1 -  (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

class MaskL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1
    
    def forward(self, pred, target, mask):
        pred_flat = (pred*mask)
        target_flat = (target*mask)
        return nn.L1Loss()(pred_flat, target_flat)*262144/mask.sum()


if __name__ == "__main__":
    lossfn = MaskL1Loss()
    print(lossfn(torch.randn((1,4,512,512)), torch.randn((1,4,512,512))))