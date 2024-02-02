import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

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
        return nn.L1Loss()(pred_flat, target_flat)

class MaskBce(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1
    
    def forward(self, pred, target, mask):
        pred_flat = (pred*mask)
        target_flat = (target*mask)
        return nn.BCELoss()(pred_flat, target_flat)


if __name__ == "__main__":
    lossfn = MaskBce()
    print(lossfn(F.sigmoid(torch.randn((1,3,512,512))), F.sigmoid(torch.randn((1,3,512,512))), torch.ones((1,1,512,512))))