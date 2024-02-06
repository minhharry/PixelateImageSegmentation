import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg19

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

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg1 = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[0:4].to(device)
        self.vgg2 = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[4:9].to(device)
        self.vgg3 = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[9:18].to(device)
        self.vgg4 = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[18:].to(device)
        self.weight = [1/8, 1/4, 1/2, 1]
        self.loss = nn.MSELoss()
        for param in self.vgg1.parameters():
            param.requires_grad = False
        for param in self.vgg2.parameters():
            param.requires_grad = False
        for param in self.vgg3.parameters():
            param.requires_grad = False
        for param in self.vgg4.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features1 = self.vgg1(input)
        vgg_target_features1 = self.vgg1(target)
        vgg_input_features2 = self.vgg2(vgg_input_features1)
        vgg_target_features2 = self.vgg2(vgg_target_features1)
        vgg_input_features3 = self.vgg3(vgg_input_features2)
        vgg_target_features3 = self.vgg3(vgg_target_features2)
        vgg_input_features4 = self.vgg4(vgg_input_features3)
        vgg_target_features4 = self.vgg4(vgg_target_features3)
        return self.loss(vgg_input_features1, vgg_target_features1)*self.weight[0]+self.loss(vgg_input_features2, vgg_target_features2)*self.weight[1]+self.loss(vgg_input_features3, vgg_target_features3)*self.weight[2]+self.loss(vgg_input_features4, vgg_target_features4)*self.weight[3]

if __name__ == "__main__":
    model = VGGLoss(device='cuda')
    summary(model, [(1,3,512,512),(1,3,512,512)])