import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None) -> None:
        super().__init__()
        if mid_channels==None:
            mid_channels = out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(mid_channels//4, mid_channels),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(out_channels//4, out_channels),
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return F.relu(self.layers(x)+self.skip_connection(x))
        
class AutoEncoder(nn.Module):
    def __init__(self, scale=0.5) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            DoubleConv(3, int(64*scale)),
            nn.MaxPool2d(2),
            DoubleConv(int(64*scale), int(128*scale)),
            nn.MaxPool2d(2),
            DoubleConv(int(128*scale), int(256*scale)),
            nn.MaxPool2d(2),
            DoubleConv(int(256*scale), int(512*scale)),
            nn.Upsample(scale_factor=2),
            DoubleConv(int(512*scale), int(256*scale)),
            nn.Upsample(scale_factor=2),
            DoubleConv(int(256*scale), int(128*scale)),
            nn.Upsample(scale_factor=2),
            DoubleConv(int(128*scale), int(64*scale)),
            nn.Conv2d(int(64*scale), 3, 3, 1, 1),
        )
    def forward(self, x):
        return self.layers(x)

class Unet(nn.Module):
    def __init__(self, scale=0.5) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            DoubleConv(3, int(64*scale)),
            nn.MaxPool2d(2)
        )
        self.down2 =  nn.Sequential(
            DoubleConv(int(64*scale), int(128*scale)),
            nn.MaxPool2d(2)
        )
        self.down3 =  nn.Sequential(
            DoubleConv(int(128*scale), int(256*scale)),
            nn.MaxPool2d(2)
        )
        self.down4 =  nn.Sequential(
            DoubleConv(int(256*scale), int(512*scale)),
            nn.MaxPool2d(2)
        )
        self.bottleneck = nn.Sequential(
            DoubleConv(int(512*scale), int(512*scale))
        )
        self.up1 =  nn.Sequential(
            DoubleConv(int(1024*scale), int(256*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up2 =  nn.Sequential(
            DoubleConv(int(512*scale), int(128*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up3 =  nn.Sequential(
            DoubleConv(int(256*scale), int(64*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up4 =  nn.Sequential(
            DoubleConv(int(128*scale), int(64*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(int(64*scale), 1, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottleneck(x4)
        out = self.up1(torch.cat([x5, x4], dim=1))
        out = self.up2(torch.cat([out, x3], dim=1))
        out = self.up3(torch.cat([out, x2], dim=1))
        out = self.up4(torch.cat([out, x1], dim=1))
        out = self.outconv(out)
        return out


class UResNet(nn.Module):
    def __init__(self, scale=0.5) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            DoubleConv(3, int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            nn.MaxPool2d(2)
        )
        self.down2 =  nn.Sequential(
            DoubleConv(int(64*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            nn.MaxPool2d(2)
        )
        self.down3 =  nn.Sequential(
            DoubleConv(int(128*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            nn.MaxPool2d(2)
        )
        self.down4 =  nn.Sequential(
            DoubleConv(int(256*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
            nn.MaxPool2d(2)
        )
        self.bottleneck = nn.Sequential(
            DoubleConv(int(512*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
            DoubleConv(int(512*scale), int(512*scale)),
        )
        self.up1 =  nn.Sequential(
            DoubleConv(int(1024*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            DoubleConv(int(256*scale), int(256*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up2 =  nn.Sequential(
            DoubleConv(int(512*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            DoubleConv(int(128*scale), int(128*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up3 =  nn.Sequential(
            DoubleConv(int(256*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.up4 =  nn.Sequential(
            DoubleConv(int(128*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            DoubleConv(int(64*scale), int(64*scale)),
            nn.Upsample(scale_factor=2)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(int(64*scale), 3, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottleneck(x4)
        out = self.up1(torch.cat([x5, x4], dim=1))
        out = self.up2(torch.cat([out, x3], dim=1))
        out = self.up3(torch.cat([out, x2], dim=1))
        out = self.up4(torch.cat([out, x1], dim=1))
        out = self.outconv(out)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, scale=1, in_channels=3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, int(64*scale), 3, 2, 1),
            nn.InstanceNorm2d(int(64*scale)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(64*scale), int(128*scale), 3, 2, 1),
            nn.InstanceNorm2d(int(128*scale)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(128*scale), int(256*scale), 3, 2, 1),
            nn.InstanceNorm2d(int(256*scale)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(256*scale), int(512*scale), 3, 2, 1),
            nn.InstanceNorm2d(int(512*scale)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(512*scale), 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Discriminator().to(device)
    summary(model, (1, 3, 512, 512))
