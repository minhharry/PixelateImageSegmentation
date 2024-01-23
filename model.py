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
            nn.InstanceNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_channels),
        )
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return F.relu(self.layers(x)+self.skip_connection(x))
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        scale = 0.75
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
            nn.Conv2d(int(64*scale), 1, 3, 1, 1),
        )
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder().to(device)
    summary(model, (1, 3, 512, 512))
