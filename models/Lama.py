import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class FU(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels*2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels*2),
            nn.ReLU()
        )
    def forward(self, x):
        out = torch.fft.rfft2(x)
        y_r, y_i = out.real, out.imag
        out = self.layers(torch.cat([y_r, y_i], dim=1))
        y_r, y_i = out[:, :out.shape[1]//2, :, :], out[:, out.shape[1]//2:, :, :]
        out = torch.fft.irfft2(torch.complex(y_r, y_i))
        return out
class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
        self.layers2 = FU(out_channels, out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
    def forward(self, x):
        skip = self.layers1(x)
        out = self.layers2(skip)
        out = out + skip
        return self.out_conv(out)

class FFCRB(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        assert in_channels%2==0, "in_channels must be divisible by 2"
        assert out_channels%2==0, "out_channels must be divisible by 2"
        self.local1 = nn.Conv2d(in_channels//2, out_channels//2, 3, 1, 1)
        self.local2 = nn.Conv2d(in_channels//2, out_channels//2, 3, 1, 1)
        self.global1 = nn.Conv2d(in_channels//2, out_channels//2, 3, 1, 1)
        self.global2 = SpectralTransform(in_channels//2, out_channels//2)
        self.norm_relu_local = nn.Sequential(
            nn.InstanceNorm2d(out_channels//2),
            nn.ReLU()
        )
        self.norm_relu_global = nn.Sequential(
            nn.InstanceNorm2d(out_channels//2),
            nn.ReLU()
        )
    def forward(self, x):
        x_l, x_g = x[:, :x.shape[1]//2, :, :], x[:, x.shape[1]//2:, :, :]
        out_l = self.local1(x_l) + self.global1(x_g)
        out_g = self.local2(x_l) + self.global2(x_g)
        out_l = self.norm_relu_local(out_l)
        out_g = self.norm_relu_global(out_g)
        return torch.cat([out_l, out_g], dim=1)

class RBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        assert in_channels%2==0, "in_channels must be divisible by 2"
        assert out_channels%2==0, "out_channels must be divisible by 2"
        if (in_channels==out_channels):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.layers = FFCRB(in_channels, out_channels)

    def forward(self, x):
        return self.skip(x)+self.layers(x)

class Lama(nn.Module):
    def __init__(self, scale=1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, int(32*scale), 1, 1, 0),
            RBlock(int(32*scale), int(64*scale)),
            nn.MaxPool2d(2),
            RBlock(int(64*scale), int(128*scale)),
            nn.MaxPool2d(2),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            RBlock(int(128*scale), int(128*scale)),
            nn.Upsample(scale_factor=2),
            RBlock(int(128*scale), int(64*scale)),
            nn.Upsample(scale_factor=2),
            RBlock(int(64*scale), int(32*scale)),
            nn.Conv2d(int(32*scale), 3, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Lama(3).to(device)
    summary(model, (1, 3, 512, 512))