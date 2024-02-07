import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
        
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
