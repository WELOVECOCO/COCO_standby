import torch
import torch.nn as nn
from torchviz import make_dot


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + x  
        return out




residual_block = ResidualBlock(3, 3)
x = torch.randn(1, 3, 32, 32) 
out = residual_block(x)

make_dot(out, params=dict(residual_block.named_parameters())).render("residual_block", format="png")