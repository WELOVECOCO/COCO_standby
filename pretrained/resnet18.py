from core.nn import Conv2d, MaxPool2d, Linear, Relu, ConvBatchNorm2D, GAP , Flatten
from core.Models import Model
import numpy as np
# class BasicBlock(Model):
#     expansion = 1
    
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         # First conv layer
#         self.conv1 = Conv2d(
#             input_channels=in_channels,
#             output_channels=out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             initialize_type='zero',
#             bias=False
#         )
#         self.bn1 = ConvBatchNorm2D(out_channels)
#         self.relu = Relu()
        
#         # Second conv layer
#         self.conv2 = Conv2d(
#             input_channels=out_channels,
#             output_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             initialize_type='zero',
#             bias=False
            
#         )
#         self.bn2 = ConvBatchNorm2D(out_channels)
        
#         self.downsample = downsample
        
#     def forward(self, x):
#         identity = x
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)
            
#         out += identity
#         out = self.relu(out)
        
#         return out
class BasicBlock(Model):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # First conv layer
        self.conv1 = Conv2d(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            initialize_type='zero',
            bias=False
        )
        self.bn1 = ConvBatchNorm2D(out_channels)
        self.relu = Relu()
        
        # Second conv layer
        self.conv2 = Conv2d(
            input_channels=out_channels,
            output_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            initialize_type='zero',
            bias=False
            
        )
        self.bn2 = ConvBatchNorm2D(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Ensure shapes match before addition
        # Get the shapes
        out_shape = out.data.shape
        identity_shape = identity.data.shape
        
        # If shapes don't match, make them match
        if out_shape[2] != identity_shape[2] or out_shape[3] != identity_shape[3]:
            # Resize identity or out to match
            if out_shape[2] > identity_shape[2] or out_shape[3] > identity_shape[3]:
                # Pad identity
                pad_h = max(0, out_shape[2] - identity_shape[2])
                pad_w = max(0, out_shape[3] - identity_shape[3])
                if pad_h > 0 or pad_w > 0:
                    identity.data = np.pad(identity.data, 
                                          ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 
                                          mode='constant')
            else:
                # Crop out
                out.data = out.data[:, :, :identity_shape[2], :identity_shape[3]]
        
        out.data = out.data + identity.data
        out = self.relu(out)
        
        return out

class DownsampleBlock(Model):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = Conv2d(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            initialize_type='zero',
            bias=False
        )
        self.bn = ConvBatchNorm2D(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class Backbone(Model):
    def __init__(self):
        super().__init__()
        # Initial conv layer
        self.conv1 = Conv2d(
            input_channels=3,
            output_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            initialize_type='zero',
            bias=False
        )
        self.bn1 = ConvBatchNorm2D(64)
        self.relu = Relu()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1 (64 filters)
        self.block1_1 = BasicBlock(64, 64)
        self.block1_2 = BasicBlock(64, 64)
        
        # Layer 2 (128 filters)
        # self.downsample2_1 = DownsampleBlock(64, 128, stride=2)
        self.block2_1 = BasicBlock(64, 128, stride=2, downsample=DownsampleBlock(64, 128, stride=2))
        self.block2_2 = BasicBlock(128, 128)
        
        # Layer 3 (256 filters)
        # self.downsample3_1 = DownsampleBlock(128, 256, stride=2)
        self.block3_1 = BasicBlock(128, 256, stride=2, downsample=DownsampleBlock(128, 256, stride=2))
        self.block3_2 = BasicBlock(256, 256)
        
        # Layer 4 (512 filters)
        # self.downsample4_1 = DownsampleBlock(256, 512, stride=2)
        self.block4_1 = BasicBlock(256, 512, stride=2, downsample=DownsampleBlock(256, 512, stride=2))
        self.block4_2 = BasicBlock(512, 512)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block1_1(x)
        x = self.block1_2(x)
        
        x = self.block2_1(x)
        x = self.block2_2(x)
        
        x = self.block3_1(x)
        x = self.block3_2(x)
        
        x = self.block4_1(x)
        x = self.block4_2(x)
        
        return x

class Classifier(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.avgpool = GAP()  # Global Average Pooling
        self.fc = Linear(512 * BasicBlock.expansion, num_classes, initialize_type='zero')
        self.flatten = Flatten()
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ResNet_18(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = Backbone()
        self.classifier = Classifier(num_classes=num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def load_from_pretrained(model, strict=True):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    
    # Download the pre-trained ResNet-18 model
    pytorch_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
    state_dict = pytorch_resnet18.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if "num_batches_tracked" not in k}
    model.load_weights_by_structure(filtered_state_dict, strict=strict)
    print("Pre-trained ResNet-18 weights loaded from .pth file")

@staticmethod
def resnet18(pretrained=True,strict=True, **kwargs):
    model = ResNet_18(**kwargs)
    if pretrained:
        load_from_pretrained(model,strict=strict)
    return model

