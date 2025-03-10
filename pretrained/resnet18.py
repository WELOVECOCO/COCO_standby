from core.nn import Conv2d, MaxPool2d, Linear, Relu, batchnorm2d, GAP
from core.Models import Model

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
            initialize_type='zero'
        )
        self.bn1 = batchnorm2d(out_channels)
        self.relu = Relu()
        
        # Second conv layer
        self.conv2 = Conv2d(
            input_channels=out_channels,
            output_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            initialize_type='zero'
        )
        self.bn2 = batchnorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
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
            initialize_type='zero'
        )
        self.bn = batchnorm2d(out_channels)
    
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
            initialize_type='zero'
        )
        self.bn1 = batchnorm2d(64)
        self.relu = Relu()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1 (64 filters)
        self.block1_1 = BasicBlock(64, 64)
        self.block1_2 = BasicBlock(64, 64)
        
        # Layer 2 (128 filters)
        self.downsample2_1 = DownsampleBlock(64, 128, stride=2)
        self.block2_1 = BasicBlock(64, 128, stride=2, downsample=self.downsample2_1)
        self.block2_2 = BasicBlock(128, 128)
        
        # Layer 3 (256 filters)
        self.downsample3_1 = DownsampleBlock(128, 256, stride=2)
        self.block3_1 = BasicBlock(128, 256, stride=2, downsample=self.downsample3_1)
        self.block3_2 = BasicBlock(256, 256)
        
        # Layer 4 (512 filters)
        self.downsample4_1 = DownsampleBlock(256, 512, stride=2)
        self.block4_1 = BasicBlock(256, 512, stride=2, downsample=self.downsample4_1)
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
    
    def forward(self, x):
        x = self.avgpool(x)
        # x = x.view(x.shape[0], -1)  # Flatten
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
    model.load_weights_by_structure(state_dict, strict=strict)
    print("Pre-trained ResNet-18 weights loaded from .pth file")

@staticmethod
def resnet18(pretrained=True,strict=True, **kwargs):
    model = ResNet_18(**kwargs)
    if pretrained:
        load_from_pretrained(model,strict=strict)
    return model

