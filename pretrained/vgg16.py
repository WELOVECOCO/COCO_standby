from core.nn import Conv2d, MaxPool2d, Linear, Relu ,GAP ,ConvBatchNorm2D , Flatten
from core.Models import Model


class Features(Model):
    def __init__(self):
        super().__init__()
        # Block 1: 2 conv layers with 64 filters
        self.conv1_1 = Conv2d(input_channels=3, output_channels=64, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu1_1 = Relu()
        self.conv1_2 = Conv2d(input_channels=64, output_channels=64, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu1_2 = Relu()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 2 conv layers with 128 filters
        self.conv2_1 = Conv2d(input_channels=64, output_channels=128, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu2_1 = Relu()
        self.conv2_2 = Conv2d(input_channels=128, output_channels=128, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu2_2 = Relu()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 3 conv layers with 256 filters
        self.conv3_1 = Conv2d(input_channels=128, output_channels=256, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu3_1 = Relu()
        self.conv3_2 = Conv2d(input_channels=256, output_channels=256, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu3_2 = Relu()
        self.conv3_3 = Conv2d(input_channels=256, output_channels=256, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu3_3 = Relu()
        self.maxpool3 = MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 3 conv layers with 512 filters
        self.conv4_1 = Conv2d(input_channels=256, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu4_1 = Relu()
        self.conv4_2 = Conv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu4_2 = Relu()
        self.conv4_3 = Conv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu4_3 = Relu()
        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 3 conv layers with 512 filters
        self.conv5_1 = Conv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu5_1 = Relu()
        self.conv5_2 = Conv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu5_2 = Relu()
        self.conv5_3 = Conv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.relu5_3 = Relu()
        self.maxpool5 = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.maxpool5(x)

        return x

class Classifier(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Assuming input to classifier is 512 * 7 * 7 (after conv layers with input size 224x224)
        self.fc1 = Linear(512 * 7 * 7, 4096, initialize_type='xavier')
        self.relu1 = Relu()
        self.fc2 = Linear(4096, 4096, initialize_type='xavier')
        self.relu2 = Relu()
        self.fc3 = Linear(4096, num_classes, initialize_type='xavier')

    def forward(self, x):
        # x = x.view(x.shape[0], -1)  # Flatten [B, 512, 7, 7] to [B, 512*7*7]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class VGG16_model(Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = Features()
        self.flatten = Flatten()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def load_from_pretrained(model):
    import torch
    from torchvision.models import vgg16, VGG16_Weights

    # Download the pre-trained VGG-16 model
    pytorch_vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1, progress=True)
    state_dict = pytorch_vgg16.state_dict()
    model.load_weights_by_structure(state_dict, strict=True)
    print("Pre-trained VGG-16 weights loaded from .pth file")


@staticmethod
def VGG16(pretrained=True, **kwargs):
    model = VGG16_model()
    if pretrained:
        load_from_pretrained(model)
    return model