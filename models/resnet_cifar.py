"""Dilated ResNet"""
import torch.nn as nn

from .customize import FrozenBatchNorm2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

def conv_1_3x3(input_channel):
    return nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3, 64, 7, 2, 3
                         FrozenBatchNorm2d(64),
                         nn.ReLU(inplace=True))
                         # nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = FrozenBatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=strides, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = FrozenBatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = FrozenBatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = FrozenBatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class basic_block(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, strides=(2, 2)):
        super(basic_block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=strides, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn1 = FrozenBatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = FrozenBatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn3 = FrozenBatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.conv3(input_tensor)
        shortcut = self.bn3(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class identity_block3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = FrozenBatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = FrozenBatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = FrozenBatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor, return_conv3_out=False):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class identity_block2(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size):
        super(identity_block2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn1 = FrozenBatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = FrozenBatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor, return_conv3_out=False):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += input_tensor
        out = self.relu(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, input_channel, num_classes, include_top=True):
        print('CIFAR Resnet50 is used')
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.include_top = include_top
        block_ex = 4

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3( self.input_channel )

        self.bottleneck_1 = bottleneck(16 * block_ex, [16 * block_ex, 16 * block_ex, 64 * block_ex], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)

        self.bottleneck_2 = bottleneck(64*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)

        self.bottleneck_3 = bottleneck(128*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, strides=(1, 1))
        self.identity_block_3_1 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)

        self.bottleneck_4 = bottleneck(256*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512*block_ex, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, FrozenBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv_3x3(input_x)
        ret1 = x
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        ret2 = x
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        ret3 = x
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        ret4 = x
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        ret5 = x

        x = self.avgpool(x)
        if self.include_top:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


class Resnet18(nn.Module):
    """
    100% CIFAR10: 95.08%
    80% CIFAR10: 94.07%
    60% CIFAR10: 93.08%
    40% CIFAR10: 91.52%
    20% CIFAR10: 86.49%
    10% CIFAR10: 77.84%
    5% CIFAR10: 62.15%
    1% CIFAR10: 38.8%
    0.5% CIFAR10: 17.46%
    """
    def __init__(self, input_channel, num_classes):
        print('CIFAR Resnet18 is used')
        super(Resnet18, self).__init__()
        self.num_classes = num_classes
        self.input_channel = input_channel

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3( self.input_channel )

        self.identity_block_1_0 = identity_block2(64, 64, kernel_size=3)
        self.identity_block_1_1 = identity_block2(64, 64, kernel_size=3)

        self.basic_block_2 = basic_block(64, 128, kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block2(128, 128, kernel_size=3)

        self.basic_block_3 = basic_block(128, 256, kernel_size=3, strides=(1, 1))
        self.identity_block_3_1 = identity_block2(256, 256, kernel_size=3)

        self.basic_block_4 = basic_block(256, 512, kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block2(512, 512, kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, FrozenBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv_3x3(input_x)
        ret1 = x
        x = self.identity_block_1_0(x)
        x = self.identity_block_1_1(x)
        ret2 = x
        x = self.basic_block_2(x)
        x = self.identity_block_2_1(x)
        ret3 = x
        x = self.basic_block_3(x)
        x = self.identity_block_3_1(x)
        ret4 = x
        x = self.basic_block_4(x)
        x = self.identity_block_4_1(x)
        ret5 = x
        print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_cifar_resnet(arch, pretrained, **kwargs):
    if arch == "resnet18":
        model = Resnet18(**kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    elif arch == "resnet50":
        model = Resnet50(**kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model








