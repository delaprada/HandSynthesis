from torch.autograd import Variable
import torch.nn as nn
import math
import torch

class ResNet(nn.Module):

    def __init__(self, block, layers, z_dim, hand_side_invariance=False):
        """
        :param block:
        :param layers:
        :param z_dim:
        :param hand_side_invariance:
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        if hand_side_invariance:
            self.fc = nn.Linear(512 * 2 * 2 + 2, z_dim * 2)
        else:
            self.fc = nn.Linear(512 * 2 * 2, z_dim * 2)
        # Init the weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, hand_side=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # 3 512 4 4
        x = x.view(x.size(0), -1)
        if hand_side is None:
            x = self.fc(x)
        else:
            x = self.fc(torch.cat((x, hand_side), dim=1))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        一个残差块。
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 3 64 32 32
        out = self.conv1(x)  # 3 64 30 30
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 3 64 28 28
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


############ DECODE & ENCODE ############

class RGBEncoder(nn.Module):

    def __init__(self, z_dim, hand_side_invariance):
        """
        输入是图片，256 * 256 输出是 2 * z_dim 的向量，前一半是均值，后一半是方差。
        :param z_dim: 隐空间的维度。
        :param hand_side_invariance:
        """
        super(RGBEncoder, self).__init__()
        self.model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], z_dim=z_dim,
                            hand_side_invariance=hand_side_invariance)

    def forward(self, x):
        return self.model(x)


class RGBDecoder(nn.Module):
    def __init__(self, z_dim):
        """
        decode for z vector to RGB, (32, ) -> (3, 128, 128)
        :param z_dim:
        """
        super(RGBDecoder, self).__init__()
        self.linear_lay = nn.Sequential(
            nn.Linear(z_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.ReLU())
        # in_channels, out_channels, kernel_size, stride=1
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3,
                               kernel_size=4, stride=2, padding=1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # out put is 3 x 256 x 256

    def forward(self, x):
        out_lay = self.linear_lay(x)
        in_conv = out_lay.view(-1, 128, 8, 8)
        out_conv = self.conv_blocks(in_conv)
        out_conv = self.upsample(out_conv)
        return out_conv