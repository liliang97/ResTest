import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        # 参数比调用多几个，模型相较于最初发文章的时候有过更新
        # block: basicblock或者bottleneck，后续会提到
        # layers：每个block的个数，如resnet50， layers=[3,4,6,3]
        # num_classes: 数据库类别数量
        # zero_init_residual：其他论文中提到的一点小trick，残差参数为0
        # groups：卷积层分组，应该是为了resnext扩展
        # width_per_group：同上，此外还可以是wideresnet扩展
        # replace_stride_with_dilation：空洞卷积，非原论文内容
        # norm_layer：原论文用BN，此处设为可自定义
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.groups = groups
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer1(block, 256, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.last1 = self.make_layer2(block, 256)
        self.last2 = self.make_layer1(block, 512, stride=2)
        self.last3 = self.make_layer2(block, 512)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.channel1 = nn.Sequential(  # spatial attention map
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            #            nn.Softmax()
        )

        self.softmax = SpatialSoftmax(7, 7, 1, temperature=1)

        # 倒数1,2层卷积操作
        self.conv_last23 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # 倒数第3层卷积操作
        self.conv_last1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.channel2 = nn.Sequential(  # spatial logits
            nn.BatchNorm2d(384),
            nn.Conv2d(384, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Softmax()
        )

        self.avg = nn.AdaptiveAvgPool2d((7, 7))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 当需要特征图需要降维或通道数不匹配的时候调用
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 每一个self.layer的第一层需要调用downsample，所以单独写，跟下面range中的1 相对应
        # block的定义看下文
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def make_layer1(self, block, planes, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 当需要特征图需要降维或通道数不匹配的时候调用
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 每一个self.layer的第一层需要调用downsample，所以单独写，跟下面range中的1 相对应
        # block的定义看下文
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def make_layer2(self, block, planes, stride=1, dilate=False):
        norm_layer = self._norm_layer
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        # 每一个self.layer的第一层需要调用downsample，所以单独写，跟下面range中的1 相对应
        # block的定义看下文

        self.inplanes = planes * block.expansion

        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        l1 = self.last1(x)
        l2 = self.last2(l1)
        l3 = self.last3(l2)
        #x = self.layer4(x)

        A = self.channel1(l3)
        A = self.softmax(A)

        #l1 Resize成7*7特征图大小
        l1 = self.avg(l1)

        x1 = self.conv_last1(l1)
        x2 = self.conv_last23(l2)
        x3 = self.conv_last23(l3)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.channel2(y)
        y = A * y

        y = y.view(-1, self.num_classes, 49)
        y = torch.sum(y, dim=2)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return y

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        # pos_x, pos_y = np.meshgrid(
        #         np.linspace(-1., 1., self.height),
        #         np.linspace(-1., 1., self.width)
        #         )
        # pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        # pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        # self.register_buffer('pos_x', pos_x)
        # self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        # expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        # expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        # expected_xy = torch.cat([expected_x, expected_y], 1)
        # feature_keypoints = expected_xy.view(-1, self.channel*2)
        softmax_attention = softmax_attention.view(-1,1,7,7)
        return softmax_attention

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    #inplanes：输入通道数
    #planes：输出通道数
    #base_width，dilation，norm_layer不在本文讨论范围
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        #中间部分省略
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #为后续相加保存输入
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            #遇到降尺寸或者升维的时候要保证能够相加
            identity = self.downsample(x)

        out += identity#论文中最核心的部分，resnet的简洁和优美的体现
        out = self.relu(out)

        return out

def ResNet18(num_classes):
    return ResNet(BasicBlock,[2, 2, 2, 2],num_classes = num_classes)

if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet18(10)
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    # print(model)
#    model.load_state_dict(torch.load(model_urls['ResNet50']))
    input = torch.randn(1, 3, 224, 224)
    # 分析FLOPs
    flops = FlopCountAnalysis(model, input)
    print("FLOPs: %.2fM", (flops.total()) / 1e6)
    out = model(input)
    # print(out.shape)