import torch
import torch.nn as nn
from BioMedGPT.models.ofa.resnet import *

class ResNet(nn.Module):
    """
    ResNet model implementation.

    Args:
        layers (list of int): Number of residual blocks in each layer.
        zero_init_residual (bool, optional): Zero-initialize the last BN in each residual branch.
        groups (int, optional): Number of groups in 3x3 convolution.
        width_per_group (int, optional): Width per group in 3x3 convolution.
        replace_stride_with_dilation (list of bool or None, optional): Replace stride with dilation in specified layers.
        norm_layer (nn.Module, optional): Normalization layer to use.
        drop_path_rate (float, optional): Drop path rate for stochastic depth.

    """
    def __init__(self, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_path_rate=0.0):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0], drop_path_rate=drop_path_rate)
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], drop_path_rate=drop_path_rate)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], drop_path_rate=drop_path_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, drop_path_rate=0.0):
        """
        Create a sequence of residual blocks.

        Args:
            block (nn.Module): Type of residual block to create.
            planes (int): Number of output channels for each block.
            blocks (int): Number of blocks to create.
            stride (int, optional): Stride for the first block.
            dilate (bool, optional): Apply dilation to the block.
            drop_path_rate (float, optional): Drop path rate for stochastic depth.

        Returns:
            nn.Sequential: Sequence of residual blocks.

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, blocks)]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path_rate=dpr[i]))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """
        Internal implementation of the forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self._forward_impl(x)
