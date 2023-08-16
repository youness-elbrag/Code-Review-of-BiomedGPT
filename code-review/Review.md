### Code-Review
in Section we will go for main folder and file that related how does BioMedGPT designed because the code based on Project is bigger and include many external packages , which means we will cover most relative file used to build the based model following :

* **BiomedGPT model**:

BiomedGPT is developed based on OFA. in Case we will focus on the most important **building Blocks of BioMedGPT** 

<div align="center">
    <img src="./assets/blocksModel/model.png" width="600" height="300" />
</div>

###### building Blocks of BioMedGPT : 

1. **Handling Multi-modalites Input/Output**:

we explained how does BioMedGPT Handle multi-modalites fisrt let us dive in to the adjustment within Model done from Co-Authors 

* **add ResNet Blocks**:

To enable inputs with a wide range of modalities, including images, language, and bounding boxes, to be
processed within a single model, it is necessary to embed them in a shared and unified space. For visual inputs,
we directly apply CNN backbones to relax the heavy image feature extraction process, including object
detection. 

<div align="center">
    <img src="./assets/blocksModel/resnet.png" width="400" height="200" />
</div>
Specifically, BiomedGPT receives the raw image $\mathbf{x}_v \in \mathbb{R}^{H \times W \times C}$ and maps it into a flattened 1D sequence of patches $\mathbf{x}_p \in \mathbb{R}^{N \times D}$ via a ResNet module as input for the transformer, where $N = \frac{H \times W}{P^2}$ is the number of patches given the patch size of $P \times P$, and $D$ is the fixed hidden size of the transformer layers.

```python
class ResNet(nn.Module):

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

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, drop_path_rate=0.0):
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
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

```

**Addionally** : we have an Hyper-parameter **resnet_drop_path_rate** used 

 * Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is misleading as 'Drop Connect' is a.sh different form of dropout in a.sh separate paper.See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and argument names to 'drop path' rather than mix DropConnect as a.sh layer name and use 'survival rate' as the argument.

 check following [File](blocks_Model/dropath.py)


and has been adjust into file **ofa** which the main transformer model use to built BioMedGPT

```python
    unify_transformer.py    
```


* **byte-pair encoding**: 