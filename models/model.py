import torch
import torch.nn.functional as F
import torch.nn as nn
import models.resnet as resnet

class model(nn.Module):
    def __init__(self,block=resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,col_channels=3):
        
        super(model,self).__init__()
        #using first half of the resnet
        
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        #now upsampling
        self.up1 = up_conv(512,256)
        self.up2 = up_conv(256,128)
        self.up3 = up_conv(128,64)
        self.up4 = up_conv(64,64)
        self.up5 = up_conv(64,col_channels,after_skip=4)
        #rgb output
        self.out_conv = nn.Conv2d(col_channels,col_channels,1)
        self.tanh=nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x2 = self.maxpool(x1)

            x2 = self.layer1(x2)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
        #print(x.shape,x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        x5 = self.up1(x4,x5)
        x5 = self.up2(x3,x5)
        x5 = self.up3(x2,x5)
        x5 = self.up4(x1,x5)
        x5 = self.up5(x[:,:1,:,:],x5)
        x = self.out_conv(x5)

        return self.tanh(x)

# expansion:
# Upsampling with Transposed Convolution
class up_conv(nn.Module):
    def __init__(self, in_channels,out_channels,after_skip=None):
        super(up_conv,self).__init__()
        # half the channels, double the resolution
        if after_skip is None:
            after_skip=2*out_channels
        #first to save gpu mem
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample=nn.ConvTranspose2d(in_channels,out_channels,2,stride=2)
        self.double_conv = double_conv(after_skip,out_channels)
    # takes two inputs: one from the skip connection and the other from
    # the last convolutional layer
    def forward(self,skip,x):
        #first upsampling
        x=self.upsample(x)
        # concatenate skip-connection and x
        x=torch.cat([skip,x],dim=1)
        x=self.double_conv(x)
        return x


# double convolution without pooling layer
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv,self).__init__()

        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )
        
    def forward(self, x):
        x = self.double_conv(x)
        return x