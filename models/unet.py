import torch
import torch.nn.functional as F
import torch.nn as nn
#from settings import s


class unet(nn.Module):
    def __init__(self, bn=True, drop_rate=0, classes=3):
        '''
        '''
        self.classes=classes
        super(unet,self).__init__()
        #first layer without pooling
        self.input = double_conv(1,64, bn)
        #begin contraction
        self.contract1 = double_conv_pool(64,128, bn, drop_rate)
        self.contract2 = double_conv_pool(128,256, bn, drop_rate)
        self.contract3 = double_conv_pool(256,512, bn, drop_rate)
        self.contract4 = double_conv_pool(512,1024, bn, drop_rate)
        #begin expansion
        self.expanse1 = up_conv(1024, bn, drop_rate)
        self.expanse2 = up_conv(512, bn, drop_rate)
        self.expanse3 = up_conv(256, bn, drop_rate)
        self.expanse4 = up_conv(128, bn, drop_rate)
        #out convolution
        self.out_conv = nn.Conv2d(64,classes,1)
        self.output = nn.Softmax(1) if classes>3 else nn.Sigmoid()
        # initializing weights:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        # contraction
        x1 = self.input(x)
        x2 = self.contract1(x1)
        x3 = self.contract2(x2)
        x4 = self.contract3(x3)
        x5 = self.contract4(x4)
        #expansion
        x = self.expanse1(x4,x5)
        x = self.expanse2(x3,x)
        x = self.expanse3(x2,x)
        x = self.expanse4(x1,x)
        x = self.out_conv(x)
        return self.output(x) #torch.sigmoid(x)


# contraction: building block of the left side of a unet
# two 3x3 convolutional layers with batchnormalization and relu activation
# additional max pooling layer in the beginning

# TODO:  * padding? in original paper the input gets downsampled --> no padding
#       * relu inplace?
class double_conv_pool(nn.Module):
    def __init__(self, in_channels, out_channels, bn, drop_rate=0):
        super(double_conv_pool,self).__init__()
        #half the resolution
        self.pool_layer=nn.MaxPool2d(2)
        if bn:
            self.layer_list = [
                nn.Conv2d(in_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if drop_rate > 0:
                self.layer_list = self.layer_list[:3] + [nn.Dropout2d(drop_rate)] + self.layer_list[3:] + [nn.Dropout2d(drop_rate)]

        else:
            self.layer_list = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
            if drop_rate > 0:
                self.layer_list = self.layer_list.append(nn.Dropout2d(drop_rate))
        
        self.double_conv = nn.Sequential(*self.layer_list)

    def forward(self,x):
        x=self.pool_layer(x)
        x=self.double_conv(x)
        return x


# double convolution without pooling layer
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bn, drop_rate=0,leaky=None):
        super(double_conv,self).__init__()
        if bn:
            self.layer_list = [
                nn.Conv2d(in_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True),
                nn.Conv2d(out_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)
            ]
            if drop_rate > 0:
                self.layer_list = self.layer_list[:3] + [nn.Dropout2d(drop_rate)] + self.layer_list[3:] + [nn.Dropout2d(drop_rate)]

        else:
            self.layer_list = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)
            ]
            if drop_rate > 0:
                self.layer_list.append(nn.Dropout2d(drop_rate))

        self.double_conv = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.double_conv(x)
        return x

# expansion:
# Upsampling with Transposed Convolution
class up_conv(nn.Module):
    def __init__(self, in_channels, bn, drop_rate=0):
        super(up_conv,self).__init__()
        # half the channels, double the resolution
        self.upsample=nn.ConvTranspose2d(in_channels,in_channels//2,2,stride=2)
        self.double_conv = double_conv(in_channels,in_channels//2, bn, drop_rate)
    # takes two inputs: one from the skip connection and the other from
    # the last convolutional layer
    def forward(self,skip,x):
        #first upsampling
        x=self.upsample(x)
        # concatenate skip-connection and x
        x=torch.cat([skip,x],dim=1)
        x=self.double_conv(x)
        return x

