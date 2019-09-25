import torch
import torch.nn.functional as F
import torch.nn as nn
from models.unet import double_conv,double_conv_pool,up_conv
#from settings import s


class middle_unet(nn.Module):
    def __init__(self, bn=True, drop_rate=0, classes=3):
        '''
        '''
        self.classes=classes
        super(middle_unet,self).__init__()
        #first layer without pooling
        self.input = double_conv(1,32, bn)
        #begin contraction
        self.contract1 = double_conv_pool(32,64, bn, drop_rate)
        self.contract2 = double_conv_pool(64,128, bn, drop_rate)
        self.contract3 = double_conv_pool(128,256, bn, drop_rate)
        self.contract4 = double_conv_pool(256,512, bn, drop_rate)
        #begin expansion
        self.expanse1 = up_conv(512, bn, drop_rate)
        self.expanse2 = up_conv(256, bn, drop_rate)
        self.expanse3 = up_conv(128, bn, drop_rate)
        self.expanse4 = up_conv(64, bn, drop_rate)
        #out convolution
        self.out_conv = nn.Conv2d(32,classes,1)
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

