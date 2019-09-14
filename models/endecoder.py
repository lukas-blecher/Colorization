import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, drop_rate=0, classes=3):
        '''
        This network is similar to the unet but misses the skip connections. In return we can make the network deeper 
        while using the same amount of gpu ram.
        '''
        self.classes=classes
        super(generator,self).__init__()

        #encoder and decoder architecutres
        self.encoder = nn.Sequential(conv_block(1,64,False,drop_rate,.2),   conv_block(64,128,True,drop_rate,.2),
                                     conv_block(128,256,True,drop_rate,.2), conv_block(256,512,True,drop_rate,.2,stride=1),
                                     conv_block(512,512,True,drop_rate,.2,stride=1), conv_block(512,512,True,drop_rate,.2,num=3,stride=1))
        
        self.decoder = nn.Sequential(up_conv(512,256),conv_block(256,256,True,drop_rate,num=2,stride=1),
                                    up_conv(256,128),conv_block(128,128,True,drop_rate,num=2,stride=1),
                                    #up_conv(128,64),conv_block(64,64,True,drop_rate,num=2,stride=1),
                                    #up_conv(64,32),conv_block(32,32,True,drop_rate,num=2,stride=1),
                                    #up_conv(32,16),conv_block(16,16,True,drop_rate,num=2,stride=1),
                                    up_conv(128,64),conv_block(64,classes,True,drop_rate,num=2,stride=1))

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
        x = self.encoder(x)
        # expansion
        x = self.decoder(x)
        return torch.sigmoid(x)
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, drop_rate=0, leaky=None, num=2, stride=2):
        super(conv_block,self).__init__()
        self.layer_list = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)]
        for i in range(num-1):
            self.layer_list.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)
            ])
        if bn:
            self.layer_list.append(nn.BatchNorm2d(out_channels))
        if drop_rate > 0:
            self.layer_list.append(nn.Dropout2d(drop_rate))

        self.convs = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.convs(x)
        return x
# Upsampling with Transposed Convolution
class up_conv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(up_conv,self).__init__()
        # half the channels, double the resolution
        #first to save gpu mem
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample=nn.ConvTranspose2d(in_channels,out_channels,2,stride=2)
        #self.double_conv = double_conv(out_channels,out_channels)
    # takes two inputs: one from the skip connection and the other from
    # the last convolutional layer
    def forward(self,x):
        #first upsampling
        x=self.upsample(x)
        # concatenate skip-connection and x
        #x=torch.cat([skip,x],dim=1)
        #x=self.double_conv(x)
        return x