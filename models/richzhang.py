import torch
import torch.nn as nn

class richzhang(nn.Module):
    def __init__(self, drop_rate=0, classes=274):
        '''
        This network is similar to the unet but misses the skip connections. In return we can make the network deeper 
        while using the same amount of gpu ram.
        '''
        self.classes=classes
        super(richzhang,self).__init__()

        #encoder and decoder architecutres
        
        self.encoder = nn.Sequential(conv_block(1,64,1,3,1,1,drop_rate=drop_rate),conv_block(64,64,1,3,1,1,drop_rate=drop_rate),conv_block(64,64,1,3,2,1,bn=True,drop_rate=drop_rate),
                                    conv_block(64,128,1,3,1,1,drop_rate=drop_rate),conv_block(128,128,1,3,2,1,bn=True,drop_rate=drop_rate),
                                    conv_block(128,256,1,3,1,1,drop_rate=drop_rate),conv_block(256,256,1,3,1,1,drop_rate=drop_rate),conv_block(256,256,1,3,2,1,bn=True,drop_rate=drop_rate),
                                    conv_block(256,512,1,3,1,1,drop_rate=drop_rate),conv_block(512,512,1,3,1,1,num=2,bn=True,drop_rate=drop_rate),
                                    conv_block(512,512,2,3,1,2,num=3,bn=True,drop_rate=drop_rate))
                                    
        
        self.decoder = nn.Sequential(conv_block(512,512,2,3,1,2,num=3,bn=True,drop_rate=drop_rate),conv_block(512,512,1,3,1,1,num=3,bn=True,drop_rate=drop_rate),
                                    up_conv(512,256,1,4,2),conv_block(256,256,1,3,1,1,num=2,bn=True,drop_rate=drop_rate),
                                    conv_block(256,classes,0,1,1,1,drop_rate=drop_rate))
        self.out = nn.Softmax(1)
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
        shape=x.shape
        # contraction
        x = self.encoder(x)
        # expansion
        x = self.decoder(x)
        return self.out(x)
        
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,pad=1,kernel=3, stride=2,dil=1, drop_rate=0, bn=False, leaky=None, num=1):
        super(conv_block,self).__init__()
        self.layer_list = [
                nn.Conv2d(in_channels, out_channels, kernel, stride, pad, dil),
                nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)] * num
        
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
    def __init__(self, in_channels,out_channels,pad=1,kernel=3, stride=2,leaky=None):
        super(up_conv,self).__init__()
        # half the channels, double the resolution
        #first to save gpu mem
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample=nn.ConvTranspose2d(in_channels,out_channels,kernel,stride,pad)
        self.relu=nn.ReLU(inplace=True) if leaky is None else nn.LeakyReLU(leaky,True)

    def forward(self,x):
        x=self.upsample(x)
        return self.relu(x)