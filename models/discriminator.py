import torch
import torch.nn as nn
from models.richzhang import conv_block

class critic(nn.Module):
    def __init__(self,im_size,classes=3):
        super(critic,self).__init__()

        self.cnn=nn.Sequential(convBlock(classes,16),
                               convBlock(16,32),
                               convBlock(32,64),
                               convBlock(64,128))
        proc_im_size=im_size//(2**4)
        self.fc=nn.Linear(128*proc_im_size**2,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        x=self.cnn(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return self.sig(x)

class gray_critic(nn.Module):
    def __init__(self,im_size):
        super(gray_critic,self).__init__()

        self.cnn=nn.Sequential(conv_Block(4,16),
                               convBlock(16,32),
                               convBlock(32,64),
                               convBlock(64,128))
        proc_im_size=im_size//(2**4)
        self.fc=nn.Linear(128*proc_im_size**2,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        x=self.cnn(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return self.sig(x)

class markov_critic(nn.Module):
    '''
    input: grayscale (first channel) and colored (last 3 channels) image

    return: real/fake classification of "patches" of input as tensor
    (generator loss will be cronstructed form mean)
    '''
    def __init__(self):
        super(markov_critic,self).__init__()

        self.cnn = nn.Sequential(conv_block(4, 64, pad=1, kernel=4, stride=2, leaky=.2),
                                conv_block(64, 128, pad=1, kernel=4, stride=2, leaky=.2, bn=True),
                                conv_block(128, 256, pad=1, kernel=4, stride=2, leaky=.2, bn=True),
                                conv_block(256, 512, pad=1, kernel=4, stride=1, leaky=.2, bn=True),
                                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                nn.Sigmoid())
    def forward(self, x):
        return self.cnn(x)
        

class convBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=2,padding=1):
        super(convBlock,self).__init__()

        self.block=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                 nn.LeakyReLU(.2,True),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout2d(.25))
    def forward(self,x):
        return self.block(x)