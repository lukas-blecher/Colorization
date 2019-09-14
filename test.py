import torch
import os
from models.unet import unet
from models.model import model
from models.endecoder import generator
from models.richzhang import richzhang
from torchvision import transforms
from settings import s
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from show import show_colorization
from functions import load_trainset
from skimage import color

def main(argv):
    
    data_path = s.data_path
    weight_path = s.weights_path
    mode=1
    drop_rate=s.drop_rate
    lab=s.lab
    classification=False
    try:
        opts, args = getopt.getopt(argv,"h:w:p:b:m:ld:c",["help", "weight-path=", "datapath=",'model=','lab','drop-rate='])
    except getopt.GetoptError as error:
        print(error)
        print( 'test.py -i <Boolean> -s <Boolean>')
        sys.exit(2)
    print("opts", opts)
    for opt, arg in opts:
        if opt == '-h':
            print( 'test.py -i <Boolean> -s <Boolean>')
            sys.exit()
        elif opt in ("-w", "--weight-path"):
            weight_path = arg
        elif opt in ("--datapath", "-p"):
            data_path = arg
        elif opt in ("--batchnorm", "-b"):
            batch_norm = arg in ["True", "true", "1"]
        elif opt=='-m':
            if arg in ('custom','0'):
                mode = 0
            elif arg in ('u','1','unet'):
                mode = 1
            elif arg in ('ende','2'):
                mode = 2
            elif arg in ('richzhang','classende','3'):
                mode = 3
        elif opt in ('-l','--lab'):
            lab=True
        elif opt in ("-d", "--drop-rate"):
            drop_rate = float(arg) 
        elif opt =='-c':
            classification=True
            lab=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset=None
    if data_path == './cifar-10':
        in_size = 32
        dataset = 0
    elif 'places' in data_path:
        in_size = 256
        dataset = 1
    in_shape=(3,in_size,in_size)
    #out_shape=(s.classes,32,32)

    trainset = load_trainset(data_path,lab=lab)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,
                                        shuffle=True, num_workers=2)
    print("Loaded dataset from", data_path)
    classes=(274 if classification else 2) if lab else 3

    #define model
    UNet=None
    zoom=False
    if mode == 0:
        UNet=model(col_channels=classes) 
    elif mode ==1:
        UNet=unet(drop_rate=drop_rate,classes=classes)
    elif mode ==2:
        UNet=generator(drop_rate,classes)
    elif mode == 3:
        UNet=richzhang(drop_rate)
        zoom=True
    #load weights
    try:
        UNet.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Did not find weight files.")
        #sys.exit(2)
    
    UNet.to(device)  
    UNet.eval()
    gray = torch.tensor([0.2989 ,0.5870, 0.1140])[:,None,None].float()
    with torch.no_grad():
        for i,batch in enumerate(trainloader):
            if dataset == 0: #cifar 10
                (image,_) = batch
            elif dataset == 1: #places
                image = batch
            X=None
            if lab:
                if dataset == 0: #cifar 10
                    image=np.transpose(image,(0,3,2,1))
                    image=np.transpose(color.rgb2lab(image),(0,3,2,1))
                    image=torch.from_numpy((image+np.array([0,128,128])[None,:,None,None])/np.array([100,255,255])[None,:,None,None]).float()
                X=torch.unsqueeze(image[:,0,:,:],1).to(device) #set X to the Lightness of the image
                image=image[:,1:,:,:].to(device) #image is a and b channel
            else:
                #convert to grayscale image
                
                #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
                X=(image.clone()*gray).sum(1).to(device).view(-1,1,*in_shape[1:])
                image=image.float().to(device)
            #print(X.min(),X.max())
            #generate colorized version with unet
            #for arr in (image[:,0,...],image[:,1,...],X):
            #    print(arr.min(),arr.max())
            try:
                unet_col=UNet(X)
            except:
                unet_col=UNet(torch.stack((X,X,X),1)[:,:,0,:,:])
            #for arr in (unet_col[0,...],unet_col[1,...]):
            #    print(arr.min().item(),arr.max().item())        
            show_colorization(unet_col,image,X,lab=lab,cl=classification,zoom=zoom)
if __name__ == '__main__':
    main(sys.argv[1:])