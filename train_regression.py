import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from itertools import count
import sys, getopt
from models.model import model
from models.unet import unet
from models.color_unet import color_unet
from models.endecoder import generator
from settings import s
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import json
from functions import load_trainset
from skimage import color

def main(argv):
    # setting argument defaults
    mbsize = s.batch_size
    report_freq=s.report_freq
    weight_path=s.weights_path
    weights_name=s.weights_name
    lr=s.learning_rate
    save_freq = s.save_freq
    mode=3
    epochs = s.epochs
    beta1,beta2=s.betas
    infinite_loop=s.infinite_loop
    data_path = s.data_path
    drop_rate = 0
    lab = s.lab
    load_list = s.load_list
    help='train_regression.py -b <batch size> -e <amount of epochs to train. standard: infinite> -r <report frequency> -w <path to weights folder> \
            -n <name> -s <save freq.> -l <learning rate> -p <path to data set> -d <dropout rate> -m <mode: differnet models> --beta1 <beta1 for adam>\
            --beta2 <beta2 for adam> --lab <No argument. If used lab colorspace is used> \
            --lambda <hyperparameter for class weights>'
    try:
        opts, args = getopt.getopt(argv,"he:b:r:w:l:s:n:m:p:d:i:",
            ['epochs=',"mbsize=","report-freq=",'weight-path=', 'lr=','save-freq=','weight-name=','mode=','data_path=','drop_rate='
            'beta1=','beta2=','lab','image-loss-weight=','load-list'])
    except getopt.GetoptError:
        print(help)
        sys.exit(2)
    print("opts" ,opts)
    for opt, arg in opts:
        if opt == '-h':
            print(help)
            sys.exit()
        elif opt in ("-b", "--mbsize"):
            mbsize = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
            infinite_loop=False
        elif opt in ('-r','--report-freq'):
            report_freq = int(arg)
        elif opt in ("-w", "--weight-path"):
            weight_path = arg
        elif opt in ("-n", "--weight-name"):
            weights_name = arg            
        elif opt in ("-s", "--save-freq"):
            save_freq=int(arg)
        elif opt in ("-l", "--lr"):
            lr = float(arg)
        elif opt=='-m':
            if arg in ('custom','0'):
                mode = 0
            elif arg in ('u','1','unet'):
                mode = 1
            elif arg in ('ende','2'):
                mode = 2
            elif arg in ('color','3','cu'):
                mode = 3
        elif opt in ("-p", "--data_path"):
            data_path = str(arg)
        elif opt in ("-d", "--drop_rate"):
            drop_rate = float(arg)
        elif opt=='--beta1':
            beta1 = float(arg)
        elif opt=='--beta2':
            beta2 = float(arg)
        elif opt=='--lab':
            lab=True
        elif opt in ('--load-list'):
            load_list=True

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=None
    if 'cifar' in data_path:
        in_size = 32
        dataset = 0
    elif 'places' in data_path:
        in_size = 224
        dataset = 1
    elif 'stl' in data_path:
        in_size = 96
        dataset = 2
    in_shape=(3,in_size,in_size)

    #out_shape=(s.classes,32,32)
    betas=(beta1,beta2)
    weight_path_ending=os.path.join(weight_path,weights_name+'.pth')

    loss_path_ending = os.path.join(weight_path, weights_name + "_" + s.loss_name)

    trainset = load_trainset(data_path,lab=lab,load_list=load_list)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize,
                                        shuffle=True, num_workers=2)
 
    print("NETWORK PATH:", weight_path_ending)
    #define output channels of the model
    classes=2 if lab else 3
    #define model
    UNet=None
    
    if mode ==0:
        UNet=model(col_channels=classes) 
    elif mode ==1:
        UNet=unet(drop_rate=drop_rate,classes=classes)
    elif mode ==2:
        UNet=generator(drop_rate,classes)
    elif mode ==3:
        UNet=color_unet(True,drop_rate,classes)
    #load weights
    try:
        UNet.load_state_dict(torch.load(weight_path_ending,map_location=device))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Initialize new weights for the generator.")
   

    UNet.to(device)

    #save the hyperparameters to a JSON-file for better oranization
    model_description_path_ending = os.path.join(weight_path, s.model_description_name)
    # initialize model dict
    try:
        with open(model_description_path_ending, "r") as file:
            model_dict = json.load(file)
    except FileNotFoundError:
        model_dict = {}


    prev_epochs=0
    # save settings in dict if new weights are beeing initialized
    if not weights_name in model_dict.keys():
        model_dict[weights_name] = {
            "loss_name": loss_path_ending,
            "epochs": 0,
            "batch_size": mbsize,
            "lr": lr,
            "lab":lab,
            "betas": betas,
            "model":['custom','unet','encoder-decoder','color-unet'][mode]
        }
    else:
        #load specified parameters from model_dict
        params=model_dict[weights_name]
        mbsize=params['batch_size']
        betas=params['betas']
        lr=params['lr']
        lab=params['lab']
        loss_path_ending=params['loss_name']
        #memorize how many epochs already were trained if we continue training
        prev_epochs=params['epochs']+1

    #optimizer
    optimizer_g=optim.Adam(UNet.parameters(),lr=lr,betas=betas)
    # l1 loss
    l1loss = nn.L1Loss().to(device)
    loss_hist=[]

    

    UNet.train()
    gray = torch.tensor([0.2989 ,0.5870, 0.1140 ])[:,None,None].float()
    # run over epochs
    for e in (range(prev_epochs, prev_epochs + epochs) if not infinite_loop else count(prev_epochs)):
        g_running=0
        #load batches          
        for i,batch in enumerate(trainloader):
            if dataset == 0: #cifar 10
                (image,_) = batch
            elif dataset in (1,2): #places and stl 10
                image = batch
                
            X=None
            #differentiate between the two available color spaces RGB and Lab
            if lab:
                if dataset == 0: #cifar 10
                    image=np.transpose(image,(0,2,3,1))
                    image=np.transpose(color.rgb2lab(image),(0,3,1,2))
                    image=torch.from_numpy((image+np.array([-50,0,0])[None,:,None,None])).float()
                X=torch.unsqueeze(image[:,0,:,:],1).to(device) #set X to the Lightness of the image
                image=image[:,1:,:,:].to(device) #image is a and b channel
            else:
                #convert to grayscale image
                #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
                X=(image.clone()*gray).sum(1).to(device).view(-1,1,*in_shape[1:])
                image=image.float().to(device)
            #----------------------------------------------------------------------------------------
            ################################### Unet optimization ###################################
            #----------------------------------------------------------------------------------------
            #clear gradients
            optimizer_g.zero_grad()
            #generate colorized version with unet
            unet_col=None
            #print(X.shape,image.shape,classes)
            if mode==0:
                unet_col=UNet(torch.stack((X,X,X),1)[:,:,0,:,:])
            else:
                unet_col=UNet(X)
            #calculate how close the generated pictures are to the ground truth
            loss_g=l1loss(unet_col,image)
            #backpropagation
            loss_g.backward()
            optimizer_g.step()

            g_running+=loss_g.item()
            loss_hist.append([e,i,loss_g.item()])

            #report running loss
            if (i+len(trainloader)*e)%report_freq==report_freq-1:
                print('Epoch %i, batch %i: \tunet loss=%.2e'%(e+1,i+1,g_running/report_freq))
                g_running=0

            if s.save_weights and (i+len(trainloader)*e)%save_freq==save_freq-1:
                #save parameters
                try:
                    torch.save(UNet.state_dict(),weight_path_ending)
                except FileNotFoundError:
                    os.makedirs(weight_path)
                    torch.save(UNet.state_dict(),weight_path_ending)
                print("Parameters saved")

                if s.save_loss:
                    #save loss history to file
                    try:
                        f=open(loss_path_ending,'a')
                        np.savetxt(f,loss_hist,'%e')
                        f.close()
                    except FileNotFoundError:
                        os.makedirs(s.loss_path)
                        np.savetxt(loss_path_ending,loss_hist,'%e')
                    loss_hist=[]

        #update epoch count in dict after each epoch
        model_dict[weights_name]["epochs"] = e  
        #save it to file
        try:
            with open(model_description_path_ending, "w") as file:
                json.dump(model_dict, file, sort_keys=True, indent=4)
        except:
            print('Could not save to model dictionary (JSON-file)')        

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    main(sys.argv[1:])