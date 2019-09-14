import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from itertools import count
import sys, getopt
from models.discriminator import markov_critic
from models.model import model
from models.unet import unet
from settings import s
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import json
from functions import load_trainset

def main(argv):
    # setting argument defaults
    mbsize = s.batch_size
    report_freq=s.report_freq
    weight_path=s.weights_path
    weights_name=s.weights_name
    lr=s.learning_rate
    save_freq = s.save_freq
    mode=0
    image_loss_weight=s.image_loss_weight
    epochs = s.epochs
    beta1,beta2=s.betas
    infinite_loop=s.infinite_loop
    data_path = s.data_path
    help='test.py -b <int> -p <string> -r <int> -w <string>'
    try:
        opts, args = getopt.getopt(argv,"he:b:r:w:l:s:n:m:p:",
            ['epochs=',"mbsize=","report-freq=",'weight-path=', 'lr=','save-freq=','weight-name=','mode=','data_path='
            'beta1=','beta2='])
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
        #elif opt in ("-p", "--data-path"):
        #    data_path = arg
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
            mode = arg in ('u','1','unet')
        elif opt in ("-p", "--data_path"):
            data_path = str(arg)
        elif opt=='--beta1':
            beta1 = float(arg)
        elif opt=='--beta2':
            beta2 = float(arg)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_path == './cifar-10':
        in_size = 32
    elif 'places' in data_path:
        in_size = 256
    in_shape=(3,in_size,in_size)

    #out_shape=(s.classes,32,32)
    betas=(beta1,beta2)
    weight_path_ending=os.path.join(weight_path,weights_name+'.pth')

    loss_path_ending = os.path.join(weight_path, weights_name + "_" + s.loss_name)

    trainset = load_trainset(data_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize,
                                        shuffle=True, num_workers=2)
 
    print("NETWORK PATH:", weight_path_ending)
    
    #define model
    UNet=None
    try:
        UNet=model() if mode==0 else unet()
        #load weights
        try:
            UNet.load_state_dict(torch.load(weight_path_ending))
            print("Loaded network weights from", weight_path)
        except FileNotFoundError:
            print("Did not find weight files.")
            #sys.exit(2)
    except RuntimeError:
        #if the wrong mode was chosen: try the other one
        UNet=model() if mode==1 else unet()
        #load weights
        try:
            UNet.load_state_dict(torch.load(weight_path_ending))
            print("Loaded network weights from", weight_path)
            #change mode to the correct one
            mode = (mode +1) %2
        except FileNotFoundError:
            print("Did not find weight files.")
            #sys.exit(2)    
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
            "betas": betas,
            "image_loss_weight": image_loss_weight,
            "model":['custom','unet'][mode]
        }
    else:
        #load specified parameters from model_dict
        params=model_dict[weights_name]
        mbsize=params['batch_size']
        betas=params['betas']
        lr=params['lr']
        image_loss_weight=params['image_loss_weight']
        loss_path_ending=params['loss_name']
        #memorize how many epochs already were trained if we continue training
        prev_epochs=params['epochs']+1

    
    #define critic 
    crit = markov_critic().to(device)
    
    #load discriminator weights
    crit_path=weight_path+'/'+weights_name+'_crit.pth'
    try:
        crit.load_state_dict(torch.load(crit_path))
        print('Loaded weights for discriminator from %s'%crit_path)
    except FileNotFoundError:
        print('Initialize new weights for discriminator')
        crit.apply(weights_init_normal)
    #optimizer
    optimizer_g=optim.Adam(UNet.parameters(),lr=lr,betas=betas)
    optimizer_c=optim.Adam(crit.parameters(),lr=lr,betas=betas)
    criterion = nn.BCELoss().to(device)
    #additional gan loss: l1 loss
    l1loss = nn.L1Loss().to(device)
    loss_hist=[]

    

    UNet.train()
    crit.train()
    #convert to black and white image using following weights
    gray = torch.tensor([0.2989 ,0.5870, 0.1140 ])[:,None,None].float()
    ones = torch.ones(mbsize,device=device)
    zeros= torch.zeros(mbsize,device=device)
    # run over epochs
    for e in (range(prev_epochs, prev_epochs + epochs) if not infinite_loop else count(prev_epochs)):
        g_running,c_running=0,0
        #load batches
        #if data_path == './cifar-10':           
        for i,batch in enumerate(trainloader):
            if data_path == './cifar-10':
                (image,_) = batch
            elif 'places' in data_path:
                image = batch
            batch_size=image.shape[0]
            #create ones and zeros tensors
            
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
            if mode==0:
                unet_col=UNet(torch.stack((X,X,X),1)[:,:,0,:,:])
            else:
                unet_col=UNet(X)
            #calculate loss as a function of how good the unet can fool the critic
            fooling_loss = criterion(crit(torch.cat((X,unet_col),dim=1)).mean(dim=(1,2,3)), ones[:batch_size])
            #calculate how close the generated pictures are to the ground truth
            image_loss=l1loss(unet_col,image)
            #combine both losses and weight them
            loss_g=fooling_loss+image_loss_weight*image_loss
            #backpropagation
            loss_g.backward()
            optimizer_g.step()

            #----------------------------------------------------------------------------------------
            ################################## Critic optimization ##################################
            #----------------------------------------------------------------------------------------
            optimizer_c.zero_grad()
            real_crit = crit(torch.cat((X,image),dim=1))
            real_loss=criterion(real_crit, torch.ones(real_crit.shape).to(device))
            #requires no gradient in unet col
            fake_crit = crit(torch.cat((X,unet_col.detach()),dim=1))
            fake_loss=criterion(fake_crit ,torch.zeros(fake_crit.shape).to(device))
            loss_c=.5*(real_loss+fake_loss)
            loss_c.backward()
            optimizer_c.step()

            g_running+=loss_g.item()
            c_running+=loss_c.item()
            loss_hist.append([e,i,loss_g.item(),loss_c.item()])

            #report running loss
            if (i+len(trainloader)*e)%report_freq==report_freq-1:
                print('Epoch %i, batch %i: \tunet loss=%.2e, \tcritic loss=%.2e'%(e+1,i+1,g_running/report_freq,c_running/report_freq))
                g_running=0
                c_running=0

            if s.save_weights and (i+len(trainloader)*e)%save_freq==save_freq-1:
                #save parameters
                try:
                    torch.save(UNet.state_dict(),weight_path_ending)
                    torch.save(crit.state_dict(),crit_path)
                except FileNotFoundError:
                    os.makedirs(weight_path)
                    torch.save(UNet.state_dict(),weight_path_ending)
                    torch.save(crit.state_dict(),crit_path)
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