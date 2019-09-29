import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from itertools import count
import sys, getopt
from models.discriminator import critic
from models.richzhang import richzhang as generator
from models.unet import unet
from models.color_unet import color_unet
from models.middle_unet import middle_unet
from settings import s
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import json
from functions import load_trainset
from functions import ab2bins
from functions import softCossEntropyLoss
from skimage import color
from scipy.ndimage.interpolation import zoom

def main(argv):
    # setting argument defaults
    mbsize = s.batch_size
    report_freq=s.report_freq
    weight_path=s.weights_path
    weights_name=s.weights_name
    lr=s.learning_rate
    save_freq = s.save_freq
    mode=3
    image_loss_weight=s.image_loss_weight
    epochs = s.epochs
    beta1,beta2=s.betas
    infinite_loop=s.infinite_loop
    data_path = s.data_path
    drop_rate = 0
    lab = True
    weighted_loss=True
    weight_lambda=.25
    load_list=s.load_list
    help='test.py -b <batch size> -e <amount of epochs to train. standard: infinite> -r <report frequency> -w <path to weights folder> \
            -n <name> -s <save freq.> -l <learning rate> -p <path to data set> -d <dropout rate> -m <mode: differnet models> --beta1 <beta1 for adam>\
            --beta2 <beta2 for adam> --lab <No argument. If used lab colorspace is cused> --weighted <No argument. If used *NO* class weights are used> \
            --lambda <hyperparameter for class weights>'
    try:
        opts, args = getopt.getopt(argv,"he:b:r:w:l:s:n:p:d:i:m:",
            ['epochs=',"mbsize=","report-freq=",'weight-path=', 'lr=','save-freq=','weight-name=','data_path=','drop_rate='
            'beta1=','beta2=','lab','image-loss-weight=','weighted','mode=','lambda='])
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
        elif opt in ("-p", "--data_path"):
            data_path = str(arg)
        elif opt in ("-d", "--drop_rate"):
            drop_rate = float(arg)
        elif opt=='-m':
            if arg in ('richzhang','0','ende'):
                mode = 0
            elif arg in ('u','1','unet'):
                mode = 1
            elif arg in ('color','2','cu'):
                mode = 2
            elif arg in ('mu','3','middle'):
                mode = 3
        elif opt=='--beta1':
            beta1 = float(arg)
        elif opt=='--beta2':
            beta2 = float(arg)
        elif opt=='--lab':
            lab=True
        elif opt =='--weighted':
            weighted_loss= not weighted_loss
        elif opt =='--load-list':
            load_list=not load_list
        elif opt =='--lambda':
            weight_lambda = float(arg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=None
    in_size = 256
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

    trainset = load_trainset(data_path,lab=lab,normalize=False,load_list=load_list)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize,
                                        shuffle=True, num_workers=2 if dataset in (0,1) else 0)
 
    print("NETWORK PATH:", weight_path_ending)
    #define output channels of the model
    classes = 150
    #define model
    if mode == 0:
        classifier = generator(drop_rate,classes)
    elif mode == 1:
        classifier = unet(True,drop_rate,classes)
    elif mode == 2:
        classifier = color_unet(True,drop_rate,classes)
    elif mode == 3:
        classifier = middle_unet(True,drop_rate,classes)
    #load weights
    try:
        classifier.load_state_dict(torch.load(weight_path_ending))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Initialize new weights for the generator.")
        #sys.exit(2)
    
    classifier.to(device)

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
            "image_loss_weight": image_loss_weight,
            "weighted_loss":weighted_loss,
            "model":'classification '+['richzhang','U-Net','color U-Net','middle U-Net'][mode]
        }
    else:
        #load specified parameters from model_dict
        params=model_dict[weights_name]
        #mbsize=params['batch_size']
        betas=params['betas']
        #lr=params['lr']
        lab=params['lab']
        image_loss_weight=params['image_loss_weight']
        weighted_loss=params['weighted_loss']
        loss_path_ending=params['loss_name']
        #memorize how many epochs already were trained if we continue training
        prev_epochs=params['epochs']+1

    #optimizer
    optimizer=optim.Adam(classifier.parameters(),lr=lr,betas=betas)
    class_weight_path='resources/class-weights.npy'
    if weighted_loss:
        weights=np.load(class_weight_path)
        if dataset==0:
            class_weight_path='resources/cifar-lab-class-weights.pt'
            weights=torch.load(class_weight_path).numpy()
        elif dataset==2:
            if weight_lambda:
                class_weight_path = 'resources/probdist_lab.pt'
                prob_dict = torch.load(class_weight_path)
                prob = np.array(list(prob_dict.values()))
                weights = 1/((1 - weight_lambda)*prob/prob.sum() + weight_lambda/classes)
            else:
                class_weight_path = 'resources/class-weights-lab150-stl.pt'
                weights = torch.load(class_weight_path)
            
        print('Class-weights loaded from ' + class_weight_path) 
    criterion = softCossEntropyLoss(weights=weights,device=device) if weighted_loss else softCossEntropyLoss(weights=None,device=device) 
    loss_hist=[] 
    soft_onehot = torch.load('resources/smooth_onehot150.pt',map_location=device)
    
    classifier.train() 
    # run over epochs
    for e in (range(prev_epochs, prev_epochs + epochs) if not infinite_loop else count(prev_epochs)):
        g_running=0
        #load batches          
        for i,batch in enumerate(trainloader):
            
            if dataset == 0: #cifar 10
                (image,_) = batch
            elif dataset in (1,2): #places
                image = batch
                
            #batch_size=image.shape[0]
            if dataset == 0: #cifar/stl 10
                image=np.transpose(image,(0,2,3,1))
                image=np.transpose(color.rgb2lab(image),(0,3,1,2))
                image=torch.from_numpy((image-np.array([50,0,0])[None,:,None,None])).float()
            
            X=image[:,:1,:,:].to(device) #set X to the Lightness of the image
            image=image[:,1:,:,:].to(device) #image is a and b channel
            
            #----------------------------------------------------------------------------------------
            ################################### Model optimization ##################################
            #----------------------------------------------------------------------------------------
            #clear gradients
            optimizer.zero_grad()
            #softmax activated distribution
            model_out=classifier(X).double()
            #create bin coded verion of ab ground truth
            binab=ab2bins(image.transpose(1,3).transpose(1,2))
            if mode==0: 
                binab=F.interpolate(binab.float(),scale_factor=(.25,.25)).long() 
            binab=torch.squeeze(binab,1) 
            binab=soft_onehot[:,binab].transpose(0,1).double()
            #calculate loss  
            loss=criterion(model_out,binab).mean(0)
            
            loss.backward()
            optimizer.step()

            g_running+=loss.item()
            loss_hist.append([e,loss.item()])

            #report running loss
            if (i+len(trainloader)*e)%report_freq==report_freq-1:
                print('Epoch %i, batch %i: \tloss=%.2e'%(e+1,i+1,g_running/report_freq))
                g_running=0


            if s.save_weights and (i+len(trainloader)*e)%save_freq==save_freq-1:
                #save parameters
                try:
                    torch.save(classifier.state_dict(),weight_path_ending)
                    #torch.save(crit.state_dict(),crit_path)
                except FileNotFoundError:
                    os.makedirs(weight_path)
                    torch.save(classifier.state_dict(),weight_path_ending)
                    #torch.save(crit.state_dict(),crit_path)
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