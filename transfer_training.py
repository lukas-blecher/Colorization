import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from itertools import count
import sys, getopt
from models.stlclassifier import Classifier
from settings import s
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import json
from functions import load_trainset
from skimage import color
from torchvision.models import alexnet

def main(argv):
    # setting argument defaults
    mbsize = s.batch_size
    report_freq=s.report_freq
    weight_path=s.weights_path
    weights_name='Metric'
    lr=s.learning_rate
    save_freq = s.save_freq
    epochs = s.epochs
    beta1,beta2=s.betas
    infinite_loop=s.infinite_loop
    data_path = s.data_path
    load_list=s.load_list
    help='test.py -b <int> -p <string> -r <int> -w <string>'
    try:
        opts, args = getopt.getopt(argv,"he:b:r:w:l:s:n:p:i:",
            ['epochs=',"mbsize=","report-freq=",'weight-path=', 'lr=','save-freq=','weight-name=','data_path='
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
        elif opt in ("-p", "--data_path"):
            data_path = str(arg)
        elif opt=='--beta1':
            beta1 = float(arg)
        elif opt=='--beta2':
            beta2 = float(arg)
        elif opt =='--load-list':
            load_list=not load_list
        
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = 96
    #out_shape=(s.classes,32,32)
    betas=(beta1,beta2)
    weight_path_ending=os.path.join(weight_path,weights_name+'.pt')

    trainset = datasets.STL10(data_path,split='test',transform=transforms.Compose([transforms.Resize(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0.5,.5,.5], [0.5,.5,.5])
                                                                 ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize, shuffle=True, num_workers=0)
    alex=alexnet()
    alex.load_state_dict(torch.load('weights/alexnet.pth'))
    alex.to(device)
    alex.eval()
    print("NETWORK PATH:", weight_path_ending)
    #define output channels of the model
    classifier = nn.Sequential(nn.Linear(1000,512),nn.ReLU(),nn.Linear(512,10))
    #load weights
    try:
        classifier.load_state_dict(torch.load(weight_path_ending))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Initialize new weights for the generator.")
        #sys.exit(2)
    
    classifier.to(device)
    #optimizer
    optimizer=optim.Adam(classifier.parameters(),lr=lr,betas=betas)

    criterion = nn.CrossEntropyLoss().to(device)

    loss_hist=[]
    
    classifier.train()
    #crit.train()
    # run over epochs
    for e in (range(epochs) if not infinite_loop else count(epochs)):
        g_running=0
        #load batches          
        for i,(X,c) in enumerate(trainloader):

            X=X.to(device)
            c=c.to(device)
            #----------------------------------------------------------------------------------------
            ################################### Model optimization ##################################
            #----------------------------------------------------------------------------------------
            #clear gradients
            optimizer.zero_grad()
            with torch.no_grad():
                class_out=alex(X)

            model_out=classifier(class_out)

            loss=criterion(model_out,c)
            
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


if __name__ == '__main__':
    main(sys.argv[1:])