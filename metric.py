import sys, getopt
import numpy as np
import torch
from functions import load_trainset,ab2bins,bins2ab,ab_from_distr,normalize
from torch.utils.data import dataloader
from skimage import color
import matplotlib.pyplot as plt
from torchvision.models import alexnet
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from models.color_unet import color_unet
from models.unet import unet
from models.middle_unet import middle_unet

def main(argv):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gray = torch.tensor([0.2989 ,0.5870, 0.1140])[:,None,None].float()

    alex = alexnet().to(device)
    alex.load_state_dict(torch.load('weights/alexnet.pth'))
    alex.eval()
    classifier_path = 'weights/alexNorm.pt'
    weight_path = None
    mbsize = 16
    col_space = None
    class_net = False
    mode = None
    drop_rate = 0
    T = 1
    classes = 3
    stl_path = '../stl-10'
    try:
        opts, args = getopt.getopt(argv,"b:w:p:cm:d:t:s:",
            ["mbsize=",'weight-path=','classifier-path=','c_space=','mode=','drop_rate=','temperature=','stl='])
    except getopt.GetoptError:
        sys.exit(2)
    print("opts" ,opts)
    for opt, arg in opts:
        if opt in ("-b", "--mbsize"):
            mbsize = int(arg)
        elif opt in ("-w", "--weight-path"):
            weight_path = str(arg)
        elif opt in ("-p", "--classifier-path"):
            classifier_path = str(arg)
        elif opt in ("-s","--c_space"):
            col_space = str(arg)
        elif opt =='-c':
            class_net = True
        elif opt == '-m':
            if arg in ('u','0','unet'):
                mode = 0
            elif arg in ('color','1','cu'):
                mode = 1
            elif arg in ('mu','2','middle'):
                mode = 2
        elif opt in ("-d", "--drop_rate"):
            drop_rate = float(arg)
        elif opt in ('-t', '--temperature'):
            T = float(arg)
        elif opt in ('--stl'):
            stl_path=arg
    if col_space == None:
        print('Specify color space')
        sys.exit(2)

    if class_net:
        if col_space == 'yuv':
            classes = 42
        elif col_space == 'lab':
            classes = 340
    if mode == 0:
        Col_Net = unet(True, drop_rate, classes).to(device)
    elif mode == 1:
        Col_Net = color_unet(True, drop_rate, classes).to(device)
    elif mode == 2:
        Col_Net = middle_unet(True, drop_rate, classes).to(device)

    Col_Net.load_state_dict(torch.load(weight_path, map_location=device))
    Col_Net.eval()

    classifier=nn.Sequential(nn.Linear(1000,512),nn.ReLU(),nn.Linear(512,10)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    testset = datasets.STL10(stl_path,split='train',transform=transforms.Compose([transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=mbsize, shuffle=True, num_workers=0)

    print(pseudo_metric(testloader, col_space, Col_Net, classifier, alex, T))


def check_colorization(batch, labels, col_space, Col_Net, classifier, alex, T=1):
    gray = torch.tensor([0.2989 ,0.5870, 0.1140])[:,None,None].float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = batch.shape[0]

    rgb, yuv, lab = [False]*3
    if col_space == 'rgb':
        rgb =True
    elif col_space == 'lab':
        lab = True
    elif col_space == 'yuv':
        yuv = True

    if yuv or lab:
        Y = batch[:,:1,:,:]
    #build gray image
    if lab or yuv:
        X=torch.unsqueeze(batch[:,0,:,:],1).to(device) #set X to the Lightness of the image
        batch=batch[:,1:,:,:].to(device) #image is a and b channel
    else:
        #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
        X=(batch.clone()*gray).sum(1).to(device).view(-1,1,96,96)
        batch=batch.float().to(device)
    
    if yuv:
        normalize(X,(.5,),(.5,),True)
    elif lab:
        normalize(X,(50,),(1,),True)
    
    #do colorization
    col_batch = Col_Net(X).detach().cpu()
    classes = col_batch.shape[1]
    
    #construct rgb image form network output
    if classes != 3:
        #for lab/yuw and GAN net
        if classes == 2:
            if yuv or lab:
                col_batch = torch.cat((Y, col_batch), 1).numpy().transpose((0,2,3,1))
        #for -c
        elif classes > 3:
            if yuv:
                col_batch = UV_from_distr(col_batch, T, Y)
            elif lab:
                col_batch = ab_from_distr(col_batch, T, Y)
        if yuv:
            rgb_batch = (color.yuv2rgb(col_batch))
        elif lab: #lab2rgb doesn't support batches
            rgb_batch=np.zeros_like(col_batch)
            for k in range(len(col_batch)):
                rgb_batch[k] = color.lab2rgb(col_batch[k])

        rgb_batch = torch.tensor(np.array(rgb_batch).transpose((0,3,1,2))).float()
    else:
        rgb_batch = col_batch
    
    rgb_batch = F.interpolate(rgb_batch, size = (224,224))
    normalize(rgb_batch,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225],True)
    
    with torch.no_grad():
        class_out = alex(rgb_batch.to(device))
        pred = classifier(class_out)

    correct_pred = (pred.argmax(1)==labels.to(device)).sum().cpu().item()
    
    return correct_pred, batch_size

def pseudo_metric(testloader, col_space, Col_Net, classifier, alex, T=1):
    im_sum = 0
    im_corr = 0
    for img in tqdm(iter(testloader)):
        pic,label = img
        if col_space == 'yuv':
            pic = pic.detach().numpy()
            pic = color.rgb2yuv(pic.transpose(0,2,3,1))
            pic = pic.transpose(0,3,1,2)
            pic = torch.tensor(pic).float()
        if col_space == 'lab':
            pic = pic.detach().numpy()
            pic = color.rgb2lab(pic.transpose(0,2,3,1))
            pic = pic.transpose(0,3,1,2)
            pic = torch.tensor(pic).float()
        corr, batchsize = check_colorization(pic, label, col_space, Col_Net, classifier, alex, T)
        im_corr += corr
        im_sum += batchsize
    return im_corr/im_sum

if __name__ == '__main__':
    main(sys.argv[1:])