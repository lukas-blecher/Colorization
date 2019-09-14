import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import color

    
def load_trainset(data_path, lab=False, load_list=False,normalize=True):
    if 'cifar' in  data_path:
        trainset = datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transforms.ToTensor())
        print('cifar loaded')
    elif 'places' in data_path:
        trainset = PlacesDataset(data_path,lab=lab,load_list=load_list,normalize=normalize)
    return trainset
class PlacesDataset(Dataset):
    def __init__(self, path, transform=True, lab=False, classification=False, load_list=True, normalize=True):
        self.path = path
        if load_list:          
            if path[-1] == "/":
                list_path = path[:-1] + '-list.txt'
            else:
                list_path = path + '-list.txt'
            try:
                with open(list_path, 'r') as f:
                    self.file_list = [line.rstrip('\n') for line in f]
            except FileNotFoundError:
                print("List not found, new list initialized")
                self.file_list = sorted(list(set(os.listdir(path))))
                with open(list_path, 'w') as f:
                    for s in self.file_list:
                        f.write(str(s) + '\n')
        else:
            self.file_list = sorted(list(set(os.listdir(path))))
        
        self.transform = transform
        self.lab = lab
        self.bins = classification
        self.norm = normalize
        #need to use transforms.Normalize in future but currently broken
        self.offset=-np.array([0,128,128])
        self.range=np.array([100,255,255])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)/255
        if self.lab:
            if self.norm:
                image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]
            else:
                image = (color.rgb2lab(image)-np.array([50,0,0])[None,None,:])

            if self.bins:
                image[:,:,1:]=ab2bins(image[:,:,1:])
        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image



#image preprocessing
binmap=torch.load('resources/binmap.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
#distance matrix
def dist_mat(X,Y):
    return -2 * X@Y.T + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, None]
def torch_dist_mat(X,Y):
    #print(X.shape,Y.shape,torch.matmul(X, torch.transpose(Y,1,2).double()).shape)
    return -2 * torch.matmul(X, torch.transpose(Y,1,2).double()) + torch.sum(Y**2,dim=2).double()[:,None] +torch.unsqueeze(torch.sum(X**2, dim=2),2)  
bins=np.load('resources/norm_bins.npy')
tbins=torch.from_numpy(bins).to('cuda' if torch.cuda.is_available() else 'cpu')
def old_ab2bins(image):
    #takes image with only ab channels and returns the 
    shape=image.shape
    im_size=shape[2 if len(shape)==4 else 1]
    mbsize = shape[0] if len(shape)==4 else 1
    if type(image)==torch.Tensor:
        bin_rep = torch_dist_mat(tbins,image.reshape(mbsize,-1,2)).argmin(1).reshape(mbsize,1,im_size,-1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
    else:
        image=np.array(image)
        bin_rep = dist_mat(bins,image.reshape(-1,2)).argmin(0).reshape(mbsize,im_size,-1,1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]

def ab2bins(image):
    #takes image with only ab channels and returns the 
    shape=image.shape
    im_size=shape[2 if len(shape)==4 else 1]
    mbsize = shape[0] if len(shape)==4 else 1
    if type(image)==torch.Tensor:
        #bin_rep = torch_dist_mat(tbins,image.reshape(mbsize,-1,2)).argmin(1).reshape(mbsize,1,im_size,-1)
        ind=torch.round(image/10).long()
        bin_rep=binmap[ind[:,0,:,:],ind[:,1,:,:]]
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
    else:
        image=np.array(image)
        bin_rep = dist_mat(bins,image.reshape(-1,2)).argmin(0).reshape(mbsize,im_size,-1,1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
        
    


def bins2lab(bin_rep,L=None):
    #takes bins representation of an image and returns rgb if Lightness is provided. Else only ab channel
    mbsize=1 if len(bin_rep.shape)<=3 else bin_rep.shape[0]
    size=bin_rep.shape[2] if len(bin_rep.shape)==4 else bin_rep.shape[1]
    #print(bin_rep.shape,bins[:3],mbsize,size)
    ab=bins[0][bin_rep.flatten()].reshape(mbsize,size,-1,2)
    if not L is None:
        ab=np.concatenate((L.reshape(mbsize,size,-1,1),ab),3)
    
    if len(bin_rep.shape)<=3:
        ab=ab[0]
    return ab

class softCossEntropyLoss(nn.Module):
    def __init__(self,weights=None,device=torch.device('cpu')):
        '''
        weights shape: (Q,)
        '''
        super(softCossEntropyLoss,self).__init__()
        self.weights=(torch.Tensor(weights).to(device).double() if type(weights)==np.ndarray else weights.to(device).double()) if not weights is None else None
        
    def forward(self,output,labels):
        '''
        output shape: (batch_size,channels,dim1,dim2)
        labels shape: (batch_size,channels,dim1,dim2) <-- from one hot encoded gaussian filtered 
        returns multinomial cross entropy loss
        '''
        if self.weights is None:
            return -torch.sum(torch.sum(labels*torch.log(output),1),(1,2))        
        else:
            return -torch.sum((self.weights[None,:,None,None]*labels).sum(1)*torch.sum(labels*torch.log(output),1),(1,2))