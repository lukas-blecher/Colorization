import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import color

    
def load_trainset(data_path,train=True, lab=True, load_list=False,normalize=True):
    if 'cifar' in  data_path:
        trainset = datasets.CIFAR10(root=data_path, train=train,
                                        download=True, transform=transforms.ToTensor())

    elif 'places-big' in data_path:
        trainset = BigPlacesDataset(data_path,lab=lab,train=train,load_list=load_list,normalize=normalize)

    elif 'places' in data_path:
        trainset = PlacesDataset(data_path,lab=lab,load_list=load_list,normalize=normalize)
    elif 'stl' in data_path:
        trainset = ImageSTL(data_path, train=train, lab=lab, download=True, transform=transforms.ToTensor())
        
    return trainset

class PlacesDataset(Dataset):
    def __init__(self, path, transform=True, lab=False, load_list=True, normalize=True):
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
        self.tf = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomCrop((224,224)),
                                              transforms.ColorJitter(hue=.025, saturation=.15),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        self.lab = lab
        self.norm = normalize
        #need to use transforms.Normalize in future but currently broken
        self.offset=np.array([50,0,0])
        self.range=np.array([1,1,1])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)
        image = self.tf(image).numpy().transpose((1,2,0))
        if self.lab:
            if self.norm:
                image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]
            else:
                image = (color.rgb2lab(image)-np.array([50,0,0])[None,None,:])

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image

class BigPlacesDataset(Dataset):
    def __init__(self, path, transform=True, train=True, lab=False, load_list=True, normalize=True):
        self.path = path
        if train:
            self.list_path = os.path.join(path, 'train.txt')
        else:
            self.list_path = os.path.join(path, 'val.txt')
            
        with open(self.list_path, 'r') as f:
            self.file_list = [line.rstrip('\n') for line in f]
        
        self.transform = transform
        self.tf = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomCrop((224,224)),
                                        transforms.ColorJitter(hue=.025, saturation=.15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        self.lab = lab
        self.norm = normalize
        self.offset=np.array([50,0,0])
        self.range=np.array([1,1,1])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)
        image = self.tf(image).numpy().transpose((1,2,0))
        if self.lab:
            if self.norm:
                image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]
            else:
                image = (color.rgb2yuv(image)-np.array([50,0,0])[None,None,:])

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image

class STL(Dataset):
    def __init__(self, path, transform=True, train=True, lab=False,download=False):
        

        if download:
            #use torchvision to download
            _=datasets.STL10(path,download=True)
        self.path = os.path.join(path,'stl10_binary')
        if train:
            self.data = np.concatenate((self.load_images(os.path.join(self.path, 'train_X.bin')),self.load_images(os.path.join(self.path, 'unlabeled_X.bin'))))
        else:
            self.data = self.load_images(os.path.join(self.path, 'test_X.bin'))
            
       
        self.transform = transform
        self.tf = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ColorJitter(hue=.025, saturation=.15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        self.lab = lab
        self.offset=np.array([50,0,0])
        self.range=np.array([1,1,1])
    def __len__(self):
        return len(self.data)

    def load_images(self,path):
        #from stl10_input.py
        with open(path, 'rb') as f:
            return np.fromfile(f, dtype=np.uint8).reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1))

    def __getitem__(self, item):
        
        image = self.data[item]
        image = self.tf(image).numpy().transpose((1,2,0))
        if self.lab:
            image = (color.rgb2yuv(image)-np.array([50,0,0])[None,None,:])

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image

class ImageSTL(Dataset):
    def __init__(self, path, download=False, transform=True,train=True, lab=True, load_list=True, normalize=True):
        if download:
            tvstl=datasets.STL10(path,download=True)
            from tqdm import tqdm_notebook
            from PIL import Image
            for i in tqdm(range(len(tvstl.data))):
                Image.fromarray(tvstl.data[i]).save(os.path.join(path,'%s/stl_%06d.png'%(('test','train')[int(train)],i)))
        self.path = os.path.join(path,'train' if train else 'test')
        if load_list:          
            if path[-1] == "/":
                list_path = self.path[:-1] + '-list.txt'
            else:
                list_path = self.path + '-list.txt'
            try:
                with open(list_path, 'r') as f:
                    self.file_list = [line.rstrip('\n') for line in f]
            except FileNotFoundError:
                print("List not found, new list initialized")
                self.file_list = sorted(list(set(os.listdir(self.path))))
                with open(list_path, 'w') as f:
                    for s in self.file_list:
                        f.write(str(s) + '\n')
        else:
            self.file_list = sorted(list(set(os.listdir(self.path))))
        
        self.transform = transform
        self.tf = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ColorJitter(hue=.025, saturation=.15),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
        self.lab = lab
        self.norm = normalize
        #need to use transforms.Normalize in future but currently broken
        self.offset=np.array([50,0,0])
        self.range=np.array([1,1,1])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)*255
        image = self.tf(image.astype(np.uint8)).numpy().transpose((1,2,0))
        if self.lab:
            image = (color.rgb2lab(image)-np.array([50,0,0])[None,None,:])

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image

#image preprocessing
#binmap=torch.load('resources/binmap.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
#distance matrix
def dist_mat(X,Y):
    return -2 * X@Y.T + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, None]
def torch_dist_mat(X,Y):
    #print(X.shape,Y.shape,torch.matmul(X, torch.transpose(Y,1,2).double()).shape)
    return -2 * torch.matmul(X, torch.transpose(Y,1,2).double()) + torch.sum(Y**2,dim=2).double()[:,None] +torch.unsqueeze(torch.sum(X**2, dim=2),2)  
#bins=np.load('resources/norm_bins.npy')
#tbins=torch.from_numpy(bins).to('cuda' if torch.cuda.is_available() else 'cpu')
tbins = torch.unsqueeze(torch.load('resources/lab_bins.pt'),0).to('cuda' if torch.cuda.is_available() else 'cpu')
nbins = torch.unsqueeze(torch.load('resources/lab_bins.pt'),0).numpy()
def ab2bins(image,bins=None):
    #takes image with only ab channels and returns the 
    shape=image.shape
    im_size=shape[2 if len(shape)==4 else 1]
    mbsize = shape[0] if len(shape)==4 else 1
    if type(image)==torch.Tensor:
        bs=tbins if bins is None else bins
        bin_rep = torch_dist_mat(bs,image.reshape(mbsize,-1,2)).argmin(1).reshape(mbsize,1,im_size,-1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
    else:
        image=np.array(image)
        bs=nbins if bins is None else bins
        bin_rep = dist_mat(bs[0],image.reshape(-1,2)).argmin(0).reshape(mbsize,im_size,-1,1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]

def decap_ab2bins(image):
    #takes image with only ab channels and returns the 
    shape=image.shape
    im_size=shape[2 if len(shape)==4 else 1]
    mbsize = shape[0] if len(shape)==4 else 1
    if type(image)==torch.Tensor:
        #bin_rep = torch_dist_mat(tbins,image.reshape(mbsize,-1,2)).argmin(1).reshape(mbsize,1,im_size,-1)
        ind=torch.round(image/10).long()
        #bin_rep=binmap[ind[:,0,:,:],ind[:,1,:,:]]
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
    else:
        image=np.array(image)
        bin_rep = dist_mat(nbins,image.reshape(-1,2)).argmin(0).reshape(mbsize,im_size,-1,1)
        if len(shape)==4:
            return bin_rep
        else:
            return bin_rep[0]
        
    


def bins2ab(bin_rep,L=None):
    #takes bins representation of an image and returns rgb if Lightness is provided. Else only ab channel
    mbsize=1 if len(bin_rep.shape)<=3 else bin_rep.shape[0]
    size=bin_rep.shape[2] if len(bin_rep.shape)==4 else bin_rep.shape[1]
    #print(bin_rep.shape,bins[:3],mbsize,size)
    ab=nbins[0][bin_rep.flatten()].reshape(mbsize,size,-1,2)
    if not L is None:
        ab=np.concatenate((L.reshape(mbsize,size,-1,1),ab),3)
    
    if len(bin_rep.shape)<=3:
        ab=ab[0]
    return ab

def ab_from_distr(distr,T=1,Y=None,bins=None):
    bins=tbins.cpu() if bins is None else bins
    distr=torch.Tensor(distr)
    temp_dist=torch.nn.functional.softmax(torch.log(distr)/T,1)
    img=torch.matmul(temp_dist.transpose(1,2).transpose(2,3),bins.float())
    #img=functions.bins2yuv(sample.detach().numpy(),Lten.numpy())
    if not Y is None:
        img=torch.cat((Lten.transpose(1,2).transpose(2,3).float(),img),3)
    return img.numpy()

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