import torch
import torch.nn.functional as F
import os
from models.unet import unet
from models.model import model
from models.endecoder import generator
from models.richzhang import richzhang
from models.color_unet import color_unet
from models.middle_unet import middle_unet
from torchvision import transforms
from settings import s
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from show import show_colorization
from PIL import Image
from skimage import color

def main(argv):
    
    data_path = None
    input_image = None
    output = None
    weight_path = None
    mode=5
    drop_rate=0
    lab=s.lab
    classification=False
    temp=1
    try:
        opts, args = getopt.getopt(argv,"w:p:b:m:ld:ct:i:o:",["weight-path=", "datapath=",'model=','lab','drop-rate=','input=','output='])
    except getopt.GetoptError as error:
        print(error)
        #print( 'test.py -i <Boolean> -s <Boolean>')
        sys.exit(2)
    print("opts", opts)
    for opt, arg in opts:
        if opt in ("-w", "--weight-path"):
            weight_path = arg
        elif opt in ("--datapath", "-p"):
            data_path = arg
        elif opt=='-m':
            if arg in ('custom','0'):
                mode = 0
            elif arg in ('u','1','unet'):
                mode = 1
            elif arg in ('ende','2'):
                mode = 2
            elif arg in ('richzhang','classende','3'):
                mode = 3
            elif arg in ('colorunet','cu','4'):
                mode = 4
            elif arg in ('mu','5','middle'):
                mode = 5
        elif opt in ('-l','--lab'):
            lab=True
        elif opt in ("-d", "--drop-rate"):
            drop_rate = float(arg) 
        elif opt =='-c':
            classification=True
            lab=True
        elif opt=='-t':
            temp=float(arg)
        elif opt in ('-i','--input'):
            input_image = arg
        elif opt in ('-o','--output'):
            output = arg

    if data_path is None and input_image is None:
        print('Please select an image or folder')
        sys.exit()
    trafo=transforms.Compose([transforms.Grayscale(3 if lab else 1), transforms.Resize((96,96))])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if data_path is not None: 
        dataset = ImageDataset(data_path,lab=lab,pretrafo=trafo)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if input_image is not None:
        img=trafo(Image.open(input_image))
        if lab:
            img=color.rgb2lab(np.asarray(img)/255)[...,:1]-np.array([50])[None,None,:]
        loader = [(transforms.ToTensor()(img)[None,...].float(),input_image)]
        
    
    classes=(340 if classification else 2) if lab else 3

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
        UNet=richzhang(drop_rate,classes)
        zoom=True
    elif mode == 4:
        UNet=color_unet(True,drop_rate,classes)
    elif mode == 5:
        UNet = middle_unet(True,drop_rate,classes)
    #load weights
    try:
        UNet.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Did not find weight files.")
        sys.exit()
    outpath=None
    UNet.to(device)  
    UNet.eval()
    with torch.no_grad():
        for i,(X,name) in enumerate(loader):
            X=X.to(device)
            unet_col=UNet(X)
            col=show_colorization(unet_col,original=X,lab=lab,cl=classification,zoom=zoom,T=temp,return_img=output is not None)
            if output:
                try:
                    fp,f=os.path.split(name)
                except TypeError:
                    fp,f=os.path.split(name[0])
                n,e=f.split('.')
                f='.'.join((n+'_color','png'))
                outpath=output if os.path.isdir(output) or os.path.isdir(os.path.basename(output)) else fp
                Image.fromarray(toInt(col[0])).save(os.path.join(outpath,f))
    if output:
        print('Finished colorization. Go to "%s" to see the colorized version(s) of the image(s)'%os.path.realpath(outpath))



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=True, lab=True, normalize=True,pretrafo=transforms.Resize((96,96))):
        self.path=path
        self.file_list = sorted(list(set(os.listdir(self.path))))
        self.file_list = [f for f in self.file_list if os.path.isfile(os.path.join(path, f))]
        self.transform = transform
        self.resize=pretrafo
        self.lab = lab
        self.norm = normalize
        #need to use transforms.Normalize in future but currently broken
        self.offset=np.array([50,0,0])
        self.range=np.array([1,1,1])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path, self.file_list[item])
        #image = plt.imread(img_name)
        image=np.asarray(self.resize(Image.open(img_name)))/255
        if self.lab:
            if self.norm:
                image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]
            else:
                image = (color.rgb2lab(image)-np.array([50,0,0])[None,None,:])

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image[:1],img_name

def toInt(x):
    return (x*255).astype(np.uint8)

if __name__ == '__main__':
    main(sys.argv[1:])