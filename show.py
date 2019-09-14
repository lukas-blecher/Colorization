import torch
import matplotlib.pyplot as plt
import numpy as np
#from unet.dataset import gta_dataset, city_dataset
from settings import s
#from unet.unet import unet
from skimage.color import lab2rgb as lr
from skimage import exposure

from functions import bins2lab
from scipy.ndimage.interpolation import zoom as scale

#not really beautiful 
def show_colorization(pred,truth=None,original=None,lab=False,cl=False,zoom=False):
    N = 1
    if len(pred.shape)==4:
         N = pred.shape[0]
    M = 1+(1 if not truth is None else 0)+(1 if not original is None else 0)+(2 if lab else 0)
    plt.figure(figsize=(5, N*5/M))
    counter=np.arange(1,1+N*M).reshape(N,M)
    if lab:
        for i in range(N):
            if truth is not None and original is not None:
                gray=original[i].detach().cpu().numpy()
                pn=pred[i].detach().cpu().numpy()
                tn=truth[i].detach().cpu().numpy()                
                if cl:
                    #print(np.bincount(pn.argmax(0).flatten().astype(int)).argmax())
                    pn=bins2lab(pn.argmax(0)).transpose((2,1,0))
                    #print(pn.shape)
                    if zoom:
                        #print(pn.shape)
                        #pn=np.fliplr(np.rot90(pn,-1))
                        pn=scale(pn,(1,4,4))
                    else:
                        #print(pn.shape)
                        pn=np.flip(np.flip((np.rot90(pn,-1,(1,2))),(0,2)),0)
                #print(truth[i].detach().cpu().numpy().min(),truth[i].detach().cpu().numpy().max())
                lab_pred=np.concatenate((100*gray,-np.array([128,128])[:,None,None]+np.array([255,255])[:,None,None]*pn))
                lab_orig=np.concatenate((100*gray,-np.array([128,128])[:,None,None]+np.array([255,255])[:,None,None]*tn))
                #for arr in (lab_orig[0,...],lab_orig[1,...],truth[i].detach().cpu().numpy()[0,...],truth[i].detach().cpu().numpy()[1,...],
                #            lab_pred[0,...],lab_pred[1,...],pred[i].detach().cpu().numpy()[0,...],pred[i].detach().cpu().numpy()[1,...]):
                #    print(arr.min(),arr.max())
                plt.subplot(N,M,counter[i,0])
                if i==0:
                    plt.title('Input image ($L$-channel)')
                plt.axis('off')
                plt.imshow(gray[0],cmap='gray')
                plt.subplot(N,M,counter[i,1])
                if i==0:
                    plt.title('Ground truth')
                plt.axis('off')
                plt.imshow(lr(np.transpose(lab_orig,(1,2,0))))
                plt.subplot(N,M,counter[i,3])
                if i==0:
                    plt.title('Colorization')
                plt.imshow(lr(np.transpose(lab_pred,(1,2,0))))
                plt.axis('off')
                plt.subplot(N,M,counter[i,2])
                if i==0:
                    plt.title('Ground truth $ab$-channels')
                plt.axis('off')
                plt.imshow(exposure.adjust_gamma(lr(np.transpose(np.concatenate((100*np.ones(gray.shape),-np.array([128,128])[:,None,None]+np.array([255,255])[:,None,None]*tn)),(1,2,0))),3,.9))
                plt.subplot(N,M,counter[i,4])
                if i==0:
                    plt.title('Colorization $ab$-channels')
                plt.imshow(exposure.adjust_gamma(lr(np.transpose(np.concatenate((100*np.ones(gray.shape),-np.array([128,128])[:,None,None]+np.array([255,255])[:,None,None]*pn)),(1,2,0))),3,.9))
                plt.axis('off')

    else:
        for i in range(N):
            if truth is None and original is None:
                plt.imshow(pred[i].detach().cpu().numpy())
                plt.axis('off')
            elif original is None:
                #print(truth.shape,pred.shape)
                plt.subplot(N,2,counter[i,0])
                if i==0:
                    plt.title('colorization')
                plt.axis('off')
                plt.imshow(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
                plt.subplot(N,2,counter[i,1])
                if i==0:
                    plt.title('ground truth')
                plt.axis('off')
                plt.imshow(np.transpose(truth[i].detach().cpu().numpy(),(1,2,0)))
            else:
                #print(N,truth.shape,pred.shape,original.shape)
                plt.subplot(N,3,counter[i,0])
                if i==0:
                    plt.title('Input image')
                plt.axis('off')
                plt.imshow(.5*(1+original[i].detach().cpu().numpy()[0]),cmap='gray')
                plt.subplot(N,3,counter[i,1])
                if i==0:
                    plt.title('Ground truth')
                plt.axis('off')
                plt.imshow(np.transpose(truth[i].detach().cpu().numpy(),(1,2,0)))
                plt.subplot(N,3,counter[i,2])
                if i==0:
                    plt.title('colorization')
                plt.imshow(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
                plt.axis('off')
    plt.show()