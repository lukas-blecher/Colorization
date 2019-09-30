import torch
import matplotlib.pyplot as plt
import numpy as np
from settings import s
from skimage.color import lab2rgb
from skimage import exposure

from functions import bins2ab,ab2bins,ab_from_distr
from scipy.ndimage.interpolation import zoom as scale

#avoid negative values
def lr(img):
    img=lab2rgb(img)
    img=np.clip(img,0,1)
    return img

def show_colorization(pred,truth=None,original=None,lab=False,cl=False,zoom=False,T=1,return_img=False):
    N = 1
    if len(pred.shape)==4:
         N = pred.shape[0]
    M = 1+(1 if not truth is None else 0)+(1 if not original is None else 0)+(2 if lab else 0)
    plt.figure(figsize=(5, N*5/M))
    counter=np.arange(1,1+N*M).reshape(N,M)
    img_list=[]
    if lab:
        for i in range(N):
            if original is not None:
                gray=original[i].detach().cpu().numpy()
                pn=pred[i].detach().cpu().numpy()
                if truth is not None: 
                    tn=truth[i].detach().cpu().numpy()                
                if cl:
                    pn=ab_from_distr(pn[None,...],T)
                    if zoom:
                        pn=scale(pn,(1,4,4,1))
                    
                
                if cl:
                    lab_pred=np.concatenate((50+gray.transpose((1,2,0)),pn[0]),2)
                else:
                    lab_pred=np.concatenate((50+gray,pn)).transpose(1,2,0)
                    
                if truth is not None: 
                    lab_orig=np.concatenate((50+gray,tn))
                plt.subplot(N,M,counter[i,0])
                if i==0:
                    plt.title('Input image ($L$-channel)')
                plt.axis('off')
                plt.imshow(gray[0],cmap='gray')
                plt.subplot(N,M,counter[i,1])
                if truth is not None: 
                    if i==0:
                        plt.title('Ground truth')
                    plt.axis('off')
                    plt.imshow(lr(np.transpose(lab_orig,(1,2,0))))
                    plt.subplot(N,M,counter[i,3])
                if i==0:
                    plt.title('Colorization')
                plt.imshow(lr(lab_pred))
                plt.axis('off')
                plt.subplot(N,M,counter[i,2])
                if truth is not None: 
                    if i==0:
                        plt.title('Ground truth $ab$-channels')
                    plt.axis('off')
                    plt.imshow(exposure.adjust_gamma(lr(np.transpose(np.concatenate((100*np.ones(gray.shape),tn)),(1,2,0))),3,.9))
                    plt.subplot(N,M,counter[i,4])
                if i==0:
                    plt.title('Colorization $ab$-channels')
                if cl:
                    plt.imshow(exposure.adjust_gamma(lr(np.concatenate((np.ones(gray.T.shape),pn[0]),2)),3,.9))
                else:
                    plt.imshow(exposure.adjust_gamma(lr(np.concatenate((np.ones(gray.shape),pn)).transpose(1,2,0)),3,.9))                
                plt.axis('off')
                if return_img:
                    img_list.append(lr(lab_pred))
    else:
        for i in range(N):
            if truth is None and original is None:
                plt.imshow(pred[i].detach().cpu().numpy())
                plt.axis('off')
            elif original is None:
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
            if return_img:
                img_list.append(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
    if return_img:
        return img_list
    else:
        plt.show()