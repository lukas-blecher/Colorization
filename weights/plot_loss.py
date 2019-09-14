import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter as savgol
import sys, getopt


def main(argv):
    showCritic=False
    try:
        opts, args = getopt.getopt(argv,"p:c",['path='])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ['--path','-p']:
            path=arg
        if opt =='-c':
            showCritic=True
    loss=None
    try:
        e,loss=np.loadtxt(path,unpack=True)
    except OSError:
        e,loss=np.loadtxt('weights/'+path,unpack=True)
    except ValueError:
        try:
            e,i,loss,loss_d=np.loadtxt(path,unpack=True)
        except OSError:
            e,i,loss,loss_d=np.loadtxt('weights/'+path,unpack=True)

    plt.plot(loss,label='raw')
    window=len(loss)//6
    window+=1 if window%2==0 else 0
    filterd_loss=savgol(loss,window,2)
    plt.plot(filterd_loss,label='savgol')
    #N=len(loss)//4
    #plt.plot(np.convolve(loss, np.ones(N)/N, mode='valid'),label='convolve')
    
    plt.xlabel('#batches')
    plt.ylabel('loss')
    plt.title('loss over time')
    #plt.legend()
    plt.show()
    if not type(loss_d) == type('') and showCritic:
        plt.plot(loss_d,label='raw')
        plt.plot(savgol(loss_d,window,2),label='savgol')
        plt.xlabel('#batches')
        plt.ylabel('loss')
        plt.title('critic loss over time')
        plt.show()
if __name__ == '__main__':
    main(sys.argv[1:])