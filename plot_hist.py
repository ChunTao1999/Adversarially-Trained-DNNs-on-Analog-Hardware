import sys
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/models/')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/configs')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/puma-functional-model-v3')
import os
import pdb
from utils import *
import torch
import numpy as np
import argparse 
import attacks
import logging
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.autograd as autograd
#import resnet20_cifar_lc
#import resnet10_cifar_lc
#import resnet20_cifar_interlc
import resnet10_cifar_interlc
import resnet10_cifar_interlc_onestep
import resnet20_cifar
import resnet10_cifar
from mvm_params import *
import matplotlib.pyplot as plt
import pandas as pd


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', action='store', default=None,help='the path to the pretrained model')
    parser.add_argument('--batch-size', default=10, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--limit-test', default=0, type=int,  help='limit the number of batches to test')

    args = parser.parse_args()
    args.pretrained = './log/cifar10/clean-resnet10w1'
    args.batch_size = 1 #keep the batch size 1 to plot a distribution of interlc
    args.limit_test = 1000

    save_path = os.path.join(args.pretrained, "clean", "interlayer_cushion")

    df = pd.read_csv(os.path.join(save_path, 'conv311_linear.csv'.format(args.batch_size*args.limit_test)),sep=',',header=None)
    print(df)
    #print avg, var and std of data
    print(df.mean())
    print(df.var())
    print(df.std())


    #plt.figure()
    df.diff().hist(color='k', alpha=0.5, bins=50)
    plt.ylabel('Frequency')
    plt.xlabel('Interlayer Cushion')
    plt.title('Clean-resnet10w1 conv311_linear')

    plt.show()
    
    
