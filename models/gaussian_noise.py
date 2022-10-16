import sys
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/configs')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/puma-functional-model-v3')

import logging
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from pytorch_mvm_class_v3 import Conv2d_mvm, Linear_mvm, NN_model
from custom_normalization_functions import custom_3channel_img_normalization_with_dataset_params
import pdb
import numpy as np
from torch.autograd import Variable
                                                        
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    #def __init__(self, sigma: float, is_relative_detach=True):
    def __init__(self, sigma: float, mean=0):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        #self.is_relative_detach = is_relative_detach
        #self.noise = torch.tensor(0).cuda()
        #self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        #relative sigma
        #scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        #sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale

        noise = Variable(x.data.new(x.size()).normal_(self.mean, self.sigma))
        #pdb.set_trace()
        x = x + noise
        return x 

class GaussianNoise_Mean(nn.Module):
    def __init__(self, sigma: float, mean: float):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        noise = Variable(x.data.new(x.size()).normal_(self.mean, self.sigma))
        #pdb.set_trace()
        x = x + noise
        return x 

#class GaussianNoise(nn.Module):
#    def __init__(self, stddev=0.1):
#        super().__init__()
#        self.stddev = stddev
#
#    def forward(self, x):
#        self.stddev = 0.1
#        #self.stddev = 0
#        noise = torch.autograd.Variable(torch.randn(x.size()).cuda()*self.stddev)
#        x = x + noise
#        return x
#pdb.set_trace()
#see if noise is too big
#class GaussianNoise(object):
#    def __init__(self, mean=0., std=1.):
#        self.std = std
#        self.mean = mean
#        
#    def __call__(self, tensor):
#        return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean
#    
#    def __repr__(self):
#        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
