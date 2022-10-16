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
from gaussian_noise import GaussianNoise, GaussianNoise_Mean
import pdb
from collections import OrderedDict
import numpy as np
                                                        
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.classes = args.classes 
        self.inflate = args.inflate        
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.use_custom_norm = args.custom_norm
        self.store_act = args.store_act
        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        self.resconv11  =nn.Sequential(
                                nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                nn.BatchNorm2d(16*self.inflate),)
        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(
                            nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(32*self.inflate),)
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(
                            nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(64*self.inflate),)
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.apply(_weights_init)
            
  
    def forward(self, x):
        if self.use_custom_norm:
            out = self.custom_norm(x)
            #print('using custom norm')
        else:
            out = x
        convout = {}
        #---- Layer 0
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)       
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #---- Layer 3.1.1
        out = self.conv311(out)
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        if self.store_act:
            return out
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out


class Model_NoisetoOneConv(nn.Module):
    def __init__(self, args):
        super(Model_NoisetoConv, self).__init__()
        
        self.classes = args.classes 
        self.inflate = args.inflate        
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.use_custom_norm = args.custom_norm
        self.store_act = args.store_act
        self.noise_layer = args.noise_layer
        self.noise_sigma = args.noise_sigma

        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 0:
            self.noise0 = GaussianNoise(sigma=self.noise_sigma)
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        self.resconv11  =nn.Sequential(
                                nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                nn.BatchNorm2d(16*self.inflate),)
        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 1:
            self.noise111 = GaussianNoise(sigma=self.noise_sigma)
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 2:
            self.noise112 = GaussianNoise(sigma=self.noise_sigma)
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(
                            nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(32*self.inflate),)
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        if self.noise_layer == 3:
            self.noise211 = GaussianNoise(sigma=self.noise_sigma)
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 4:
            self.noise212 = GaussianNoise(sigma=self.noise_sigma)
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(
                            nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(64*self.inflate),)
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        if self.noise_layer == 5:
            self.noise311 = GaussianNoise(sigma=self.noise_sigma)
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 6:
            self.noise312 = GaussianNoise(sigma=self.noise_sigma)
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 7:
            self.noise321 = GaussianNoise(sigma=self.noise_sigma)
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        if self.noise_layer == 8:
            self.noise322 = GaussianNoise(sigma=self.noise_sigma)
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.apply(_weights_init)
            
  
    def forward(self, x):
        if self.use_custom_norm:
            out = self.custom_norm(x)
            #print('using custom norm')
        else:
            out = x
        convout = {}
        #---- Layer 0
        out = self.conv0(out)
        if self.noise_layer == 0:
            out = self.noise0(out)
        convout['conv0'] = out.clone()
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        if self.noise_layer == 1:
            out = self.noise111(out)
        convout['conv111'] = out.clone()
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        if self.noise_layer == 2:
            out = self.noise112(out)
        convout['conv112'] = out.clone()
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)       
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        if self.noise_layer == 3:
            out = self.noise211(out)
        convout['conv211'] = out.clone()
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        if self.noise_layer == 4:
            out = self.noise212(out)
        convout['conv212'] = out.clone()
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #---- Layer 3.1.1
        out = self.conv311(out)
        if self.noise_layer == 5:
            out = self.noise311(out)
        convout['conv311'] = out.clone()
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        if self.noise_layer == 6:
            out = self.noise312(out)
        convout['conv312'] = out.clone()
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        if self.noise_layer == 7:
            out = self.noise321(out)
        convout['conv321'] = out.clone()
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        if self.noise_layer == 8:
            out = self.noise322(out)
        convout['conv322'] = out.clone()
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        if self.store_act:
            return out
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        #return out
        return convout



class Model_NoisetoEveryConv(nn.Module):
    def __init__(self, args):
        super(Model_NoisetoEveryConv, self).__init__()
        
        self.classes = args.classes 
        self.inflate = args.inflate        
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.use_custom_norm = args.custom_norm
        self.store_act = args.store_act
        #self.noise_layer = args.noise_layer
        self.noise_sigma = np.load(args.sigma_path, allow_pickle=True).item()
        
        #pdb.set_trace()
        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise0 = GaussianNoise(sigma=self.noise_sigma['conv0'])
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        if self.inflate>1:
            self.resconv11  = nn.Sequential(OrderedDict({
                                            '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                            'gauss':GaussianNoise(sigma=self.noise_sigma['resconv11']),
                                            '1':nn.BatchNorm2d(16*self.inflate),}))
        else:
            self.resconv11  = nn.Sequential(OrderedDict({
                                            '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                            '1':nn.BatchNorm2d(16*self.inflate),}))
        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise111 = GaussianNoise(sigma=self.noise_sigma['conv111'])
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise112 = GaussianNoise(sigma=self.noise_sigma['conv112'])
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(OrderedDict({
                                        '0':nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                                        'gauss':GaussianNoise(sigma=self.noise_sigma['resconv21']),
                                        '1':nn.BatchNorm2d(32*self.inflate),}))
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise211 = GaussianNoise(sigma=self.noise_sigma['conv211'])
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise212 = GaussianNoise(sigma=self.noise_sigma['conv212'])
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(OrderedDict({
                                        '0':nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                                        'gauss':GaussianNoise(sigma=self.noise_sigma['resconv31']),
                                        '1':nn.BatchNorm2d(64*self.inflate),}))
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise311 = GaussianNoise(sigma=self.noise_sigma['conv311'])
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise312 = GaussianNoise(sigma=self.noise_sigma['conv312'])
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise321 = GaussianNoise(sigma=self.noise_sigma['conv321'])
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise322 = GaussianNoise(sigma=self.noise_sigma['conv322'])
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.noiselinear = GaussianNoise(sigma=self.noise_sigma['linear'])
        self.apply(_weights_init)
            
  
    def forward(self, x):
        if self.use_custom_norm:
            out = self.custom_norm(x)
            #print('using custom norm')
        else:
            out = x
        #---- Layer 0
        out = self.conv0(out) 
        out = self.noise0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
            #residual = self.noise11(residual)
            #residual = self.bn11(residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        out = self.noise111(out)
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        out = self.noise112(out)
        out = self.bn112(out)
        #---- add residual
        out+=residual        
        out = self.relu11(out)       
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #residual = self.noise21(residual)
        #residual = self.bn21(residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        out = self.noise211(out)
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        out = self.noise212(out)
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #residual = self.noise31(residual)
        #residual = self.bn31(residual)
        #---- Layer 3.1.1
        out = self.conv311(out)
        out = self.noise311(out)
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        out = self.noise312(out)
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        out = self.noise321(out)
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        out = self.noise322(out)
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #if self.store_act:
            #return out_before, out_after
            #pdb.set_trace()
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.noiselinear(out)
        return out



class Model_NoisetoEveryConv_Mean(nn.Module):
    def __init__(self, args):
        super(Model_NoisetoEveryConv_Mean, self).__init__()
        
        self.classes = args.classes 
        self.inflate = args.inflate        
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.use_custom_norm = args.custom_norm
        self.store_act = args.store_act
        self.noise_sigma = np.load(args.sigma_path, allow_pickle=True).item()
        self.noise_mean = np.load(args.mean_path, allow_pickle=True).item()
        
        #pdb.set_trace()
        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise0 = GaussianNoise_Mean(sigma=self.noise_sigma['conv0'], mean=self.noise_mean['conv0'])
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        if self.inflate>1:
            self.resconv11  = nn.Sequential(OrderedDict({
                                            '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                            'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv11'], mean=self.noise_mean['resconv11']),
                                            '1':nn.BatchNorm2d(16*self.inflate),}))
        else:
            self.resconv11  = nn.Sequential(OrderedDict({
                                            '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                            '1':nn.BatchNorm2d(16*self.inflate),}))
        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise111 = GaussianNoise_Mean(sigma=self.noise_sigma['conv111'], mean=self.noise_mean['conv111'])
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise112 = GaussianNoise_Mean(sigma=self.noise_sigma['conv112'], mean=self.noise_mean['conv112'])
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(OrderedDict({
                                        '0':nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                                        'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv21'], mean=self.noise_mean['resconv21']),
                                        '1':nn.BatchNorm2d(32*self.inflate),}))
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise211 = GaussianNoise_Mean(sigma=self.noise_sigma['conv211'], mean=self.noise_mean['conv211'])
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise212 = GaussianNoise_Mean(sigma=self.noise_sigma['conv212'], mean=self.noise_mean['conv212'])
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(OrderedDict({
                                        '0':nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                                        'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv31'], mean=self.noise_mean['resconv31']),
                                        '1':nn.BatchNorm2d(64*self.inflate),}))
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise311 = GaussianNoise_Mean(sigma=self.noise_sigma['conv311'], mean=self.noise_mean['conv311'])
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise312 = GaussianNoise_Mean(sigma=self.noise_sigma['conv312'], mean=self.noise_mean['conv312'])
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise321 = GaussianNoise_Mean(sigma=self.noise_sigma['conv321'], mean=self.noise_mean['conv321'])
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise322 = GaussianNoise_Mean(sigma=self.noise_sigma['conv322'], mean=self.noise_mean['conv322'])
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.noiselinear = GaussianNoise_Mean(sigma=self.noise_sigma['linear'], mean=self.noise_mean['linear'])
        self.apply(_weights_init)
            
  
    def forward(self, x):
        if self.use_custom_norm:
            out = self.custom_norm(x)
            #print('using custom norm')
        else:
            out = x
        #---- Layer 0
        out = self.conv0(out) 
        out = self.noise0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
            #residual = self.noise11(residual)
            #residual = self.bn11(residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        out = self.noise111(out)
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        out = self.noise112(out)
        out = self.bn112(out)
        #---- add residual
        out+=residual        
        out = self.relu11(out)       
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #residual = self.noise21(residual)
        #residual = self.bn21(residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        out = self.noise211(out)
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        out = self.noise212(out)
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #residual = self.noise31(residual)
        #residual = self.bn31(residual)
        #---- Layer 3.1.1
        out = self.conv311(out)
        out = self.noise311(out)
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        out = self.noise312(out)
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        out = self.noise321(out)
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        out = self.noise322(out)
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #if self.store_act:
            #return out_before, out_after
            #pdb.set_trace()
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.noiselinear(out)
        return out
