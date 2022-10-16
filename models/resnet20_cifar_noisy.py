import logging
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from pytorch_mvm_class_v3 import Conv2d_mvm, Linear_mvm, NN_model
from custom_normalization_functions import custom_3channel_img_normalization_with_dataset_params
import pdb
from collections import OrderedDict
from gaussian_noise import GaussianNoise, GaussianNoise_Mean
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
        
        self.store_act = args.store_act
        self.classes = args.classes 
        self.inflate = args.inflate
        self.use_custom_norm = args.custom_norm
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda') 
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
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn121      = nn.BatchNorm2d(16*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu12    = nn.ReLU(inplace=True)
        #---- Block 1.3
        #---- Layer 1.3.1
        self.conv131    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn131      = nn.BatchNorm2d(16*self.inflate)
        self.relu131    = nn.ReLU(inplace=True)
        #---- Layer 1.3.2
        self.conv132    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn132      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu13    = nn.ReLU(inplace=True)
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
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn221      = nn.BatchNorm2d(32*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn222      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)
        #---- Block 2.3
        #---- Layer 2.3.1
        self.conv231    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn231      = nn.BatchNorm2d(32*self.inflate)
        self.relu231    = nn.ReLU(inplace=True)
        #---- Layer 2.3.2
        self.conv232    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn232      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu23     = nn.ReLU(inplace=True)
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
        #---- Block 3.3
        #---- Layer 3.3.1
        self.conv331    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn331      = nn.BatchNorm2d(64*self.inflate)
        self.relu331    = nn.ReLU(inplace=True)
        #---- Layer 3.3.2
        self.conv332    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn332      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu33     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.apply(_weights_init)
            
  
    def forward(self, x):

        if self.use_custom_norm:
            out = self.custom_norm(x)
        else:
            out = x
        #---- Layer 0
        out = self.conv0(out)
        #if self.store_act: act['conv0'] = out.clone()
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        #if self.store_act: act['conv111'] = out.clone()
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        #if self.store_act: act['conv112'] = out.clone()
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        out = self.conv121(out)
        #if self.store_act: act['conv121'] = out.clone()
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        out = self.conv122(out)
        #if self.store_act: act['conv122'] = out.clone()
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        out = self.conv131(out)
        #if self.store_act: act['conv131'] = out.clone()
        out = self.bn131(out)
        out = self.relu131(out)
        #---- Layer 1.3.2
        out = self.conv132(out)
        #if self.store_act: act['conv132'] = out.clone()
        out = self.bn132(out)
        #---- add residual
        out+=residual
        out = self.relu13(out)
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        #if self.store_act: act['conv211'] = out.clone()
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        #if self.store_act: act['conv212'] = out.clone()
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        out = self.conv221(out)
        #if self.store_act: act['conv221'] = out.clone()
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        out = self.conv222(out)
        #if self.store_act: act['conv222'] = out.clone()
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        out = self.conv231(out)
        #if self.store_act: act['conv231'] = out.clone()
        out = self.bn231(out)
        out = self.relu231(out)
        #---- Layer 2.3.2
        out = self.conv232(out)
        #if self.store_act: act['conv232'] = out.clone()
        out = self.bn232(out)
        #---- add residual
        out+=residual
        out = self.relu23(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #---- Layer 3.1.1
        out = self.conv311(out)
        #if self.store_act: act['conv311'] = out.clone()
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        #if self.store_act: act['conv312'] = out.clone()
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        #if self.store_act: act['conv321'] = out.clone()
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        #if self.store_act: act['conv322'] = out.clone()
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        out = self.conv331(out)
        #if self.store_act: act['conv331'] = out.clone()
        out = self.bn331(out)
        out = self.relu331(out)
        #---- Layer 3.3.2
        out = self.conv332(out)
        #if self.store_act: act['conv332'] = out.clone()
        out = self.bn332(out)
        #---- add residual
        out+=residual
        out = self.relu33(out)
        #if args.store_act: return out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.store_act: act['linear'] = out

        return out


class Model_NoisetoEveryConv(nn.Module):
    def __init__(self, args):
        super(Model_NoisetoEveryConv, self).__init__()
        
        self.store_act = args.store_act
        self.classes = args.classes 
        self.inflate = args.inflate
        self.use_custom_norm = args.custom_norm
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.noise_sigma = np.load(args.sigma_path, allow_pickle=True).item()

        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise0 = GaussianNoise(sigma=self.noise_sigma['conv0'])
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        if self.inflate>1:
            self.resconv11  =nn.Sequential(OrderedDict({
                                '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                'gauss':GaussianNoise(sigma=self.noise_sigma['resconv11']),
                                '1':nn.BatchNorm2d(16*self.inflate),}))
        else:
            self.resconv11  =nn.Sequential(OrderedDict({
                                '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                '1':nn.BatchNorm2d(16*self.inflate),}))

        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise111   = GaussianNoise(sigma=self.noise_sigma['conv111'])
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise112   = GaussianNoise(sigma=self.noise_sigma['conv112'])
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise121   = GaussianNoise(sigma=self.noise_sigma['conv121'])
        self.bn121      = nn.BatchNorm2d(16*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise122   = GaussianNoise(sigma=self.noise_sigma['conv122'])
        self.bn122      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu12    = nn.ReLU(inplace=True)
        #---- Block 1.3
        #---- Layer 1.3.1
        self.conv131    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise131   = GaussianNoise(sigma=self.noise_sigma['conv131'])
        self.bn131      = nn.BatchNorm2d(16*self.inflate)
        self.relu131    = nn.ReLU(inplace=True)
        #---- Layer 1.3.2
        self.conv132    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise132   = GaussianNoise(sigma=self.noise_sigma['conv132'])
        self.bn132      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu13    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(OrderedDict({
                            '0':nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            'gauss':GaussianNoise(sigma=self.noise_sigma['resconv21']),
                            '1':nn.BatchNorm2d(32*self.inflate),}))
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise211   = GaussianNoise(sigma=self.noise_sigma['conv211'])
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise212   = GaussianNoise(sigma=self.noise_sigma['conv212'])
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise221   = GaussianNoise(sigma=self.noise_sigma['conv221'])
        self.bn221      = nn.BatchNorm2d(32*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise222   = GaussianNoise(sigma=self.noise_sigma['conv222'])
        self.bn222      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)
        #---- Block 2.3
        #---- Layer 2.3.1
        self.conv231    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise231   = GaussianNoise(sigma=self.noise_sigma['conv231'])
        self.bn231      = nn.BatchNorm2d(32*self.inflate)
        self.relu231    = nn.ReLU(inplace=True)
        #---- Layer 2.3.2
        self.conv232    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise232   = GaussianNoise(sigma=self.noise_sigma['conv232'])
        self.bn232      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu23     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(OrderedDict({
                            '0':nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            'gauss':GaussianNoise(sigma=self.noise_sigma['resconv31']),
                            '1':nn.BatchNorm2d(64*self.inflate),}))
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise311   = GaussianNoise(sigma=self.noise_sigma['conv311'])
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise312   = GaussianNoise(sigma=self.noise_sigma['conv312'])
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise321   = GaussianNoise(sigma=self.noise_sigma['conv321'])
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise322   = GaussianNoise(sigma=self.noise_sigma['conv322'])
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Block 3.3
        #---- Layer 3.3.1
        self.conv331    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise331   = GaussianNoise(sigma=self.noise_sigma['conv331'])
        self.bn331      = nn.BatchNorm2d(64*self.inflate)
        self.relu331    = nn.ReLU(inplace=True)
        #---- Layer 3.3.2
        self.conv332    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise332   = GaussianNoise(sigma=self.noise_sigma['conv332'])
        self.bn332      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu33     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.noiselinear = GaussianNoise(sigma=self.noise_sigma['linear'])
        self.apply(_weights_init)
            
  
    def forward(self, x):

        if self.use_custom_norm:
            out = self.custom_norm(x)
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
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        out = self.conv121(out)
        out = self.noise121(out)
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        out = self.conv122(out)
        out = self.noise122(out)
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        out = self.conv131(out)
        out = self.noise131(out)
        out = self.bn131(out)
        out = self.relu131(out)
        #---- Layer 1.3.2
        out = self.conv132(out)
        out = self.noise132(out)
        out = self.bn132(out)
        #---- add residual
        out+=residual
        out = self.relu13(out)
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
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
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        out = self.conv221(out)
        out = self.noise221(out)
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        out = self.conv222(out)
        out = self.noise222(out)
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        out = self.conv231(out)
        out = self.noise231(out)
        out = self.bn231(out)
        out = self.relu231(out)
        #---- Layer 2.3.2
        out = self.conv232(out)
        out = self.noise232(out)
        out = self.bn232(out)
        #---- add residual
        out+=residual
        out = self.relu23(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
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
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        out = self.conv331(out)
        out = self.noise331(out)
        out = self.bn331(out)
        out = self.relu331(out)
        #---- Layer 3.3.2
        out = self.conv332(out)
        out = self.noise332(out)
        out = self.bn332(out)
        #---- add residual
        out+=residual
        out = self.relu33(out)
        #if args.store_act: return out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.noiselinear(out)

        return out



class Model_NoisetoEveryConv_Mean(nn.Module):
    def __init__(self, args):
        super(Model_NoisetoEveryConv_Mean, self).__init__()
        
        self.store_act = args.store_act
        self.classes = args.classes 
        self.inflate = args.inflate
        self.use_custom_norm = args.custom_norm
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda')
        self.noise_sigma = np.load(args.sigma_path, allow_pickle=True).item()
        self.noise_mean = np.load(args.mean_path, allow_pickle=True).item()

        #---- Layer 0
        self.conv0  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise0 = GaussianNoise_Mean(sigma=self.noise_sigma['conv0'], mean=self.noise_mean['conv0'])
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        if self.inflate>1:
            self.resconv11  =nn.Sequential(OrderedDict({
                                '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv11'], mean=self.noise_mean['resconv11']),
                                '1':nn.BatchNorm2d(16*self.inflate),}))
        else:
            self.resconv11  =nn.Sequential(OrderedDict({
                                '0':nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                '1':nn.BatchNorm2d(16*self.inflate),}))

        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise111   = GaussianNoise_Mean(sigma=self.noise_sigma['conv111'], mean=self.noise_mean['conv111'])
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise112   = GaussianNoise_Mean(sigma=self.noise_sigma['conv112'], mean=self.noise_mean['conv112'])
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise121   = GaussianNoise_Mean(sigma=self.noise_sigma['conv121'], mean=self.noise_mean['conv121'])
        self.bn121      = nn.BatchNorm2d(16*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise122   = GaussianNoise_Mean(sigma=self.noise_sigma['conv122'], mean=self.noise_mean['conv122'])
        self.bn122      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu12    = nn.ReLU(inplace=True)
        #---- Block 1.3
        #---- Layer 1.3.1
        self.conv131    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise131   = GaussianNoise_Mean(sigma=self.noise_sigma['conv131'], mean=self.noise_mean['conv131'])
        self.bn131      = nn.BatchNorm2d(16*self.inflate)
        self.relu131    = nn.ReLU(inplace=True)
        #---- Layer 1.3.2
        self.conv132    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise132   = GaussianNoise_Mean(sigma=self.noise_sigma['conv132'], mean=self.noise_mean['conv132'])
        self.bn132      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu13    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21  = nn.Sequential(OrderedDict({
                            '0':nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv21'], mean=self.noise_mean['resconv21']),
                            '1':nn.BatchNorm2d(32*self.inflate),}))
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise211   = GaussianNoise_Mean(sigma=self.noise_sigma['conv211'], mean=self.noise_mean['conv211'])
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise212   = GaussianNoise_Mean(sigma=self.noise_sigma['conv212'], mean=self.noise_mean['conv212'])
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise221   = GaussianNoise_Mean(sigma=self.noise_sigma['conv221'], mean=self.noise_mean['conv221'])
        self.bn221      = nn.BatchNorm2d(32*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise222   = GaussianNoise_Mean(sigma=self.noise_sigma['conv222'], mean=self.noise_mean['conv222'])
        self.bn222      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)
        #---- Block 2.3
        #---- Layer 2.3.1
        self.conv231    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise231   = GaussianNoise_Mean(sigma=self.noise_sigma['conv231'], mean=self.noise_mean['conv231'])
        self.bn231      = nn.BatchNorm2d(32*self.inflate)
        self.relu231    = nn.ReLU(inplace=True)
        #---- Layer 2.3.2
        self.conv232    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise232   = GaussianNoise_Mean(sigma=self.noise_sigma['conv232'], mean=self.noise_mean['conv232'])
        self.bn232      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu23     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31  = nn.Sequential(OrderedDict({
                            '0':nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            'gauss':GaussianNoise_Mean(sigma=self.noise_sigma['resconv31'], mean=self.noise_mean['resconv31']),
                            '1':nn.BatchNorm2d(64*self.inflate),}))
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.noise311   = GaussianNoise_Mean(sigma=self.noise_sigma['conv311'], mean=self.noise_mean['conv311'])
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise312   = GaussianNoise_Mean(sigma=self.noise_sigma['conv312'], mean=self.noise_mean['conv312'])
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise321   = GaussianNoise_Mean(sigma=self.noise_sigma['conv321'], mean=self.noise_mean['conv321'])
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise322   = GaussianNoise_Mean(sigma=self.noise_sigma['conv322'], mean=self.noise_mean['conv322'])
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Block 3.3
        #---- Layer 3.3.1
        self.conv331    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise331   = GaussianNoise_Mean(sigma=self.noise_sigma['conv331'], mean=self.noise_mean['conv331'])
        self.bn331      = nn.BatchNorm2d(64*self.inflate)
        self.relu331    = nn.ReLU(inplace=True)
        #---- Layer 3.3.2
        self.conv332    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.noise332   = GaussianNoise_Mean(sigma=self.noise_sigma['conv332'], mean=self.noise_mean['conv332'])
        self.bn332      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu33     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.noiselinear = GaussianNoise_Mean(sigma=self.noise_sigma['linear'], mean=self.noise_mean['linear'])
        self.apply(_weights_init)
            
  
    def forward(self, x):

        if self.use_custom_norm:
            out = self.custom_norm(x)
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
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        out = self.conv121(out)
        out = self.noise121(out)
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        out = self.conv122(out)
        out = self.noise122(out)
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        out = self.conv131(out)
        out = self.noise131(out)
        out = self.bn131(out)
        out = self.relu131(out)
        #---- Layer 1.3.2
        out = self.conv132(out)
        out = self.noise132(out)
        out = self.bn132(out)
        #---- add residual
        out+=residual
        out = self.relu13(out)
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
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
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        out = self.conv221(out)
        out = self.noise221(out)
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        out = self.conv222(out)
        out = self.noise222(out)
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        out = self.conv231(out)
        out = self.noise231(out)
        out = self.bn231(out)
        out = self.relu231(out)
        #---- Layer 2.3.2
        out = self.conv232(out)
        out = self.noise232(out)
        out = self.bn232(out)
        #---- add residual
        out+=residual
        out = self.relu23(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
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
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        out = self.conv331(out)
        out = self.noise331(out)
        out = self.bn331(out)
        out = self.relu331(out)
        #---- Layer 3.3.2
        out = self.conv332(out)
        out = self.noise332(out)
        out = self.bn332(out)
        #---- add residual
        out+=residual
        out = self.relu33(out)
        #if args.store_act: return out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.noiselinear(out)

        return out
