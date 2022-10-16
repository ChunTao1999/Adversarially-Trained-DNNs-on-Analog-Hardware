import sys
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/configs')

import logging
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from custom_normalization_functions import custom_3channel_img_normalization_with_dataset_params
import pdb

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def layercushion_conv(x, A, Ax):   
     
    l2_x = torch.norm(x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]), dim=1)
    l2_Ax = torch.norm(Ax.reshape(Ax.shape[0], Ax.shape[1]*Ax.shape[2]*Ax.shape[3]), dim=1)
    l2_A = torch.norm(A)
    
    lc = l2_Ax/(l2_x*l2_A)
    
    return lc
    
def activation_contraction(x, phi_x):

    l2_x = torch.norm(x)
    l2_phi_x = torch.norm(phi_x)

    ac = l2_x/l2_phi_x

    return ac

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
        lc = {}
        #ac = {}
        
        #---- Layer 0
        #pdb.set_trace()
        inp0 = out.clone()
        out = self.conv0(out)
        lc['conv0'] = layercushion_conv(inp0, self.conv0.weight.data, out.clone())
        out = self.bn0(out)
        #bact0 = out.clone()
        out = self.relu0(out)
        #ac['relu0'] = activation_contraction(bact0, out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
        #---- Layer 1.1.1
        inp111 = out.clone()
        out = self.conv111(out)
        lc['conv111'] = layercushion_conv(inp111, self.conv111.weight.data.clone(), out.clone())
        out = self.bn111(out)
        #bact111 = out.clone()
        out = self.relu111(out)
        #ac['relu111'] = activation_contraction(bact111, out)
        #---- Layer 1.1.2
        inp112 = out.clone()
        out = self.conv112(out)
        lc['conv112'] = layercushion_conv(inp112, self.conv112.weight.data.clone(), out.clone())
        out = self.bn112(out)
        #---- add residual
        out+=residual
        #bact11 = out.clone()
        out = self.relu11(out)
        #ac['relu11'] = activation_contraction(bact11, out)
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #---- Layer 2.1.1
        inp211 = out.clone()
        out = self.conv211(out)
        lc['conv211'] = layercushion_conv(inp211, self.conv211.weight.data.clone(), out.clone())
        out = self.bn211(out)
        #bact211 = out.clone()
        out = self.relu211(out)
        #ac['relu211'] = activation_contraction(bact211, out)
        #---- Layer 2.1.2
        inp212 = out.clone()
        out = self.conv212(out)
        lc['conv212'] = layercushion_conv(inp212, self.conv212.weight.data.clone(), out.clone())
        out = self.bn212(out)
        #---- add residual
        out+=residual
        #bact21 = out.clone()
        out = self.relu21(out)
        #ac['relu21'] = activation_contraction(bact21, out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #---- Layer 3.1.1
        inp311 = out.clone()
        out = self.conv311(out)
        lc['conv311'] = layercushion_conv(inp311, self.conv311.weight.data.clone(), out.clone())
        out = self.bn311(out)
        #bact311 = out.clone()
        out = self.relu311(out)
        #ac['relu311'] = activation_contraction(bact311, out)
        #---- Layer 3.1.2
        inp312 = out.clone()
        out = self.conv312(out)
        lc['conv312'] = layercushion_conv(inp312, self.conv312.weight.data.clone(), out.clone())
        out = self.bn312(out)
        #---- add residual
        out+=residual
        #bact31 = out.clone()
        out = self.relu31(out)
        #ac['relu31'] = activation_contraction(bact31, out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        inp321 = out.clone()
        out = self.conv321(out)
        lc['conv321'] = layercushion_conv(inp321, self.conv321.weight.data.clone(), out.clone())
        out = self.bn321(out)
        #bact321 = out.clone()
        out = self.relu321(out)
        #ac['relu321'] = activation_contraction(bact321, out)
        #---- Layer 3.2.2
        inp322 = out.clone()
        out = self.conv322(out)
        lc['conv322'] = layercushion_conv(inp322, self.conv322.weight.data.clone(), out.clone())
        out = self.bn322(out)
        #---- add residual
        out+=residual
        #bact32 = out.clone()
        out = self.relu32(out)
        #ac['relu32'] = activation_contraction(bact32, out)
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        l2_x = torch.norm(out, dim=1)
        out = self.linear(out) 
        l2_A = torch.norm(self.linear.weight.data)
        l2_Ax = torch.norm(out, dim=1)
        lc['linear'] = l2_Ax/(l2_x*l2_A)
        return lc



