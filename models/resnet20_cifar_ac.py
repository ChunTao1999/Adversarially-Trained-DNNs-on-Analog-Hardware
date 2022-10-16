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
        ac = {}

        #---- Layer 0
        inp0 = out.clone()
        out = self.conv0(out)
        #lc['conv0'] = layercushion_conv(inp0, self.conv0.weight.data, out)
        out = self.bn0(out)
        bact0 = out.clone()
        out = self.relu0(out)
        ac['relu0'] = activation_contraction(bact0, out)        
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11(residual)
        #---- Layer 1.1.1
        inp111 = out.clone()
        out = self.conv111(out)
        #lc['conv111'] = layercushion_conv(inp111, self.conv111.weight.data.clone(), out.clone())
        out = self.bn111(out)
        bact111 = out.clone()
        out = self.relu111(out)
        ac['relu111'] = activation_contraction(bact111, out)  
        #---- Layer 1.1.2
        inp112 = out.clone()
        out = self.conv112(out)
        #lc['conv112'] = layercushion_conv(inp112, self.conv112.weight.data.clone(), out.clone())
        out = self.bn112(out)
        #---- add residual
        out+=residual
        bact11 = out.clone()
        out = self.relu11(out)
        ac['relu11'] = activation_contraction(bact11, out)  
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        inp121 = out.clone()
        out = self.conv121(out)
        #lc['conv121'] = layercushion_conv(inp121, self.conv121.weight.data.clone(), out.clone())
        out = self.bn121(out)
        bact121 = out.clone()
        out = self.relu121(out)
        ac['relu121'] = activation_contraction(bact121, out)  
        #---- Layer 1.2.2
        inp122 = out.clone()
        out = self.conv122(out)
        #lc['conv122'] = layercushion_conv(inp122, self.conv122.weight.data.clone(), out.clone())
        out = self.bn122(out)
        #---- add residual
        out+=residual
        bact12 = out.clone()
        out = self.relu12(out)
        ac['relu12'] = activation_contraction(bact12, out)  
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        inp131 = out.clone()
        out = self.conv131(out)
        #lc['conv131'] = layercushion_conv(inp131, self.conv131.weight.data.clone(), out.clone())
        out = self.bn131(out)
        bact131 = out.clone()
        out = self.relu131(out)
        ac['relu131'] = activation_contraction(bact131, out)  
        #---- Layer 1.3.2
        inp132 = out.clone()
        out = self.conv132(out)
        #lc['conv132'] = layercushion_conv(inp132, self.conv132.weight.data.clone(), out.clone())
        out = self.bn132(out)
        #---- add residual
        out+=residual
        bact13 = out.clone()
        out = self.relu13(out)
        ac['relu13'] = activation_contraction(bact13, out)  
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual)
        #---- Layer 2.1.1
        inp211 = out.clone()
        out = self.conv211(out)
        #lc['conv211'] = layercushion_conv(inp211, self.conv211.weight.data.clone(), out.clone())
        out = self.bn211(out)
        bact211 = out.clone()
        out = self.relu211(out)
        ac['relu211'] = activation_contraction(bact211, out)  
        #---- Layer 2.1.2
        inp212 = out.clone()
        out = self.conv212(out)
        #lc['conv212'] = layercushion_conv(inp212, self.conv212.weight.data.clone(), out.clone())
        out = self.bn212(out)
        #---- add residual
        out+=residual
        bact21 = out.clone()
        out = self.relu21(out)
        ac['relu21'] = activation_contraction(bact21, out)  
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        inp221 = out.clone()
        out = self.conv221(out)
        #lc['conv221'] = layercushion_conv(inp221, self.conv221.weight.data.clone(), out.clone())
        out = self.bn221(out)
        bact221 = out.clone()
        out = self.relu221(out)
        ac['relu221'] = activation_contraction(bact221, out)  
        #---- Layer 2.2.2
        inp222 = out.clone()
        out = self.conv222(out)
        #lc['conv222'] = layercushion_conv(inp222, self.conv222.weight.data.clone(), out.clone())
        out = self.bn222(out)
        #---- add residual
        out+=residual
        bact22 = out.clone()
        out = self.relu22(out)
        ac['relu22'] = activation_contraction(bact22, out)  
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        inp231 = out.clone()
        out = self.conv231(out)
        #lc['conv231'] = layercushion_conv(inp231, self.conv231.weight.data.clone(), out.clone())
        out = self.bn231(out)
        bact231 = out.clone()
        out = self.relu231(out)
        ac['relu231'] = activation_contraction(bact231, out)  
        #---- Layer 2.3.2
        inp232 = out.clone() 
        out = self.conv232(out)
        #lc['conv232'] = layercushion_conv(inp232, self.conv232.weight.data.clone(), out.clone())
        out = self.bn232(out)
        #---- add residual
        out+=residual
        bact23 = out.clone()
        out = self.relu23(out)
        ac['relu23'] = activation_contraction(bact23, out)  
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31(residual) 
        #---- Layer 3.1.1
        inp311 = out.clone()
        out = self.conv311(out)
        #lc['conv311'] = layercushion_conv(inp311, self.conv311.weight.data.clone(), out.clone())
        out = self.bn311(out)
        bact311 = out.clone()
        out = self.relu311(out)
        ac['relu311'] = activation_contraction(bact311, out)  
        #---- Layer 3.1.2
        inp312 = out.clone()
        out = self.conv312(out)
        #lc['conv312'] = layercushion_conv(inp312, self.conv312.weight.data.clone(), out.clone())
        out = self.bn312(out)
        #---- add residual
        out+=residual
        bact31 = out.clone()
        out = self.relu31(out)
        ac['relu31'] = activation_contraction(bact31, out)  
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        inp321 = out.clone()
        out = self.conv321(out)
        #lc['conv321'] = layercushion_conv(inp321, self.conv321.weight.data.clone(), out.clone())
        out = self.bn321(out)
        bact321 = out.clone()
        out = self.relu321(out)
        ac['relu321'] = activation_contraction(bact321, out)  
        #---- Layer 3.2.2
        inp322 = out.clone()
        out = self.conv322(out)
        #lc['conv322'] = layercushion_conv(inp322, self.conv322.weight.data.clone(), out.clone())
        out = self.bn322(out)
        #---- add residual
        out+=residual
        bact32 = out.clone()
        out = self.relu32(out)
        ac['relu32'] = activation_contraction(bact32, out)  
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        inp331 = out.clone()
        out = self.conv331(out)
        #lc['conv331'] = layercushion_conv(inp331, self.conv331.weight.data.clone(), out.clone())
        out = self.bn331(out)
        bact331 = out.clone()
        out = self.relu331(out)
        ac['relu331'] = activation_contraction(bact331, out)  
        #---- Layer 3.3.2
        inp332 = out.clone()
        out = self.conv332(out)
        #lc['conv332'] = layercushion_conv(inp332, self.conv332.weight.data.clone(), out.clone())
        out = self.bn332(out)
        #---- add residual
        out+=residual
        bact33 = out.clone()
        out = self.relu33(out)
        ac['relu33'] = activation_contraction(bact33, out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        #l2_x = torch.norm(out, dim=1)
        out = self.linear(out)
        #l2_A = torch.norm(self.linear.weight.data)
        #l2_Ax = torch.norm(out, dim=1)
        #lc['linear'] = l2_Ax/(l2_x*l2_A)
        #pdb.set_trace() 
        return ac
