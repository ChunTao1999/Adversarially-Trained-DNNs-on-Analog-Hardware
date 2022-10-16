import math
import logging
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from pytorch_mvm_class_v3 import Conv2d_mvm, Linear_mvm, NN_model
from custom_normalization_functions import custom_3channel_img_normalization_with_dataset_params

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
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,224,224], 'cuda') 
        #---- Layer 0
        self.conv0      = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False) # 224 -> 112 , 331 -> 166
        self.bn0        = nn.BatchNorm2d(64)
        self.relu0      = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 112 -> 56 , 166 -> 83
        #---- Group 1
        #---- Block 1.1
        self.resconv11  = nn.Sequential(
                                nn.Conv2d(64,64*self.inflate, kernel_size=1, stride=1, padding =0, bias=False),
                                nn.BatchNorm2d(64*self.inflate),)
        #---- Layer 1.1.1
        self.conv111    = nn.Conv2d(64,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False) # no change
        self.bn111      = nn.BatchNorm2d(64*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu11     = nn.ReLU(inplace=True)
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn121      = nn.BatchNorm2d(64*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu12     = nn.ReLU(inplace=True)        
        #---- Group 2
        #---- Block 2.1
        self.resconv21  = nn.Sequential(
                            nn.Conv2d(64*self.inflate,128*self.inflate, kernel_size=1, stride=2, padding =0, bias=False), 
                            nn.BatchNorm2d(128*self.inflate),)
        #---- Layer 2.1.1
        self.conv211    = nn.Conv2d(64*self.inflate,128*self.inflate, kernel_size=3, stride=2, padding=1, bias=False) # 56 -> 28, 83 -> 42
        self.bn211      = nn.BatchNorm2d(128*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = nn.Conv2d(128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn212      = nn.BatchNorm2d(128*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221    = nn.Conv2d(128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn221      = nn.BatchNorm2d(128*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222    = nn.Conv2d(128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn222      = nn.BatchNorm2d(128*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)       
        #---- Group 3
        #---- Block 3.1
        self.resconv31  = nn.Sequential(
                            nn.Conv2d(128*self.inflate,256*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(256*self.inflate),)
        #---- Layer 3.1.1
        self.conv311    = nn.Conv2d(128*self.inflate,256*self.inflate, kernel_size=3, stride=2, padding=1, bias=False) # 28 -> 14, 42 -> 21 
        self.bn311      = nn.BatchNorm2d(256*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = nn.Conv2d(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn312      = nn.BatchNorm2d(256*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = nn.Conv2d(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn321      = nn.BatchNorm2d(256*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = nn.Conv2d(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn322      = nn.BatchNorm2d(256*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Group 4 
        #---- Block 4.1
        self.resconv41  = nn.Sequential(
                            nn.Conv2d(256*self.inflate,512*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
                            nn.BatchNorm2d(512*self.inflate),)
        #---- Layer 4.1.1
        self.conv411    = nn.Conv2d(256*self.inflate,512*self.inflate, kernel_size=3, stride=2, padding=1, bias=False) # 14 -> 7, 21 -> 11
        self.bn411      = nn.BatchNorm2d(512*self.inflate)
        self.relu411    = nn.ReLU(inplace=True)
        #---- Layer 4.1.2
        self.conv412    = nn.Conv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn412      = nn.BatchNorm2d(512*self.inflate)
        #---- post-merge activation
        self.relu41     = nn.ReLU(inplace=True)
        #---- Block 4.2
        #---- Layer 4.2.1
        self.conv421    = nn.Conv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn421      = nn.BatchNorm2d(512*self.inflate)
        self.relu421    = nn.ReLU(inplace=True)
        #---- Layer 4.2.2
        self.conv422    = nn.Conv2d(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn422      = nn.BatchNorm2d(512*self.inflate)
        #---- post-merge activation
        self.relu42     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.avgpool    = nn.AdaptiveAvgPool2d(1) 
        self.linear     = nn.Linear(512*self.inflate, self.classes, bias = False)
        self.apply(_weights_init)
            
  
    def forward(self, x):
        act = {}
        out = self.custom_norm(x)
        #---- Layer 0
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.maxpool(out)
        #---- Group 1
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
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        out = self.conv121(out)
       
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        out = self.conv122(out)
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
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
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        out = self.conv221(out)
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        out = self.conv222(out)
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
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
        #---- Group 4
        #---- Block 4.1
        residual = out.clone() 
        residual = self.resconv41(residual) 
        #---- Layer 3.1.1
        out = self.conv411(out)
        out = self.bn411(out)
        out = self.relu311(out)
        #---- Layer 4.1.2
        out = self.conv412(out)
        out = self.bn412(out)
        #---- add residual
        out+=residual
        out = self.relu41(out)
        #---- Block 4.2
        residual = out.clone() 
        #---- Layer 4.2.1 
        out = self.conv421(out)
        out = self.bn421(out)
        out = self.relu421(out)
        #---- Layer 4.2.2
        out = self.conv422(out)
        out = self.bn422(out)
        #---- add residual
        out+=residual
        out = self.relu42(out)
        #---- BNReLU
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out


class MVM_Model(nn.Module):
    def __init__(self, args, mvm_params):
        super(MVM_Model, self).__init__()

        self.classes = args.classes 
        self.inflate = args.inflate
        self.store_act = args.store_act
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,224,224], 'cuda') 
        
        wbit_frac           = mvm_params['wbit_frac']
        ibit_frac           = mvm_params['ibit_frac']
        bit_slice_in        = mvm_params['bit_slice_in']
        bit_stream_in       = mvm_params['bit_stream_in']
        wbit_total          = mvm_params['wbit_total']
        ibit_total          = mvm_params['ibit_total']
        self.Xbar_params    = mvm_params[args.mvm_type]
        adc_bit             = mvm_params['adc_bit']
        acm_bits            = mvm_params['acm_bits']
        acm_bit_frac        = mvm_params['acm_bit_frac']
        self.Xbar_params['ocv'] = mvm_params['ocv']
        self.Xbar_params['ocv_delta'] = mvm_params['ocv_delta']
        self.Xbar_params['seed'] = mvm_params['seed']
        if mvm_params[args.mvm_type]['genieX']:
            print("loading Xbar model from ===> {}".format(self.Xbar_params['path']))
            self.Xbar_model = NN_model(self.Xbar_params['size'])
            checkpoint = torch.load(self.Xbar_params['path'])
            self.Xbar_model.load_state_dict(checkpoint['state_dict'])
            self.Xbar_model.cuda()
            self.Xbar_model.eval()
        else:
            self.Xbar_model = []
        
        #---- Layer 0
        self.conv0  = Conv2d_mvm(3, 64, kernel_size=7, stride=2, padding=3, bias=False, 
                                bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn0    = nn.BatchNorm2d(64)
        self.relu0  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #---- Group 1 
        #---- Block 1.1
        self.resconv11  = Conv2d_mvm(64,64*self.inflate, kernel_size=1, stride=1, padding=0, bias=False, 
                                            bit_slice=bit_slice_in,bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn11       = nn.BatchNorm2d(64*self.inflate)
        #---- Layer 1.1.1
        self.conv111    = Conv2d_mvm( 64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False,  
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn111      = nn.BatchNorm2d(64*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112    = Conv2d_mvm( 64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False,  
                                        bit_slice=bit_slice_in,bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn112      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121    = Conv2d_mvm( 64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn121      = nn.BatchNorm2d(64*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122    = Conv2d_mvm( 64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn122      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu12    = nn.ReLU(inplace=True)
        
        #---- Group 2
        #---- Block 2.1
        self.resconv21  = Conv2d_mvm( 64*self.inflate,128*self.inflate, kernel_size=1, stride=2, padding =0, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn21       = nn.BatchNorm2d(128*self.inflate)
        #---- Layer 2.1.1
        self.conv211    = Conv2d_mvm( 64*self.inflate,128*self.inflate, kernel_size=3, stride=2, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn211      = nn.BatchNorm2d(128*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212    = Conv2d_mvm( 128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn212      = nn.BatchNorm2d(128*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221    = Conv2d_mvm( 128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn221      = nn.BatchNorm2d(128*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222    = Conv2d_mvm( 128*self.inflate,128*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                    bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                    adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn222      = nn.BatchNorm2d(128*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)
        #---- Group 3 
        #---- Block 3.1
        self.resconv31  = Conv2d_mvm(128*self.inflate,256*self.inflate, kernel_size=1, stride=2, padding =0, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn31       = nn.BatchNorm2d(256*self.inflate)
        #---- Layer 3.1.1
        self.conv311    = Conv2d_mvm(256*self.inflate,256*self.inflate, kernel_size=3, stride=2, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn311      = nn.BatchNorm2d(256*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312    = Conv2d_mvm(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn312      = nn.BatchNorm2d(256*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321    = Conv2d_mvm(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn321      = nn.BatchNorm2d(256*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322    = Conv2d_mvm(256*self.inflate,256*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn322      = nn.BatchNorm2d(256*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Group 4 
        #---- Block 4.1
        self.resconv41  = Conv2d_mvm(256*self.inflate,512*self.inflate, kernel_size=1, stride=2, padding =0, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn41       = nn.BatchNorm2d(512*self.inflate)
        #---- Layer 4.1.1
        self.conv411    = Conv2d_mvm(512*self.inflate,512*self.inflate, kernel_size=3, stride=2, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn411      = nn.BatchNorm2d(512*self.inflate)
        self.relu411    = nn.ReLU(inplace=True)
        #---- Layer 4.1.2
        self.conv412    = Conv2d_mvm(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn412      = nn.BatchNorm2d(512*self.inflate)
        #---- post-merge activation
        self.relu41     = nn.ReLU(inplace=True)
        #---- Block 4.2
        #---- Layer 4.2.1
        self.conv421    = Conv2d_mvm(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn421      = nn.BatchNorm2d(512*self.inflate)
        self.relu421    = nn.ReLU(inplace=True)
        #---- Layer 4.2.2
        self.conv422    = Conv2d_mvm(512*self.inflate,512*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn422      = nn.BatchNorm2d(512*self.inflate)
        #---- post-merge activation
        self.relu42     = nn.ReLU(inplace=True)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        #---- Linear Classifier
        self.linear     = Linear_mvm(512*self.inflate, self.classes, bias=False, 
                                        bit_slice = bit_slice_in, bit_stream = bit_stream_in, weight_bits=wbit_total, weight_bit_frac=12, input_bits=ibit_total, input_bit_frac=11, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)


            
  
    def forward(self, x):
        act = {}
        out = self.custom_norm(x)
        #---- Layer 0
        self.Xbar_params['seed'] = 0
        out = self.conv0(out, self.Xbar_params, self.Xbar_model)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.maxpool(out)
        #---- Group 1
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            self.Xbar_params['seed'] = 1
            residual = self.resconv11(residual, self.Xbar_params, self.Xbar_model)
            residual = self.bn11(residual)
        #---- Layer 1.1.1
        self.Xbar_params['seed'] = 2
        out = self.conv111(out, self.Xbar_params, self.Xbar_model)
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        self.Xbar_params['seed'] = 3
        out = self.conv112(out, self.Xbar_params, self.Xbar_model)
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        self.Xbar_params['seed'] = 4
        out = self.conv121(out, self.Xbar_params, self.Xbar_model)
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        self.Xbar_params['seed'] = 5
        out = self.conv122(out, self.Xbar_params, self.Xbar_model)
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Group 2 
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21(residual, self.Xbar_params, self.Xbar_model)
        residual = self.bn21(residual)
        #---- Layer 2.1.1
        self.Xbar_params['seed'] = 7
        out = self.conv211(out, self.Xbar_params, self.Xbar_model)
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        self.Xbar_params['seed'] = 8
        out = self.conv212(out, self.Xbar_params, self.Xbar_model)        
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        self.Xbar_params['seed'] = 9
        out = self.conv221(out, self.Xbar_params, self.Xbar_model)
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        self.Xbar_params['seed'] = 10
        out = self.conv222(out, self.Xbar_params, self.Xbar_model)
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        self.Xbar_params['seed'] = 11
        residual = self.resconv31(residual, self.Xbar_params, self.Xbar_model) 
        residual = self.bn31(residual)
        #---- Layer 3.1.1
        self.Xbar_params['seed'] = 12
        out = self.conv311(out, self.Xbar_params, self.Xbar_model)
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        self.Xbar_params['seed'] = 13
        out = self.conv312(out, self.Xbar_params, self.Xbar_model)
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        self.Xbar_params['seed'] = 14
        out = self.conv321(out, self.Xbar_params, self.Xbar_model)
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        self.Xbar_params['seed'] = 15
        out = self.conv322(out, self.Xbar_params, self.Xbar_model)
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #---- Group 4
        #---- Block 4.1
        residual = out.clone() 
        self.Xbar_params['seed'] = 16
        residual = self.resconv41(residual, self.Xbar_params, self.Xbar_model) 
        residual = self.bn41(residual)
        #---- Layer 4.1.1
        self.Xbar_params['seed'] = 17
        out = self.conv411(out, self.Xbar_params, self.Xbar_model)
        out = self.bn411(out)
        out = self.relu411(out)
        #---- Layer 4.1.2
        self.Xbar_params['seed'] = 18
        out = self.conv412(out, self.Xbar_params, self.Xbar_model)
        out = self.bn412(out)
        #---- add residual
        out+=residual
        out = self.relu41(out)
        #---- Block 4.2
        residual = out.clone() 
        #---- Layer 4.2.1 
        self.Xbar_params['seed'] = 19
        out = self.conv421(out, self.Xbar_params, self.Xbar_model)
        out = self.bn421(out)
        out = self.relu321(out)
        #---- Layer 4.2.2
        self.Xbar_params['seed'] = 20
        out = self.conv422(out, self.Xbar_params, self.Xbar_model)
        out = self.bn422(out)
        #---- add residual
        out+=residual
        out = self.relu42(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        self.Xbar_params['seed'] = 21
        out = self.linear(out, self.Xbar_params, self.Xbar_model)
        
        return out

