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
        act = {}

        out = self.conv0(out)
        if self.store_act: act['conv0'] = out.clone()
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16out)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            residual = self.resconv11[0](residual)
            if self.store_act:
                act['resconv11'] = residual.clone()
            residual = self.resconv11[1](residual)
        #---- Layer 1.1.1
        out = self.conv111(out)
        if self.store_act: act['conv111'] = out.clone()
        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        out = self.conv112(out)
        if self.store_act: act['conv112'] = out.clone()
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        out = self.conv121(out)
        if self.store_act: act['conv121'] = out.clone()
        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        out = self.conv122(out)
        if self.store_act: act['conv122'] = out.clone()
        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        out = self.conv131(out)
        if self.store_act: act['conv131'] = out.clone()
        out = self.bn131(out)
        out = self.relu131(out)
        #---- Layer 1.3.2
        out = self.conv132(out)
        if self.store_act: act['conv132'] = out.clone()
        out = self.bn132(out)
        #---- add residual
        out+=residual
        out = self.relu13(out)
        #---- Group 2
        #---- Block 2.1
        residual = out.clone() 
        residual = self.resconv21[0](residual)
        if self.store_act: act['resconv21'] = residual.clone()
        residual = self.resconv21[1](residual)
        #---- Layer 2.1.1
        out = self.conv211(out)
        if self.store_act: act['conv211'] = out.clone()
        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        out = self.conv212(out)
        if self.store_act: act['conv212'] = out.clone()
        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1 
        out = self.conv221(out)
        if self.store_act: act['conv221'] = out.clone()
        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        out = self.conv222(out)
        if self.store_act: act['conv222'] = out.clone()
        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        out = self.conv231(out)
        if self.store_act: act['conv231'] = out.clone()
        out = self.bn231(out)
        out = self.relu231(out)
        #---- Layer 2.3.2
        out = self.conv232(out)
        if self.store_act: act['conv232'] = out.clone()
        out = self.bn232(out)
        #---- add residual
        out+=residual
        out = self.relu23(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 
        residual = self.resconv31[0](residual)
        if self.store_act: act['resconv31'] = residual.clone()
        residual = self.resconv31[1](residual)
        #---- Layer 3.1.1
        out = self.conv311(out)
        if self.store_act: act['conv311'] = out.clone()
        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        out = self.conv312(out)
        if self.store_act: act['conv312'] = out.clone()
        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        out = self.conv321(out)
        if self.store_act: act['conv321'] = out.clone()
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        out = self.conv322(out)
        if self.store_act: act['conv322'] = out.clone()
        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        out = self.conv331(out)
        if self.store_act: act['conv331'] = out.clone()
        out = self.bn331(out)
        out = self.relu331(out)
        #---- Layer 3.3.2
        out = self.conv332(out)
        if self.store_act: act['conv332'] = out.clone()
        out = self.bn332(out)
        #---- add residual
        out+=residual
        out = self.relu33(out)
        #if args.store_act: return out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.store_act: act['linear'] = out.clone()
        if self.store_act:
            return act
        else:
            return out


class MVM_Model(nn.Module):
    def __init__(self, args, mvm_params):
        super(MVM_Model, self).__init__()

        self.classes = args.classes 
        self.inflate = args.inflate
        self.store_act = args.store_act
        self.use_custom_norm = args.custom_norm
        self.custom_norm = custom_3channel_img_normalization_with_dataset_params(mean, std, [3,32,32], 'cuda') 
       
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
        self.conv0_digital  = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv0  = Conv2d_mvm(3,16, kernel_size=3, stride=1, padding=1, bias=False, 
                                bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn0    = nn.BatchNorm2d(16)
        self.relu0  = nn.ReLU(inplace=True)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        self.resconv11_digital = nn.Conv2d(16,16*self.inflate, kernel_size=1, stride=1, padding =0, bias=False)
        self.resconv11  = Conv2d_mvm(16,16*self.inflate, kernel_size=1, stride=1, padding=0, bias=False, 
                                            bit_slice=bit_slice_in,bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn11       = nn.BatchNorm2d(16*self.inflate)
        #---- Layer 1.1.1
        self.conv111_digital    = nn.Conv2d(16,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv111    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False,  
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn111      = nn.BatchNorm2d(16*self.inflate)
        self.relu111    = nn.ReLU(inplace=True)
        #---- Layer 1.1.2
        self.conv112_digital    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv112    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False,  
                                        bit_slice=bit_slice_in,bit_stream=bit_stream_in,weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn112      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu11    = nn.ReLU(inplace=True)
        #---- Block 1.2
        #---- Layer 1.2.1
        self.conv121_digital    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv121    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn121      = nn.BatchNorm2d(16*self.inflate)
        self.relu121    = nn.ReLU(inplace=True)
        #---- Layer 1.2.2
        self.conv122_digital    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv122    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn122      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu12    = nn.ReLU(inplace=True)
        #---- Block 1.3
        #---- Layer 1.3.1
        self.conv131_digital    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv131    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn131      = nn.BatchNorm2d(16*self.inflate)
        self.relu131    = nn.ReLU(inplace=True)
        #---- Layer 1.3.2
        self.conv132_digital    = nn.Conv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv132    = Conv2d_mvm( 16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn132      = nn.BatchNorm2d(16*self.inflate)
        #---- post-merge activation
        self.relu13    = nn.ReLU(inplace=True)
        #---- Group 2 (32x) (32x32 -> 16x16)
        #---- Block 2.1
        self.resconv21_digital = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False)
        self.resconv21  = Conv2d_mvm( 16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn21       = nn.BatchNorm2d(32*self.inflate)
        #---- Layer 2.1.1
        self.conv211_digital    = nn.Conv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv211    = Conv2d_mvm( 16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn211      = nn.BatchNorm2d(32*self.inflate)
        self.relu211    = nn.ReLU(inplace=True)
        #---- Layer 2.1.2
        self.conv212_digital    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv212    = Conv2d_mvm( 32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn212      = nn.BatchNorm2d(32*self.inflate)
        #---- post activation
        self.relu21     = nn.ReLU(inplace=True)
        #---- Block 2.2
        #---- Layer 2.2.1
        self.conv221_digital    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv221    = Conv2d_mvm( 32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn221      = nn.BatchNorm2d(32*self.inflate)
        self.relu221    = nn.ReLU(inplace=True)
        #---- Layer 2.2.2
        self.conv222_digital    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv222    = Conv2d_mvm( 32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                    bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                    adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn222      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu22     = nn.ReLU(inplace=True)
        #---- Block 2.3
        #---- Layer 2.3.1
        self.conv231_digital    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv231    = Conv2d_mvm( 32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn231      = nn.BatchNorm2d(32*self.inflate)
        self.relu231    = nn.ReLU(inplace=True)
        #---- Layer 2.3.2
        self.conv232_digital    = nn.Conv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv232    = Conv2d_mvm( 32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn232      = nn.BatchNorm2d(32*self.inflate)
        #---- post-merge activation
        self.relu23     = nn.ReLU(inplace=True)
        #---- Group 3 (64x) (16x16 -> 8x8)
        #---- Block 3.1
        self.resconv31_digital  = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False)
        self.resconv31  = Conv2d_mvm(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn31       = nn.BatchNorm2d(64*self.inflate)
        #---- Layer 3.1.1
        self.conv311_digital    = nn.Conv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv311    = Conv2d_mvm(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn311      = nn.BatchNorm2d(64*self.inflate)
        self.relu311    = nn.ReLU(inplace=True)
        #---- Layer 3.1.2
        self.conv312_digital    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv312    = Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn312      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu31     = nn.ReLU(inplace=True)
        #---- Block 3.2
        #---- Layer 3.2.1
        self.conv321_digital    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv321    = Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn321      = nn.BatchNorm2d(64*self.inflate)
        self.relu321    = nn.ReLU(inplace=True)
        #---- Layer 3.2.2
        self.conv322_digital    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv322    = Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn322      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu32     = nn.ReLU(inplace=True)
        #---- Block 3.3
        #---- Layer 3.3.1
        self.conv331_digital    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv331    = Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, 
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn331      = nn.BatchNorm2d(64*self.inflate)
        self.relu331    = nn.ReLU(inplace=True)
        #---- Layer 3.3.2
        self.conv332_digital    = nn.Conv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv332    = Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False,
                                        bit_slice=bit_slice_in, bit_stream=bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)
        self.bn332      = nn.BatchNorm2d(64*self.inflate)
        #---- post-merge activation
        self.relu33     = nn.ReLU(inplace=True)
        #---- Linear Classifier
        self.linear_digital     = nn.Linear(64*self.inflate, self.classes, bias = False)
        self.linear     = Linear_mvm(64*self.inflate, self.classes, bias=False, 
                                        bit_slice = bit_slice_in, bit_stream = bit_stream_in, weight_bits=wbit_total, weight_bit_frac=wbit_frac, input_bits=ibit_total, input_bit_frac=ibit_frac, 
                                        adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)


            
  
    def forward(self, x, layer_name):
        if self.use_custom_norm:
            out = self.custom_norm(x)
        else:
            out = x

        act = []

        #---- Layer 0
        if layer_name == 'conv0':
            out = self.conv0(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv0_digital(out)
        if self.store_act & (layer_name=='conv0'): act = out.clone()
            
        out = self.bn0(out)
        out = self.relu0(out)
        #---- Group 1 (16x) (32x32 -> 32x32)
        #---- Block 1.1
        residual = out.clone()
        if self.inflate > 1:
            if layer_name == 'resconv11':
                residual = self.resconv11(residual, self.Xbar_params, self.Xbar_model)
            else: 
                residual = self.resconv11_digital(residual)
            if self.store_act & (layer_name=='resconv11'): act = residual.clone()

            residual = self.bn11(residual)
        #---- Layer 1.1.1
        if layer_name == 'conv111':
            out = self.conv111(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv111_digital(out)
        if self.store_act & (layer_name=='conv111'): act = out.clone()

        out = self.bn111(out)
        out = self.relu111(out)
        #---- Layer 1.1.2
        if layer_name == 'conv112':
            out = self.conv112(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv112_digital(out)
        if self.store_act & (layer_name=='conv112'): act = out.clone()
        
        out = self.bn112(out)
        #---- add residual
        out+=residual
        out = self.relu11(out)
        #---- Block 1.2
        residual = out.clone()
        #---- Layer 1.2.1
        if layer_name == 'conv121':
            out = self.conv121(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv121_digital(out)
        if self.store_act & (layer_name=='conv121'): act = out.clone()

        out = self.bn121(out)
        out = self.relu121(out)
        #---- Layer 1.2.2
        if layer_name == 'conv122':
            out = self.conv122(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv122_digital(out)
        if self.store_act & (layer_name=='conv122'): act = out.clone()

        out = self.bn122(out)
        #---- add residual
        out+=residual
        out = self.relu12(out)
        #---- Block 1.3
        residual = out.clone()
        #---- Layer 1.3.1
        if layer_name == 'conv131':
            out = self.conv131(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv131_digital(out)
        if self.store_act & (layer_name=='conv131'): act = out.clone()
        
        out = self.bn131(out)
        out = self.relu131(out)
        #---- Layer 1.3.2
        if layer_name == 'conv132':
            out = self.conv132(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv132_digital(out)
        if self.store_act & (layer_name=='conv132'): act = out.clone()
        
        out = self.bn132(out)
        #---- add residual
        out+=residual
        out = self.relu13(out)
        #---- Group 2 
        #---- Block 2.1
        residual = out.clone() 
        
        if layer_name == 'resconv21':
            residual = self.resconv21(residual, self.Xbar_params, self.Xbar_model)
        else:
            residual = self.resconv21_digital(residual)
        if self.store_act & (layer_name=='resconv21'): act = residual.clone()

        residual = self.bn21(residual)
        #---- Layer 2.1.1
        if layer_name == 'conv211':
            out = self.conv211(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv211_digital(out)
        if self.store_act & (layer_name=='conv211'): act = out.clone()

        out = self.bn211(out)
        out = self.relu211(out)
        #---- Layer 2.1.2
        if layer_name == 'conv212':
            out = self.conv212(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv212_digital(out)
        if self.store_act & (layer_name=='conv212'): act = out.clone()

        out = self.bn212(out)
        #---- add residual
        out+=residual
        out = self.relu21(out)
        #---- Block 2.2
        residual = out.clone() 
        #---- Layer 2.2.1
        if layer_name == 'conv221':
            out = self.conv221(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv221_digital(out)
        if self.store_act & (layer_name=='conv221'): act = out.clone()

        out = self.bn221(out)
        out = self.relu221(out)
        #---- Layer 2.2.2
        if layer_name == 'conv222':
            out = self.conv222(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv222_digital(out)
        if self.store_act & (layer_name=='conv222'): act = out.clone()

        out = self.bn222(out)
        #---- add residual
        out+=residual
        out = self.relu22(out)
        #---- Block 2.3
        residual = out.clone() 
        #---- Layer 2.3.1
        if layer_name == 'conv231':
            out = self.conv231(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv231_digital(out)
        if self.store_act & (layer_name=='conv231'): act = out.clone()

        out = self.bn231(out)
        out = self.relu231(out)
        #---- Layer 2.3.2
        if layer_name == 'conv232':
            out = self.conv232(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv232_digital(out)
        if self.store_act & (layer_name=='conv232'): act = out.clone()

        out = self.bn232(out)
        #---- add residual
        out+=residual
        out = self.relu23(out)
        #---- Group 3
        #---- Block 3.1
        residual = out.clone() 

        if layer_name == 'resconv31':
            residual = self.resconv31(residual, self.Xbar_params, self.Xbar_model)
        else:
            residual = self.resconv31_digital(residual)
        if self.store_act & (layer_name=='resconv31'): act = residual.clone()
        
        residual = self.bn31(residual)
        #---- Layer 3.1.1
        if layer_name == 'conv311':
            out = self.conv311(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv311_digital(out)
        if self.store_act & (layer_name=='conv311'): act = out.clone()

        out = self.bn311(out)
        out = self.relu311(out)
        #---- Layer 3.1.2
        if layer_name == 'conv312':
            out = self.conv312(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv312_digital(out)
        if self.store_act & (layer_name=='conv312'): act = out.clone()

        out = self.bn312(out)
        #---- add residual
        out+=residual
        out = self.relu31(out)
        #---- Block 3.2
        residual = out.clone() 
        #---- Layer 3.2.1 
        if layer_name == 'conv321':
            out = self.conv321(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv321_digital(out)
        if self.store_act & (layer_name=='conv321'): act = out.clone()
        
        out = self.bn321(out)
        out = self.relu321(out)
        #---- Layer 3.2.2
        if layer_name == 'conv322':
            out = self.conv322(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv322_digital(out)
        if self.store_act & (layer_name=='conv322'): act = out.clone()

        out = self.bn322(out)
        #---- add residual
        out+=residual
        out = self.relu32(out)
        #---- Block 3.3
        residual = out.clone() 
        #---- Layer 3.3.1 
        if layer_name == 'conv331':
            out = self.conv331(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv331_digital(out)
        if self.store_act & (layer_name=='conv331'): act = out.clone()

        out = self.bn331(out)
        out = self.relu331(out)
        #---- Layer 3.3.2
        if layer_name == 'conv332':
            out = self.conv332(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.conv332_digital(out)
        if self.store_act & (layer_name=='conv332'): act = out.clone()

        out = self.bn332(out)
        #---- add residual
        out+=residual
        out = self.relu33(out)
        #if args.store_act: return out
        #---- BNReLU
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        if layer_name == 'linear':
            out = self.linear(out, self.Xbar_params, self.Xbar_model)
        else:
            out = self.linear_digital(out)
        if self.store_act & (layer_name=='linear'): act = out.clone()

        if self.store_act:
            return act
        else:
            return out

