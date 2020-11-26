import torch.nn as nn
import torch
import torch.nn.functional as F
import os

if_bit_slicing = True
debug = True

## Use global parameters (below) for all layers or layer specific parameters
val = True
ifglobal_weight_bits = val
ifglobal_weight_bit_frac = val
ifglobal_input_bits = val
ifglobal_input_bit_frac = val
ifglobal_xbar_col_size = val
ifglobal_xbar_row_size = val
ifglobal_tile_col = val
ifglobal_tile_row = val
ifglobal_bit_stream = val
ifglobal_bit_slice = val
ifglobal_adc_bit = val
ifglobal_acm_bits = val
ifglobal_acm_bit_frac = val
ifglobal_xbmodel = val
ifglobal_xbmodel_weight_path = val
ifglobal_dataset = True  # if True data collected from all layers


## Tiling configurations
tile_row = 8
tile_col = 8

## GENIEx configurations
loop = False # executes GENIEx with batching when set to False

## GENIEx data collection configuations
dataset = False
direc = 'geniex_dataset'  # folder containing geneix dataset
rows = 1 # num of crossbars in row dimension
cols = 1 # num of crossbars in col dimension
Vmax =0.25



# Dump the current global configurations
def dump_config():
    param_dict = {
        'weight_bits':weight_bits, 'weight_bit_frac':weight_bit_frac, 
        'input_bits':input_bits, 'input_bit_frac':input_bit_frac, 
        'tile_row':tile_row, 'tile_col':tile_col,
        'bit_stream':bit_stream, 'bit_slice':bit_slice, 
        'adc_bit':adc_bit, 'acm_bits':acm_bits, 'acm_bit_frac':acm_bit_frac,
    }

    print("==> Functional simulator configurations:", end=' ')
    for key, val in param_dict.items():
        t_str = key + '=' + str(val)
        print (t_str, end=', ')
    print('\n')


