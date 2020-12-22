
mvm_params = {}
#---- Bit Slicing Parameters
mvm_params['wbit_frac']         = 13
mvm_params['ibit_frac']         = 13
mvm_params['bit_slice_in']      = 4
mvm_params['bit_stream_in']     = 4
mvm_params['wbit_total']        = 16
mvm_params['ibit_total']        = 16
#---- ADC parameters (for 16 bit fixed point computation) 
mvm_params['adc_bit']           = 14
mvm_params['acm_bits']          = 32
mvm_params['acm_bit_frac']      = 24

#---- GenieX model properties
mvm_params['loop']                      = False

mvm_params['ideal']                     = {}
mvm_params['ideal']['genieX']           = False
mvm_params['ocv']                       = False
mvm_params['ocv_delta']                 = 0.02
mvm_params['seed']                      = 0

#---- 16x16_100k
mvm_params['16x16_100k']                = {}
mvm_params['16x16_100k']['size']        = 16
mvm_params['16x16_100k']['Ron']         = 100*10*3
mvm_params['16x16_100k']['Ron_Roff']    = 6
mvm_params['16x16_100k']['inmax_test']  = 1.22
mvm_params['16x16_100k']['inmin_test']  = 0.81
mvm_params['16x16_100k']['path']        = './puma-functional-model/genieX_models/final_16x16_mlp2layer_xbar_16x16_100_all_dataset_500_100k_standard_sgd.pth.tar'
mvm_params['16x16_100k']['genieX']      = True
#---- 32x32_100k
mvm_params['32x32_100k']                = {}
mvm_params['32x32_100k']['size']        = 32
mvm_params['32x32_100k']['Ron']         = 100*10**3
mvm_params['32x32_100k']['Ron_Roff']    = 6
mvm_params['32x32_100k']['inmax_test']  = 1.4
mvm_params['32x32_100k']['inmin_test']  = 0.92
mvm_params['32x32_100k']['path']        = './genieX_models/final_32x32_mlp2layer_xbar_32x32_100_all_dataset_500_100k_standard_sgd.pth.tar'
mvm_params['32x32_100k']['genieX']      = True

#---- 64x64_100k
mvm_params['64x64_100k']                = {}
mvm_params['64x64_100k']['size']        = 64
mvm_params['64x64_100k']['Ron']         = 100*10**3
mvm_params['64x64_100k']['Ron_Roff']    = 6
mvm_params['64x64_100k']['inmax_test']  = 1.2
mvm_params['64x64_100k']['inmin_test']  = 0.85
mvm_params['64x64_100k']['path']        = './genieX_models/final_64x64_mlp2layer_xbar_64x64_100_all_v2_dataset_500_100k_standard_sgd.pth.tar'
mvm_params['64x64_100k']['genieX']      = True

#---- 64x64_50k
mvm_params['64x64_50k']                 = {}
mvm_params['64x64_50k']['size']         = 64
mvm_params['64x64_50k']['Ron']          = 50*10**3
mvm_params['64x64_50k']['Ron_Roff']     = 6
mvm_params['64x64_50k']['inmax_test']   = 1.65
mvm_params['64x64_50k']['inmin_test']   = 1.1
mvm_params['64x64_50k']['path']         = './genieX_models/final_64x64_mlp2layer_xbar_64x64_100_all_dataset_500_50k_standard_sgd.pth.tar'
mvm_params['64x64_50k']['genieX']        = True
#---- 64x64_300k
mvm_params['64x64_300k']                = {}
mvm_params['64x64_300k']['size']        = 64
mvm_params['64x64_300k']['Ron']         = 300*10**3
mvm_params['64x64_300k']['Ron_Roff']    = 6
mvm_params['64x64_300k']['inmax_test']  = 1.27
mvm_params['64x64_300k']['inmin_test']  = 0.88
mvm_params['64x64_300k']['path']        = './genieX_models/final_64x64_mlp2layer_xbar_64x64_100_all_dataset_500_300k_standard_sgd.pth.tar'
mvm_params['64x64_300k']['genieX']      = True



