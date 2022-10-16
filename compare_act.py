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
import resnet20_cifar
import resnet20_cifar_in_out
import resnet10_cifar
import resnet10_cifar_in_out
from mvm_params import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #---- System Setup
    parser.add_argument('--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 8)')
    parser.add_argument('--half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--seed', default=0, type=int,  help='provide seed')
    parser.add_argument('--print-freq', default=1, type=int,  help='frequency of printing batch results')
    parser.add_argument('--limit-test', default=0, type=int,  help='limit the number of batches to test')
    parser.add_argument('--debug', action='store_true',  help='enable debug mode')
    parser.add_argument('--store-act', action='store_true',  help='store activations')
    parser.add_argument('--gpus', default='0')


    #---- MVM Arguments (default arguments are for 16 bit fixed point evaluation on MVM
    parser.add_argument('--mvm', default=False, help='evaluate on PUMA', action='store_true')
    parser.add_argument('--mvm-type', choices=['16x16_100k', '32x32_100k', '64x64_100k', '64x64_50k', '64x64_300k', 'ideal'], help='mvm model')
    
    #---- Model setup
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices = ['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--classes', default=10, choices = [10, 100],  type=int, help='num classes')
    parser.add_argument('--datadir', default='/home/nano01/a/tao88/Datasets/', help='dataset folder')

    parser.add_argument('--arch', default='resnet20', choices=['resnet10', 'resnet20', 'resnet32'], help='model description')
    parser.add_argument('--inflate', default=1, type=int, help='model description')
    parser.add_argument('--pretrained', action='store', default=None,help='the path to the pretrained model')
    parser.add_argument('--batch-size', default=10, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--custom-norm', default=True)

    args = parser.parse_args()

    if len(args.gpus) > 1:
        args.custom_norm = False
    args.custom_norm = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.mvm = True 
    args.mvm_type = '32x32_100k'
    args.store_act = True

    #args.arch = 'resnet10'
    #args.inflate = 1
    #args.dataset = 'cifar10'
    #args.pretrained = './log/cifar10/pgd-linf-eps8iter50-resnet10w1'
    #args.batch_size = 10
    #args.limit_test = 100
    
    criterion = nn.CrossEntropyLoss().cuda()
        
    if args.custom_norm == True:
        transform=transforms.ToTensor()
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    #---- Preparing the Dataset 
    if args.dataset == 'cifar10': 
        print('loading cifar 10 dataset')
        testset     = torchvision.datasets.CIFAR10(root=args.datadir, train=False,  download=True, transform=transform)
        mvm_params['wbit_frac']         = 13
        mvm_params['ibit_frac']         = 13 
        args.classes = 10
    elif args.dataset == 'cifar100':
        print('loading cifar 100 dataset')
        testset     = torchvision.datasets.CIFAR100(root=args.datadir, train=False,  download=True, transform=transform)
        mvm_params['wbit_frac']         = 12
        mvm_params['ibit_frac']         = 12
        args.classes = 100
    else:
        logging.info("EROOR: Invalid dataset")
        exit(0)

    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #----- Loading Model 

    if args.arch == 'resnet20':
        pretrained_model = resnet20_cifar_in_out.Model(args)
    elif args.arch == 'resnet32':
        pretrained_model = resnet32_cifar.Model(args)
    elif args.arch == 'resnet10':
        pretrained_model = resnet10_cifar_in_out.Model(args)
    else: 
        print("ERROR: Invalid Model Architecture")
        exit(0)

    checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.th'))
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.pretrained))

    pdb.set_trace()
    model_eval = pretrained_model
    
    if args.arch == 'resnet20':
        mvm_model_eval = resnet20_cifar_in_out.MVM_Model(args, mvm_params)
    elif args.arch == 'resnet32':
        mvm_model_eval = resnet32_cifar.MVM_Model(args, mvm_params)
    elif args.arch == 'resnet10':
        mvm_model_eval = resnet10_cifar_in_out.MVM_Model(args, mvm_params)
    else:
        print("ERROR: Invalid Model Architecture")
        exit(0)      
    
    mvm_model_eval = ModelClone(model_to=mvm_model_eval, model_from=pretrained_model)
    
    save_path = os.path.join(args.pretrained, "clean", "digital_vs_{}".format(args.mvm_type))
        
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    logging.info('saving logs to ==> {}'.format(save_path))
    
    model_eval.cuda()
    model_eval.eval()

    mvm_model_eval.cuda()
    mvm_model_eval.eval()
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()    
    end = time.time()
    
    dig_vs_mvm = {}
    
    for i, (input, target) in enumerate(testloader):
        print('{}/{}'.format(i+1, len(testloader)))
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        digital_output = model_eval(input_var)
        mvm_output = mvm_model_eval(input_var)

        for key, value in digital_output.items():
            err_in = {}
            err_out = {}
            digital = {}
            mvm = {}
            #pdb.set_trace()
            digital['in'] = digital_output[key]['in'].detach().cpu().numpy().reshape([args.batch_size, -1])
            mvm['in'] = mvm_output[key]['in'].detach().cpu().numpy().reshape([args.batch_size, -1])

            digital['out'] = digital_output[key]['out'].detach().cpu().numpy().reshape([args.batch_size, -1])
            mvm['out'] = mvm_output[key]['out'].detach().cpu().numpy().reshape([args.batch_size, -1])
            
            err_in['l2_noise'] = np.sum(np.square(digital['in']-mvm['in']), 1)
            err_in['l2_digital'] = np.sum(np.square(digital['in']), 1)
            err_in['l2_mvm'] = np.sum(np.square(mvm['in']), 1)

            err_out['l2_noise'] = np.sum(np.square(digital['out']-mvm['out']), 1)
            err_out['l2_digital'] = np.sum(np.square(digital['out']), 1)
            err_out['l2_mvm'] = np.sum(np.square(mvm['out']), 1)
            if i == 0:
                dig_vs_mvm[key] = {}
                dig_vs_mvm[key]['in'] = err_in
                dig_vs_mvm[key]['out'] = err_out
            else:
                for l2_type, val in err_in.items():
                    dig_vs_mvm[key]['in'][l2_type] = np.append(dig_vs_mvm[key]['in'][l2_type], err_in[l2_type])
                    dig_vs_mvm[key]['out'][l2_type] = np.append(dig_vs_mvm[key]['out'][l2_type], err_out[l2_type])

        #pdb.set_trace()
        if (i+1)%args.limit_test == 0:
            break
    
    #pdb.set_trace()

    np.save(os.path.join(save_path, 'l2_activations_N{}.npy'.format(args.batch_size*args.limit_test)), dig_vs_mvm, allow_pickle=True)
    
    #noise_sensitivity = {}
    #snr_out = {}
    #mvm_snr_out = {}
    #snr_in = {}
    #noise_factor = {}
    
    logging.info("snr, noise_sensivity, noise_factor")
    for key, value in dig_vs_mvm.items():
        noise_sensitivity = np.mean(dig_vs_mvm[key]['out']['l2_noise']/dig_vs_mvm[key]['out']['l2_digital'])
        snr_out = dig_vs_mvm[key]['out']['l2_mvm']/dig_vs_mvm[key]['out']['l2_noise']
        mvm_snr_out = np.mean(snr_out)
        if key == 'conv0':
            logging.info('{},{}'.format(mvm_snr_out, noise_sensitivity))
            continue
        else:
            snr_in = dig_vs_mvm[key]['in']['l2_mvm']/dig_vs_mvm[key]['in']['l2_noise']
            noise_factor = np.mean(snr_in/snr_out) 
            #noise_ratio = dig_vs_mvm[key]['out']['l2_noise']/dig_vs_mvm[key]['out']['l2_noise']
            #logging.info(mvm_snr_out, noise_sensitivity, noise_factor)
            logging.info('{}, {}, {}'.format(mvm_snr_out, noise_sensitivity, noise_factor))

    #print('\nSNR OUT:')
    #for key,value in mvm_snr_out.items():
        #print(value)
    
    #print('\nNoise Sensitivity:')
    #for key,value in noise_sensitivity.items():
        #print(value)

    #print('\nNoise Factor:')
    #for key,value in noise_factor.items():
        #print(value)
        
    
