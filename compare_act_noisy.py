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
import resnet10_cifar_noisy_in_out
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

    #---- Noise setup
    parser.add_argument('--noise-sigma', default=1, type=int, help='10*relative sigma of noise injection')

    args = parser.parse_args()

    if len(args.gpus) > 1:
        args.custom_norm = False
    args.custom_norm = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #args.store_act = True

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
        model_eval_noisy = resnet20_cifar_noisy_in_out.Model_NoisetoEveryConv(args)
    elif args.arch == 'resnet32':
        model_eval_noisy = resnet32_cifar.MVM_Model(args, mvm_params)
    elif args.arch == 'resnet10':
        model_eval_noisy = resnet10_cifar_noisy_in_out.Model_NoisetoEveryConv(args)
    else:
        print("ERROR: Invalid Model Architecture")
        exit(0)      

    checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.th'))
    model_eval_noisy.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.pretrained))
    
    save_path = os.path.join(args.pretrained, "clean", "ideal_vs_noise{}".format(args.noise_sigma/10.0))
        
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    logging.info('saving logs to ==> {}'.format(save_path))
    
    model_eval_noisy.cuda()
    model_eval_noisy.eval()
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()    
    end = time.time()
    
    nf = {}
    for i, (input, target) in enumerate(testloader):
        print('{}/{}'.format(i+1, len(testloader)))
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        #digital_output = model_eval(input_var)
        convout_before, convout_after = model_eval_noisy(input_var)
    
        #pdb.set_trace()
        err = {}
        for key, value in convout_before.items():
            ideal = {}
            nf_batch = {}

            convout_before[key] = convout_before[key].detach().cpu().numpy().reshape([args.batch_size,-1]) #ideal convout
            convout_after[key] = convout_after[key].detach().cpu().numpy().reshape([args.batch_size,-1]) #non-ideal convout
            
            err[key] = convout_before[key]-convout_after[key] #ideal - non-ideal
            #pdb.set_trace()
            ideal[key] = convout_before[key] #ideal
            #ideal[key][ideal[key]<1e-15] = 1e-15
            nf_batch[key] = np.true_divide(err[key], ideal[key])
            
            #Here we compute nf mean in each batch, and finally compute mean of nf across batches
            if i==0:
                nf[key] = np.mean(nf_batch[key])
            else:
                nf[key] = np.append(nf[key], np.mean(nf_batch[key]))
        
        if (i+1)%args.limit_test == 0:
            break
    
    np.save(os.path.join(save_path, 'non-idealitiy-factor_gauss_std{}.npy'.format(args.noise_sigma/10.0)), nf, allow_pickle=True)
    logging.info("non_ideality_factor")
    for key, value in nf.items():
        avg_nf = np.mean(value)
        logging.info('{}:{}'.format(key, avg_nf))
        
    
