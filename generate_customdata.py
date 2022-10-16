import sys
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/models/')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/configs')
sys.path.append('/home/nano01/a/tao88/AdversarialTraining_AnalogHardware/puma-functional-model-v3')
import os
import pdb
from utils import *
import torch
import math
import time
import numpy as np
import torch.nn as nn
import pkbar
import argparse 
import attacks
import logging
import torchattacks
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet10_cifar
import resnet20_cifar
from mvm_params import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #---- System Setup
    parser.add_argument('--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 8)')
    parser.add_argument('--half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--seed', default=5, type=int,  help='provide seed')
    parser.add_argument('--print-freq', default=100, type=int,  help='frequency of printing batch results')
    parser.add_argument('--store-act', action='store_true',  help='store activations')
    parser.add_argument('--debug', action='store_true',  help='Enable debug mode')
    parser.add_argument('--customdir', default='./customdata/', help='custom dataset dir')

    #---- MVM Arguments (default arguments are for 16 bit fixed point evaluation on MVM
    parser.add_argument('--mvm', default=False, help='evaluate on PUMA', action='store_true')
    parser.add_argument('--genieX', default=False, help='evaluate with GenieX', action='store_true')
    parser.add_argument('--ocv', default=False, help='Enable On Chip Variations', action='store_true')
    parser.add_argument('--mvm-type', choices=['16x16_100k', '32x32_100k', '64x64_100k', '64x64_50k', '64x64_300k', 'ideal'], help='mvm model')
    
    #---- Model setup
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices = ['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--classes', default=10, choices = [10, 100],  type=int, help='num classes')
    parser.add_argument('--datadir', default='/home/nano01/a/tao88/Datasets/', help='dataset folder')

    parser.add_argument('--arch', default='resnet10', choices=['resnet10', 'resnet20', 'resnet32'], help='model description')
    parser.add_argument('--inflate', default=1, type=int, help='model description')
    parser.add_argument('--modelfile', action='store', default=None,help='filename of ')
    parser.add_argument('--batch-size', default=500, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--custom-norm', default=True)



    #---- Adversarial Attack 
    parser.add_argument('--attack-iter', help='Adversarial attack iteration', type=int, default=50)
    parser.add_argument('--attack-epsilon', help='Adversarial attack maximal perturbation', type=int, default=8)
    parser.add_argument('--attack-step-size', help='Adversarial attack step size', type=float, default=1.0)
    
    args = parser.parse_args()
    args.dataset = 'cifar10'
    args.arch = 'resnet10'


    args.modelfile = './log/cifar10/pgd-linf-eps2iter50-resnet10w1/best_model.th'
    args.attack_epsilon = 1
    modeltype = args.modelfile.split('/')[3]
    

     
    
    if args.ocv:  
        Xbar_params['ocv'] = True
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    custom_data = {}
    custom_data['data'] = []
    custom_data['label'] = []
    custom_data['target'] = []
    
    #---- Preparing the Dataset 
    if args.dataset == 'cifar10': 
        print('loading cifar 10 dataset')
        testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transforms.ToTensor())
        mvm_params['wbit_frac']         = 13
        mvm_params['ibit_frac']         = 13 
        args.classes = 10
    elif args.dataset == 'cifar100':
        print('loading cifar 100 dataset')
        testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, download=True, transform=transforms.ToTensor())
        mvm_params['wbit_frac']         = 12
        mvm_params['ibit_frac']         = 12
        args.classes = 100
    elif args.dataset == 'imnet':
        valdir =  valdir = os.path.join(args.datadir, 'val')
        testsampler = list(range(0, 50000, 50))
        testset = torch.utils.data.Subset(datasets.ImageFolder(valdir, transforms.Compose([
                                                                                transforms.Resize(256),
                                                                                transforms.CenterCrop(224),
                                                                                transforms.ToTensor(),])), 
                                                                testsampler)
        mvm_params['wbit_frac']         = 12
        mvm_params['ibit_frac']         = 12
        args.classes = 1000
    else:
        logging.info("EROOR: Invalid dataset")
        exit(0)
   
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)  

    if args.arch == 'resnet10':
        pretrained_model = resnet10_cifar.Model(args)
    elif args.arch == 'resnet20':
        pretrained_model = resnet20_cifar.Model(args)
    else:
        print("ERROR: Invalid Model Architecture")
        exit(0)

    checkpoint = torch.load(args.modelfile)
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.modelfile))

    if args.mvm:
        print("Copying ideal weights to mvm model")  
        if args.arch == 'resnet10':
            model_eval = resnet10_cifar.MVM_Model(args, mvm_params)
        elif args.arch == 'resnet20':
            model_eval = resnet20_cifar.MVM_Model(args, mvm_params)
        else:
            print("ERROR: Invalid Model Architecture")
            exit(0)      
        model_eval = ModelClone(model_to=model_eval, model_from=pretrained_model)
        save_path = os.path.join(args.customdir, args.dataset, "HW-in-Loop-WB-PGD-attacks", "{}-mvm-{}-eps{}-iter{}".format(
            modeltype, args.mvm_type, args.attack_epsilon, args.attack_iter))
    else:
        model_eval = pretrained_model
        save_path = os.path.join(args.customdir, args.dataset, 'Non-Adaptive-WB-attacks', "{}-eps{}-iter{}".format(
            modeltype, args.attack_epsilon, args.attack_iter))
    
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('saving logs to ==> {}'.format(save_path))
    logging.info('loading pretrained model from ==> {}'.format(args.modelfile))

    model_eval.cuda()
    model_eval.eval()

    clip_min = 0
    clip_max = 1
    epsilon = args.attack_epsilon/255.0
    a = args.attack_step_size/255.0
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    
    
    for i, (input, target) in enumerate(testloader):
                
        target = target.cuda()
        x_nat = input.cuda()
        target_var = target.cuda() 

        init_start = 2*epsilon*torch.rand_like(x_nat) - epsilon
        x = x_nat + init_start
        x = torch.clamp(x, clip_min, clip_max)

        for j in range(0,args.attack_iter):
            #print('{}/{} eps {} iter {}'.format(i, len(testloader), args.attack_epsilon, j))            
                            
            x.requires_grad = True
            x.grad = None
            out = model_eval(x)
            loss = loss_func(out, target)
            loss.backward()
            grad = x.grad.data
            x.requires_grad_(False)
            x.grad = None
            model_eval.zero_grad()
            x += a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + epsilon), x_nat - epsilon)
            x = torch.clamp(x, clip_min, clip_max)                
            
        adv_input = x.clone()

        if custom_data['data'] == []:          
            custom_data['data'] = adv_input.cpu().numpy()
            custom_data['label'] = target.cpu().numpy()
            custom_data['target'] = target.cpu().numpy()

        else:                
            custom_data['data']=np.append(custom_data['data'], adv_input.cpu().numpy(), axis=0)
            custom_data['label']=np.append(custom_data['label'], target.cpu().numpy(), axis=0)
            custom_data['target']=np.append(custom_data['target'], target.cpu().numpy(), axis=0)

        #f=os.path.join(save_path, 'custom_test-{}'.format(i))
        #print('saving upto batch {} of iter {} ==> {}'.format(i, f, args.attack_iter))
        #np.save(f, custom_data, allow_pickle=True)
        print('batch {} / {}'.format(i+1, len(testloader)))
        
      
    np.save(os.path.join(save_path, 'custom_test'), custom_data, allow_pickle=True)


      
