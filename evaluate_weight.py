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
import resnet10_cifar
from mvm_params import *
from custom_dataset import *

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
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--customdir', default='./customdata/cifar100/Non-Adaptive-WB-attacks/')
    parser.add_argument('--customdata')

    #---- MVM Arguments (default arguments are for 16 bit fixed point evaluation on MVM
    parser.add_argument('--mvm', default=False, help='evaluate on PUMA', action='store_true')
    parser.add_argument('--genieX', default=False, help='evaluate with GenieX', action='store_true')
    parser.add_argument('--mvm-type', choices=['16x16_100k', '32x32_100k', '64x64_100k', '64x64_50k', '64x64_300k', 'ideal'], help='mvm model')
    
    #---- Model setup
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100', choices = ['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--classes', default=10, choices = [10, 100],  type=int, help='num classes')
    parser.add_argument('--datadir', default='/home/nano01/a/tao88/Datasets/', help='dataset folder')

    parser.add_argument('--arch', default='resnet20', choices=['resnet10', 'resnet20', 'resnet32'], help='model description')
    parser.add_argument('--inflate', default=1, type=int, help='model description')
    parser.add_argument('--pretrained', action='store', default=None,help='the path to the pretrained model')
    parser.add_argument('--batch-size', default=10, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--custom-norm', default=True)

    args = parser.parse_args()
 

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.custom_norm == True:
        transform=transforms.ToTensor()
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    #---- Preparing the Dataset 

    if args.customdata:
        path = os.path.join(args.customdir, args.customdata)
        print('loading custom dataset from ==>', path)
        testset     = CUSTOM_DATA(root=path, split='test', transform=transforms.ToTensor()) 
        if args.dataset == 'cifar10': 
            print('loading cifar 10 custom dataset from ==> {}'.format(path))
            mvm_params['wbit_frac']         = 13
            mvm_params['ibit_frac']         = 13 
            args.classes = 10
        elif args.dataset == 'cifar100':
            print('loading cifar 100 custom dataset from ==> {}'.format(path))
            mvm_params['wbit_frac']         = 12
            mvm_params['ibit_frac']         = 12
            args.classes = 100
        else:
            print.info("ERROR: Invalid dataset")
            exit(0)           
    elif args.dataset == 'cifar10': 
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
        logging.info("ERROR: Invalid dataset")
        exit(0)

    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #----- default no attack 
    attacker = attacks.NoAttack()
    attack_model = [] 

    #----- Loading Defending Model 

    if args.arch == 'resnet10':
        pretrained_model = resnet10_cifar.Model(args)
    elif args.arch == 'resnet20':
        pretrained_model = resnet20_cifar.Model(args)
    else: 
        print("ERROR: Invalid Model Architecture")
        exit(0)

    checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.th'))
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.pretrained))
    
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
        save_path = os.path.join(args.pretrained, "test_mvm-{}".format(args.mvm_type))
    else:
        model_eval = pretrained_model
        save_path = os.path.join(args.pretrained, "test")


    if args.customdata:
        save_path = os.path.join(save_path, args.customdata)
    else:
        save_path = os.path.join(save_path, "clean")
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    logging.info('saving logs to ==> {}'.format(save_path))
    
    model_eval.cuda()
    model_eval.eval()
    if args.customdata:
        [test_loss, testacc] = custom_validate(testloader, model_eval, criterion, attacker, attack_model, args.print_freq, args.limit_test, args.debug)
    else:
        [test_loss, testacc] = validate(testloader, model_eval, criterion, attacker, attack_model, args.print_freq, args.limit_test, args.debug)
    logging.info('Test \t({:.2f}%)]\tLoss: {:.6f}'.format(testacc, test_loss))

