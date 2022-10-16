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
import resnet20_cifar_convout
import resnet10_cifar
import resnet10_cifar_in_out
import resnet10_cifar_convout
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

    #args.mvm = True 
    #args.mvm_type = '32x32_100k'
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
        pretrained_model = resnet20_cifar_convout.Model(args)
    elif args.arch == 'resnet32':
        pretrained_model = resnet32_cifar.Model(args)
    elif args.arch == 'resnet10':
        pretrained_model = resnet10_cifar_convout.Model(args)
    else: 
        print("ERROR: Invalid Model Architecture")
        exit(0)

    checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.th'))
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.pretrained))

    #pdb.set_trace()
    model_eval = pretrained_model
    
    if args.arch == 'resnet20':
        mvm_model_eval = resnet20_cifar_convout.MVM_Model(args, mvm_params)
    elif args.arch == 'resnet32':
        mvm_model_eval = resnet32_cifar.MVM_Model(args, mvm_params)
    elif args.arch == 'resnet10':
        mvm_model_eval = resnet10_cifar_convout.MVM_Model(args, mvm_params)
    else:
        print("ERROR: Invalid Model Architecture")
        exit(0)      
    
    mvm_model_eval = ModelClone_hybrid(model_to=mvm_model_eval, model_from=pretrained_model)
    
    save_path = os.path.join(args.pretrained, "clean", "digital_vs_{}_findstd".format(args.mvm_type))
        
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
        mvm_output = {}
        
        if args.arch == 'resnet10':
            mvm_output['conv0'] = mvm_model_eval(input_var, 'conv0')
            if args.inflate > 1:
                mvm_output['resconv11'] = mvm_model_eval(input_var, 'resconv11')
            mvm_output['conv111'] = mvm_model_eval(input_var, 'conv111')
            mvm_output['conv112'] = mvm_model_eval(input_var, 'conv112')
            mvm_output['resconv21'] = mvm_model_eval(input_var, 'resconv21')
            mvm_output['conv211'] = mvm_model_eval(input_var, 'conv211')
            mvm_output['conv212'] = mvm_model_eval(input_var, 'conv212')
            mvm_output['resconv31'] = mvm_model_eval(input_var, 'resconv31')
            mvm_output['conv311'] = mvm_model_eval(input_var, 'conv311')
            mvm_output['conv312'] = mvm_model_eval(input_var, 'conv312')
            mvm_output['conv321'] = mvm_model_eval(input_var, 'conv321')
            mvm_output['conv322'] = mvm_model_eval(input_var, 'conv322')
            mvm_output['linear'] = mvm_model_eval(input_var, 'linear')
        
        if args.arch == 'resnet20':
            mvm_output['conv0'] = mvm_model_eval(input_var, 'conv0')
            if args.inflate > 1:
                mvm_output['resconv11'] = mvm_model_eval(input_var, 'resconv11')
            mvm_output['conv111'] = mvm_model_eval(input_var, 'conv111')
            mvm_output['conv112'] = mvm_model_eval(input_var, 'conv112')
            mvm_output['conv121'] = mvm_model_eval(input_var, 'conv121')
            mvm_output['conv122'] = mvm_model_eval(input_var, 'conv122')
            mvm_output['conv131'] = mvm_model_eval(input_var, 'conv131')
            mvm_output['conv132'] = mvm_model_eval(input_var, 'conv132')
            mvm_output['resconv21'] = mvm_model_eval(input_var, 'resconv21')
            mvm_output['conv211'] = mvm_model_eval(input_var, 'conv211')
            mvm_output['conv212'] = mvm_model_eval(input_var, 'conv212')
            mvm_output['conv221'] = mvm_model_eval(input_var, 'conv221')
            mvm_output['conv222'] = mvm_model_eval(input_var, 'conv222')
            mvm_output['conv231'] = mvm_model_eval(input_var, 'conv231')
            mvm_output['conv232'] = mvm_model_eval(input_var, 'conv232')
            mvm_output['resconv31'] = mvm_model_eval(input_var, 'resconv31')
            mvm_output['conv311'] = mvm_model_eval(input_var, 'conv311')
            mvm_output['conv312'] = mvm_model_eval(input_var, 'conv312')
            mvm_output['conv321'] = mvm_model_eval(input_var, 'conv321')
            mvm_output['conv322'] = mvm_model_eval(input_var, 'conv322')
            mvm_output['conv331'] = mvm_model_eval(input_var, 'conv331')
            mvm_output['conv332'] = mvm_model_eval(input_var, 'conv332')
            mvm_output['linear'] = mvm_model_eval(input_var, 'linear')
        
        #pdb.set_trace()
        err_batch = {}
        for key, value in digital_output.items():
            #err_batch[key] = digital_output[key]-mvm_output[key]
            #if i == 0:
            #    dig_vs_mvm[key] = err_batch[key]
            #else:
            #    dig_vs_mvm[key] = torch.cat((dig_vs_mvm[key], err_batch[key]))
                
            digital_output[key] = digital_output[key].detach().cpu().numpy().reshape([args.batch_size, -1])
            #print(mvm_output[key].type())
            mvm_output[key] = mvm_output[key].detach().cpu().numpy().reshape([args.batch_size, -1])
            
            err_batch[key] = digital_output[key] - mvm_output[key]

            if i == 0:
                dig_vs_mvm[key] = err_batch[key]     
            else:
                dig_vs_mvm[key] = np.append(dig_vs_mvm[key], err_batch[key])
        
        #pdb.set_trace()
        if (i+1)%args.limit_test == 0:
            break
    
    logging.info("Gaussian noise std to use for NVM crossbar model:{}, pretrained_model:{}".format(args.mvm_type, args.pretrained))
    dig_vs_mvm_std = {}
    for key, value in dig_vs_mvm.items():
        std = np.std(dig_vs_mvm[key])
        logging.info('{} {}'.format(key, std))
        dig_vs_mvm_std[key] = std
        
    if os.path.exists(os.path.join(save_path, 'dig_vs_mvm_std_N{}.npy'.format(args.batch_size*args.limit_test))):
        os.remove(os.path.join(save_path, 'dig_vs_mvm_std_N{}.npy'.format(args.batch_size*args.limit_test)))
        logging.info("Old File Removed")
    else:
        logging.info("The file did not exist before")
        np.save(os.path.join(save_path, 'dig_vs_mvm_std_N{}.npy'.format(args.batch_size*args.limit_test)), dig_vs_mvm_std, allow_pickle=True)
        logging.info('Saved results dig_vs_mvm_std_N{} to ==> {}'.format(args.batch_size*args.limit_test, save_path))

