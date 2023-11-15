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
import torch.autograd as autograd
import resnet20_cifar
import resnet10_cifar
import resnet20_new_intersmoothness_conv232_conv332
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

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    
    #---- Result saving
    #parser.add_argument('--filename', help='the filename to save the results')
    
    args = parser.parse_args()

    args.custom_norm = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.store_act = False

    #args.arch = 'resnet10'
    #args.inflate = 1
    #args.dataset = 'cifar10'
    #args.pretrained = './log/cifar10/clean-resnet10w1'
    #args.pretrained = './log/cifar10/pgd-linf-eps8iter50-resnet10w1'
    #args.batch_size = 1 #keep the batch size 1 to plot a distribution of interlayer smoothness
    #args.limit_test = 1000
    
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
        args.classes = 10
    elif args.dataset == 'cifar100':
        print('loading cifar 100 dataset')
        testset     = torchvision.datasets.CIFAR100(root=args.datadir, train=False,  download=True, transform=transform)
        args.classes = 100
    else:
        print("ERROR: Invalid dataset")
        exit(0)

    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #----- Loading Model 

    if args.arch == 'resnet20':
        pretrained_model = resnet20_new_intersmoothness_conv232_conv332.Model(args)
    elif args.arch == 'resnet10':
        pretrained_model = resnet10_new_intersmoothness_conv211_conv322.Model(args)
    else: 
        print("ERROR: Invalid Model Architecture")
        exit(0)

    checkpoint = torch.load(os.path.join(args.pretrained, 'best_model.th'))
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    print('loading pretrained model from ==> {}'.format(args.pretrained))

    model_eval = pretrained_model   
    
    
    save_path = os.path.join(args.pretrained, "clean", "interlayer_smoothness")
        
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    logging.info('saving logs to ==> {}'.format(save_path))
    
    model_eval.cuda()
    model_eval.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()    
    end = time.time()
    
    intersmoothness = []

    for i, (input, target) in enumerate(testloader):
        print('{}/{}'.format(i+1, len(testloader)))
        input_var = input.cuda()
        target_var = target.cuda()
        
        output_midclean, residual = model_eval.forward1(input_var)
        
        output_midclean = output_midclean.cuda()
        residua = residual.cuda()

        output_finalclean = model_eval.forward2_clean(output_midclean, residual)
        output_midnoisy, output_finalnoisy = model_eval.forward2_noisy(output_midclean, residual)
        
        jacobian = torch.autograd.functional.jacobian(model_eval.forward2_noisy, (output_midclean, residual))

        output_midclean_norm = torch.norm(output_midclean)
        output_finalclean_norm = torch.norm(output_finalclean)

        #pdb.set_trace()
        jacobian_useful = jacobian[1][0].squeeze()
        #pdb.set_trace()
        jacobian_useful = jacobian_useful.reshape(jacobian_useful.shape[0], -1)
        output_midnoisy = output_midnoisy.squeeze().flatten()
        lhs_norm = torch.norm(output_finalnoisy-torch.matmul(jacobian_useful, output_midnoisy))
        #pdb.set_trace()
        eta_norm = torch.norm(output_midnoisy-output_midclean.squeeze().flatten())
        
        intersmoothness = np.append(intersmoothness, lhs_norm.item() * output_midclean_norm.item() / output_finalclean_norm.item() / eta_norm.item())
        #print(jacobian.size())  
        #pdb.set_trace()
        if (i+1 == args.limit_test):
            break
        
    #pdb.set_trace()
    print(np.mean(intersmoothness))
    #np.save(os.path.join(save_path, 'interlc.npy'.format(args.batch_size*args.limit_test)), interlc, allow_pickle=True)
    if os.path.exists(os.path.join(save_path, 'conv232_conv332.csv'.format(args.batch_size*args.limit_test))):
        os.remove(os.path.join(save_path, 'conv232_conv332.csv'.format(args.batch_size*args.limit_test)))
        print("Old File conv232_conv332.csv Removed")

    else:
        print("The file did not exist before")

    np.savetxt(os.path.join(save_path, 'conv232_conv332.csv'.format(args.batch_size*args.limit_test)), intersmoothness, delimiter = ',')
    print("New file conv232_conv332.csv saved")

    print(args.pretrained)
    print("conv232_conv332")

