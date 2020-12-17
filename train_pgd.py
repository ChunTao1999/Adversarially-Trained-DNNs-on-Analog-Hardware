import sys
import os
sys.path.append('/home/nano01/a/roy77/AdversarialTraining_AnalogHardware/models/')
sys.path.append('/home/nano01/a/roy77/AdversarialTraining_AnalogHardware/configs')
sys.path.append('/home/nano01/a/roy77/AdversarialTraining_AnalogHardware/puma-functional-model-v3')

import pdb
import torch
import math
import time
import argparse 
import attacks
import logging
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
#import resnet32_cifar
#import resnet20_cifar
import resnet10_cifar
from utils import *


global best_val_acc
global best_train_acc

def train(train_loader, model, criterion, optimizer, epoch, attacker):
    global best_train_acc
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # adversarial attack
        input_var = attacker.perturb(input_var, target_var, model)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
    acc = top1.avg             
    return losses.avg, acc

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #---- System Setup
    parser.add_argument('--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 8)')
    parser.add_argument('--half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--seed', default=0, type=int,  help='provide seed')
    parser.add_argument('--print-freq', default=100, type=int,  help='frequency of printing batch results')
    parser.add_argument('--limit-test', default=0, type=int,  help='limit the number of batches to test')
    parser.add_argument('--debug', action='store_true',  help='enable debug mode')
    parser.add_argument('--store-act', action='store_true',  help='store activations')

    #---- Model setup
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices = ['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--classes', default=10, choices = [10, 100],  type=int, help='num classes')
    parser.add_argument('--datadir', default='/home/nano01/a/roy77/Datasets/', help='dataset folder')

    parser.add_argument('--logdir', default='./log/cifar10/', help='log dir')
    parser.add_argument('--arch', default='resnet20', choices=['resnet10', 'resnet20', 'resnet32'], help='model description')
    parser.add_argument('--inflate', default=1, type=int, help='model description')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)') 
    parser.add_argument('--pretrained', action='store', default=None,help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--batch-size', default=128, type=int,metavar='N', help='mini-batch size')

    #---- Training setup 
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

    #---- Adversarial Training setup
    parser.add_argument('--attack-iter', help='Adversarial attack iteration', type=int, default=50)
    parser.add_argument('--attack-epsilon', help='Adversarial attack maximal perturbation', type=int, default=8)
    parser.add_argument('--attack-step-size', help='Adversarial attack step size', type=float, default=1.0)

    
    args = parser.parse_args()
    #---- Default Settings
    args.arch = 'resnet10'

    #---- Setting Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    #---- Loss Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    
    #---- Attack Model during Training 
    attacker = attacks.LinfPGDAttack(clip_min=0, clip_max=1, epsilon=args.attack_epsilon/255.0, 
                                            k=args.attack_iter, a=args.attack_step_size/255.0, 
                                            random_start=True, loss_func = 'xent', prob_start_from_clean=0.1)
    
    save_path = os.path.join(args.logdir, 'pgd-linf-eps{}iter{}-{}w{}'.format(args.attack_epsilon, args.attack_iter, args.arch, args.inflate))
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    logging.info("Saving logs to ==> {}".format(save_path))

    #---- Preparing the Dataset 
    if args.dataset == 'cifar10': 
        logging.info('loading cifar 10 dataset')
        trainset    = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                                                                               transforms.RandomCrop(32, 4),
                                                                                                                               transforms.ToTensor()]))
        valset      = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transforms.ToTensor())
        testset     = torchvision.datasets.CIFAR10(root=args.datadir, train=False,  download=True, transform=transforms.ToTensor())
        args.classes = 10
    elif args.dataset == 'cifar100':
        logging.info('loading cifar 100 dataset')
        trainset    = torchvision.datasets.CIFAR100(root=args.datadir, train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                                                                               transforms.RandomCrop(32, 4),
                                                                                                                               transforms.ToTensor()]))
        valset      = torchvision.datasets.CIFAR100(root=args.datadir, train=True, download=True, transform=transforms.ToTensor())
        testset     = torchvision.datasets.CIFAR100(root=args.datadir, train=False,  download=True, transform=transforms.ToTensor())
        args.classes = 100
    else:
        logging.info("EROOR: Invalid dataset")
        exit(0)

    valid_size = 0.1
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler   = SubsetRandomSampler(valid_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, sampler=train_sampler)
    valloader   = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.workers, sampler=val_sampler)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    #---- Selecting the model
    if args.arch == 'resnet10':
        model = resnet10_cifar.Model(args)
    elif args.arch == 'resnet20':
        model = resnet20_cifar.Model(args)
    elif args.arch == 'resnet32':
        model = resnet32_cifar.Model(args)
    else:
        print("ERROR: Invalid Model Architecture")
        exit(0)

    model.cuda()   
    cudnn.benchmark = True
    logging.info(model)

    if args.resume: #---- For resuming interrupted training
        logging.info('==> Resuming from checkpoint.. {}'.format(args.resume))        
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_acc = checkpoint['valacc']
        best_train_acc = checkpoint['trainacc']
        best_test_acc = checkpoint['clean_testacc']
        args.start_epoch = checkpoint['epoch']
    elif args.pretrained: #---- for Finetuning
        logging.info('==> finetuning the model from .. {}'.format(args.pretrained))        
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_acc = 0
        best_train_acc = 0
        best_test_acc = 0
        args.start_epoch = 0
    else:
        logging.info('==> Initializing model parameters ...')
        best_test_acc = 0
        best_train_acc = 0
        best_val_acc = 0
        args.start_epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    for epoch in range(args.start_epoch+1, args.epochs+1):
    
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        [train_loss, trainacc] = train(trainloader, model, criterion, optimizer, epoch, attacker)
        lr_scheduler.step()
        logging.info('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}'.format(epoch, trainacc, train_loss))
        if trainacc > best_train_acc:
            best_train_acc = trainacc
            logging.info('Best Train Accuracy : {}'.format(best_train_acc))
        clean_attacker = attacks.NoAttack()
        [val_loss, valacc] = validate(valloader, model, criterion, attacker, model, args.print_freq, args.limit_test, args.debug)
        [clean_test_loss, clean_testacc] = validate(testloader, model, criterion, clean_attacker, model, args.print_freq, args.limit_test, args.debug)

        is_best = valacc > best_val_acc
        best_val_acc = max(valacc, best_val_acc)

        logging.info('Val Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}'.format(epoch, valacc, val_loss))
        logging.info('Test Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}'.format(epoch, clean_testacc, clean_test_loss))


        
        if is_best:
            torch.save({'state_dict': model.state_dict(), 'trainacc': trainacc, 'testacc' : clean_testacc, 'valacc': valacc, 'epoch' : epoch}, os.path.join(save_path, 'best_model.th'))
            print('Saving Best Model')
            best_clean_test_acc = clean_testacc
            best_train_acc = trainacc
            [best_adv_test_loss, best_adv_test_acc] = validate(testloader, model, criterion, attacker, model, args.print_freq, args.limit_test, args.debug)
            
        logging.info(' * Best Model Train Adversarial Prec@1 {}'.format(best_train_acc))
        logging.info(' * Best Model Val   Adversarial Prec@1 {}'.format(best_val_acc))
        logging.info(' * Best Model Test  Clean Prec@1 {}'.format(best_clean_test_acc))
        logging.info(' * Best Model Test  Adversarial Prec@1 {}'.format(best_adv_test_acc))
        
        
        results.add(epoch=epoch, train_loss=train_loss, test_loss=clean_test_loss, train_error1=100 - trainacc, test_error1=100 - clean_testacc)
        results.save()
        
