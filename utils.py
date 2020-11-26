import sys
sys.path.append("./puma-functional-model-v3")
import pdb
import time
import logging
import torch
import torch.nn as nn
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
from pytorch_mvm_class_v3 import Conv2d_mvm, Linear_mvm
import numpy as np

def ModelClone(model_to, model_from):
    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    #pdb.set_trace()

    for m in model_from.modules():
        if isinstance(m, nn.Conv2d):
            weights_conv.append(m.weight.data.clone())
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            bn_data.append(m.weight.data.clone())
            bn_bias.append(m.bias.data.clone())
            running_mean.append(m.running_mean.data.clone())
            running_var.append(m.running_var.data.clone())
            num_batches.append(m.num_batches_tracked.clone())
        elif isinstance(m, nn.Linear):
            weights_lin.append(m.weight.data.clone())

    i=0
    j=0
    k=0
    for m in model_to.modules():
        
        if isinstance(m, Conv2d_mvm):
            m.weight.data = weights_conv[i]
            i = i+1
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        elif isinstance(m, Linear_mvm):
            m.weight.data = weights_lin[k]
            k=k+1

    return model_to


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class ResultsLog(object):

    def __init__(self, path='results.txt', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

   # def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
     #   self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(val_loader, model, criterion, attacker, attack_model, print_freq, limit_test, debug):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()    
    end = time.time()
    model.eval()
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        input_var = attacker.perturb(input_var, target_var, attack_model)
        # compute output
        output = model(input_var)
        # ---- insert debugging code if needed using debug flag
        if debug:
            pdb.set_trace()

        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #pdb.set_trace()
        if (i+1) % print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i+1, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        if i+1 == limit_test:
            break

    acc = top1.avg

    return losses.avg, acc


def custom_validate(val_loader, model, criterion, attacker, attack_model, print_freq, limit_test, debug):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()    
    end = time.time()
    
    for i, (input, target, label) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        label = label.cuda()
        input_var = attacker.perturb(input_var, target_var, attack_model)
        # compute output
        output = model(input_var)
        # ---- insert debugging code if needed using debug flag
        if debug:
            pdb.set_trace()

        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, label)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #pdb.set_trace()
        if (i+1) % print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i+1, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        if i+1 == limit_test:
            break

    acc = top1.avg

    return losses.avg, acc

def ModelClone(model_to, model_from):
    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    #pdb.set_trace()

    for m in model_from.modules():
        if isinstance(m, nn.Conv2d):
            weights_conv.append(m.weight.data.clone())
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            bn_data.append(m.weight.data.clone())
            bn_bias.append(m.bias.data.clone())
            running_mean.append(m.running_mean.data.clone())
            running_var.append(m.running_var.data.clone())
            num_batches.append(m.num_batches_tracked.clone())
        elif isinstance(m, nn.Linear):
            weights_lin.append(m.weight.data.clone())

    i=0
    j=0
    k=0
    for m in model_to.modules():
        
        if isinstance(m, Conv2d_mvm):
            m.weight.data = weights_conv[i]
            i = i+1
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        elif isinstance(m, Linear_mvm):
            m.weight.data = weights_lin[k]
            k=k+1

    return model_to

def imnet_validate(val_loader, model, criterion, attacker, attack_model, print_freq, limit_test, debug):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter() 
    top5 = AverageMeter()
    end = time.time()
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        
        input_var = attacker.perturb(input_var, target_var, attack_model)
        # compute output
        output = model(input_var)
        # ---- insert debugging code if needed using debug flag
        if debug:
            pdb.set_trace()

        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #pdb.set_trace()
        if (i+1) % print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i+1, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        if i+1 == limit_test:
            break

    acc = top1.avg
    acc5 = top5.avg

    return losses.avg, acc, acc5

def get_data_indices(args):
    assert 'partition' in args, \
        'partition argumenet is expected but not present in args'
    assert 'partition_size' in args, \
        'partition_size argumenet is expected but not present in args'

    data_indices = {}
    data_indices['start_idx'] = args.partition * args.partition_size
    data_indices['end_idx'] = (args.partition + 1) * args.partition_size
    return data_indices

#### Square Attack Utils

def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = np.random.choice(lst_classes)
    return y_test_new


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def predict(val_loader, model):
    """
    Run prediction, return logits
    """
    batch_time = AverageMeter()
    logits_list = []
    end = time.time()
    model.eval()
    
    for i, (input, target) in enumerate(val_loader):
        input_var = input.cuda()
        target_var = target.cuda()
        # compute output
        output = model(input_var)
        # ---- insert debugging code if needed using debug flag
        output = output.float()
        
        logits = output.cpu().detach().numpy()       
        logits_list.append(logits)
                        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #pdb.set_trace()
        logging.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                         i+1, len(val_loader), batch_time=batch_time))
    logits = np.vstack(logits_list)

    return logits

