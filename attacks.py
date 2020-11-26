#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:12:54 2019

@author: tibrayev
"""
import pdb
import torch
import torch.nn as nn


#def FGSM(clip_min, clip_max, epsilon, image, data_grad):
#    # Collect the element-wise sign of the data gradient
#    sign_data_grad = data_grad.sign()
#    # Create the perturbed image by adjusting each pixel of the input image
#    perturbed_image = image + epsilon * sign_data_grad
#    # Adding clipping to maintain image in the range [clip_min, clip_max]
#    perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)
#    # Return the perturbed image
#    return perturbed_image

class NoAttack:
    def __init__(self):
        """Attack parameter initialization. The attack performs single step of size epsilon."""
                
    def perturb(self, x_nat, y, model):
        return x_nat

class LinfFGSMSingleStepAttack:
    def __init__(self, clip_min, clip_max, epsilon, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs single step of size epsilon."""
        super(LinfFGSMSingleStepAttack, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon  = epsilon
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model):
        x = x_nat.clone().detach()
        x.requires_grad_(True)
        x.grad = None
        out = model(x)
        loss = self.loss_func(out, y)
        loss.backward()
        grad = x.grad.data # get gradients
        x.requires_grad_(False)
        x.grad = None
        model.zero_grad()
        x += self.epsilon * torch.sign(grad)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfFGSMSingleStepAttack_with_normalization:
    def __init__(self, clip_min, clip_max, epsilon, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs single step of size epsilon."""
        super(LinfFGSMSingleStepAttack_with_normalization, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon  = epsilon
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model, normalization_function):
        x = x_nat.clone().detach()
        x.requires_grad_(True)
        x.grad = None
        x_norm = normalization_function(x)
        out = model(x_norm)
        loss = self.loss_func(out, y)
        loss.backward()
        grad = x.grad.data # get gradients
        x.requires_grad_(False)
        x.grad = None
        model.zero_grad()
        x += self.epsilon * torch.sign(grad)
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfPGDAttack:
    def __init__(self, clip_min, clip_max, epsilon, k, a, random_start=True, loss_func = 'xent', prob_start_from_clean=0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfPGDAttack, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.clean = prob_start_from_clean
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            init_start = 2*self.epsilon*torch.rand_like(x_nat) - self.epsilon
            start_from_noise =  torch.gt(torch.rand(1), self.clean)
            x = x_nat + init_start*start_from_noise.float().cuda()
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            x.requires_grad = True
            x.grad = None
            out = model(x)
            loss = self.loss_func(out, y)            
            loss.backward()
            grad = x.grad.data
            x.requires_grad_(False)
            x.grad = None
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)

        return x

class LinfPGDAttack_with_normalization:
    def __init__(self, clip_min, clip_max, epsilon, k, a, random_start, loss_func = 'xent', prob_start_from_clean=0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfPGDAttack_with_normalization, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.clean = prob_start_from_clean
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model, normalization_function):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            init_start = 2*self.epsilon*torch.rand_like(x_nat) - self.epsilon
            start_from_noise =  torch.gt(torch.rand(1), self.clean)
            x = x_nat + init_start*start_from_noise.float().cuda()
            x = torch.clamp(x, self.clip_min, self.clip_max)
            
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            x.requires_grad_(True)
            x.grad = None
            x_norm = normalization_function(x)
            out = model(x_norm)
            loss = self.loss_func(out, y)
            loss.backward()
            grad = x.grad.data # get gradients
            x.requires_grad_(False)
            x.grad = None
            model.zero_grad()
            x += self.a * torch.sign(grad)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

class LinfPGDAttackEnsemble:
    def __init__(self, clip_min, clip_max, epsilon, k, a, random_start=True, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        super(LinfPGDAttackEnsemble, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        if loss_func == 'xent':
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            raise ValueError('C&W loss is not implemented!')
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
    def perturb(self, x_nat, y, model_list):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + (2*self.epsilon)*torch.rand_like(x_nat) - self.epsilon
            x = torch.clamp(x, self.clip_min, self.clip_max)
        else:
            x = x_nat.clone().detach()
        
        for i in range(self.k):
            grad = torch.zeros(len(model_list), x.size(0), x.size(1), x.size(2), x.size(3)).to(x.device)
            for m in range(len(model_list)):
                x.requires_grad_(True)
                x.grad = None
                out = model_list[m](x)
                loss = self.loss_func(out, y)
                loss.backward()
                grad[m] = x.grad.data # get gradients
                x.requires_grad_(False)
                x.grad = None
                model_list[m].zero_grad()
            grad_average = grad.mean(dim=0)           
            x += self.a * torch.sign(grad_average)
            x = torch.max(torch.min(x, x_nat + self.epsilon), x_nat - self.epsilon)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x
