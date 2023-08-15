import resnet, resnet_pruned, resnetBig_imgNet
import resnetBig, resnetBig_pruned,resnetBig_imgNet_pruned
import vgg, vgg_pruned
from torch.nn.utils import prune
import torch
import aux_tools as at
import os
import vit, vit_pruned
# import torchvision.models as models

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

available_arcs= model_names + ["resnet18", "resnet50", "vgg16", 'ViTsmall']

def is_available(name):
    return name in available_arcs

def load_arch(name, num_classes, resume = "", already_pruned = True):
    if not(is_available(name)):
        print("Architecture requested not available")
        return None

    if num_classes <=100:
        if name in model_names:
            model = torch.nn.DataParallel(resnet.__dict__[name](num_classes))
        else:
            if "vgg" in name:
                model = vgg.__dict__[name]()
            elif 'ViT' in name:
                model = vit.__dict__[name]()
            else:
                model = resnetBig.__dict__[name](num_classes)
    else:
        model = resnetBig_imgNet.__dict__[name]()
        # model = torch.nn.DataParallel(model)    
    
    model.cuda()

    if not(resume == "")  and already_pruned:
        for m in model.modules(): 
            if isinstance(m,torch.nn.Linear):
                continue
            if hasattr(m, 'weight'):
                pruning_par=[((m,'weight'))]

                if hasattr(m, 'bias') and not(m.bias==None):
                    pruning_par.append((m,'bias'))

                prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=1e-18)

                
    # optionally resume from a checkpoint

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"# (epoch {})"
                .format(checkpoint['best_prec1']))#, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return model

def load_arch_pruned(name, num_classes, resume = "", already_pruned = True):
    if not(is_available(name)):
        print("Architecture requested not available")
        return None

    if num_classes <=100:
        if name in model_names:
            model = torch.nn.DataParallel(resnet_pruned.__dict__[name](num_classes))
        else:
            if "vgg" in name:
                model = vgg_pruned.__dict__[name]()
            elif 'ViT' in name:
                model = vit_pruned.__dict__[name]()
            else:
                model = resnetBig_pruned.__dict__[name](num_classes)
    else:
        model = resnetBig_imgNet_pruned.__dict__[name]()
        # model = torch.nn.DataParallel(model)    
    model.cuda()

                
    # optionally resume from a checkpoint

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint['best_prec1'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return model