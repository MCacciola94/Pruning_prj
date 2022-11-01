import aux_tools as at
from torch.nn.utils import prune
import torch

def prune_thr(model, thr):
    for m in model.modules(): 
      if isinstance(m,torch.nn.Conv2d) or isinstance(m,torch.nn.BatchNorm2d):
          pruning_par=[((m,'weight'))]
  
          if hasattr(m, 'bias') and not(m.bias==None):
              pruning_par.append((m,'bias'))
  
          prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=thr)

def prune_struct(model, thr = 0.05,struct= 'convs_and_batchnorm'):
    if struct== 'convs_and_batchnorm':
        if isinstance(model,torch.nn.DataParallel):
            model=model.module

        for block in model.conv_batchnorm_blocks():
            conv = block['conv']
            bnorm = block['batchnorm']
            for i in range(conv.out_channels):
                if conv.weight_mask[i,:].sum()/conv.weight_mask[i,:].numel()>thr:
                    conv.weight_mask[i,:]=1
                    bnorm.weight_mask[i]=1
                    bnorm.bias_mask[i]=1
                else:
                    conv.weight_mask[i,:]=0
                    bnorm.weight_mask[i]=0
                    bnorm.bias_mask[i]=0


    elif struct == 'single_convs':
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):
                for i in range(m.out_channels):
                    if m.weight_mask[i,:].sum()/m.weight_mask[i,:].numel()>thr:
                        m.weight_mask[i,:]=1
                    else:
                        m.weight_mask[i,:]=0

def param_saving(layers, skip = 1 , freq = 2, filter_size = 9):
    layers = layers[1:]
    first = 0
    second = 1
    tot = 0
    while second < len(layers):
        pruned_filetrs = sum([int(e) for e in layers[first]])
        rem_filters = len(layers[second]) - sum([int(e) for e in layers[second]])
        print(first, second, pruned_filetrs, rem_filters )
        tot += filter_size * pruned_filetrs * rem_filters
        first += freq
        second += freq
    return tot