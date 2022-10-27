import torch
from torch.nn.utils import prune
import torch.nn as nn
import numpy as np

#Pruning crieterion for unstructured threshold pruning
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


#Computing sparsity information of the model
def sparsityRate(model,verb_lev=-1,opt="channels"):
    # breakpoint()

    #retrocompatible with previous version
    if verb_lev==False:
        verb_lev=0
    elif verb_lev== True:
        verb_lev=1

    out=[]
    tot_pruned=0
    tot_struct_pruned=0
    for m in model.modules():
        #Fully Connected layers
        if isinstance(m,nn.Linear):
            v=[]
            for i in range(m.out_features):
                el=float((m.weight[i,:]==0).sum()/m.weight[i,:].numel())
                v=v+[el]
                tot_pruned+=m.in_features*el
                if el==1.0:
                    tot_struct_pruned+=m.in_features
                    
                
            if verb_lev==1:
                print("in module ",m,"\n sparsity of  ", v)
            elif verb_lev==0:
                print("\n sparsity of  ", v)
            out=out+[v]

        #Convolutional layers 
        if isinstance(m,torch.nn.Conv2d):
            if opt=="channels":
                v=[]
                for i in range(m.out_channels):
                    el= float((m.weight[i,:,:,:]==0).sum()/m.weight[i,:,:,:].numel())
                    v=v+[el]

                    tot_pruned+=m.kernel_size[0]*m.kernel_size[1]*m.in_channels*el
                    if el==1.0:
                        tot_struct_pruned+=m.kernel_size[0]*m.kernel_size[1]*m.in_channels

                if verb_lev==1:
                    print("in module ",m,"\n sparsity of  ", v)
                elif verb_lev==0:
                    print("\n sparsity of  ", v)
                out=out+[v]
            else:
                v=[]
                for i in range(m.out_channels):
                    for j in range(m.in_channels):
                        el= float((m.weight[i,j,:,:]==0).sum()/m.weight[i,j,:,:].numel())
                        v=v+[el]

                        tot_pruned+=m.kernel_size[0]*m.kernel_size[1]*el
                        if el==1.0:
                            tot_struct_pruned+=m.kernel_size[0]*m.kernel_size[1]

                        if verb_lev==1:
                            print("in module ",m,"\n sparsity of  ", v)
                        elif verb_lev==0:
                            print("\n sparsity of  ", v)

                        out=out+[v]

    return out,(tot_pruned,tot_struct_pruned)


#method that prune neurons of the model based on their sparsity
def thresholdNeuronPruning(module,mask,threshold=0.95):

    #Fully Connected layers
   if isinstance(module,nn.Linear):
        for i in range(module.out_features):
            if float((module.weight[i,:]==0).sum()/module.weight[i,:].numel())>threshold:
                mask[i]=0

    #Convolutional layers    
   if isinstance(module,nn.Conv2d):
        for i in range(module.out_channels):
            if float((module.weight[i,:,:,:]==0).sum()/module.weight[i,:,:,:].numel())>threshold:
                mask[i]=0
            
   return 1-mask.sum()/mask.numel()            


# compute the maximum weight for each neuron
def maxVal(model): 
    out=[]
    for m in model.modules():
          #Fully Connected layers
          if isinstance(m,nn.Linear):
                # for i in range(m.out_features):  
                #     v=v+ [float(torch.norm(m.weight[i,:],np.inf))]
                v=torch.norm(m.weight,dim=1,p=np.inf)
          else:
          #Convolutional layers
                if isinstance(m,nn.Conv2d):
                # for i in range(m.out_channels):  
                #     v=v+[float(torch.norm(m.weight[i,:,:,:],np.inf))]
                    v=torch.norm(m.weight,dim=(1,2,3),p=np.inf)
                else:
                    continue

          print("\nmax weight is ", v)
          out+=[v]
    return out


def layerwise_M(model, const = False, scale = 1.0):
       Mdict={}
       if const:
           for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=1.0
       else:
            for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=torch.norm(m.weight,p=np.inf).item() * scale

       return Mdict

def paramwise_M(model, const = False, scale = 1.0):
       Mdict={}
       model.eval()
       with torch.no_grad:
        if const:
            for par in model.parameters():
                    if par.requires_grad:
                        Mdict[par]=scale
        else:
                for par in model.parameters():
                    if par.requires_grad:
                        Mdict[par]=torch.norm(par.data,p=np.inf).item() * scale

        return Mdict


def noReg(net, loss, lamb=0.1):
    return loss,0

