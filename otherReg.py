import torch
import numpy as np

################################################################################
##################################### OPTIONS ##################################
################################################################################


class l1l2reg: 
    def __init__(self,alpha,scaled=True, option= 'convs_and_batchnorm'):
        self.scaled=scaled
        self.option=option
        self.alpha=alpha
    #Computation of the current factor
    def __call__(self,net, lamb = 0.1):
        

        if self.option=='single_convs':
            reg= self.single_convs(net)

        if self.option=='convs_and_batchnorm':
            reg= self.convs_and_batchnorm(net)

        return lamb* reg

    def compatible_group_computation(self,group):
        alpha= self.alpha
        norm2= torch.norm(group,dim=1,p=2)
        if self.scaled:
            num_el_struct=group.size(1)
        else: num_el_struct =1

        reg=(alpha * (norm2**2).sum() + ((1-alpha) * norm2).sum() )*num_el_struct
                        
        return reg

    def single_convs(self,net):
        reg = 0    
        tot=0
        for m in net.modules():
            if isinstance(m,torch.nn.Conv2d):
                group = m.weight
                reg+=self.compatible_group_computation(group.reshape(group.size(0),-1))
                tot+=group.numel()

        if not self.scaled:
            tot=1

        return reg/tot

    #mode is supposed to  have a method to retrive conv-batchnorm blocks
    def convs_and_batchnorm(self,net):
        if isinstance(net,torch.nn.DataParallel):
            net=net.module
        reg = 0    
        tot=0
        for block in net.conv_batchnorm_blocks():
            conv = block['conv']
            bnorm = block['batchnorm']
            conv_w = conv.weight
            conv_w=conv_w.reshape(conv_w.size(0),-1)
            group = torch.cat((conv_w,bnorm.weight.unsqueeze(1),bnorm.bias.unsqueeze(1)),dim=1)
            reg+=self.compatible_group_computation(group)
            tot+=group.numel()

        if not self.scaled:
            tot=1

        return reg/tot


    
  