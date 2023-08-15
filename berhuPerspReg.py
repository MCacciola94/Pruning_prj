import torch
import numpy as np

################################################################################
##################################### OPTIONS ##################################
################################################################################


class perspBerhu: 
    def __init__(self,alpha,M, track_stats = False):
        self.alpha=alpha
        self.M=M
        self.const=(torch.sqrt(torch.Tensor([alpha/(1-alpha)]))).cuda()
        # self.option=option
        self.track_stats=track_stats
        self.stats={'case1':0,'case2':0,'case3':0}
    #Computation of the current factor
    def __call__(self,net, lamb = 0.1):


        reg= self.compute_by_convs(net)

        return lamb* reg

    def berhu_computation(self,weights, M):
        alpha=self.alpha
        const=self.const
        
        absval=torch.abs(weights)       


        bo1 = max(1/M,const.item())*absval>=1
        reg1 = alpha*absval**2+1-alpha

        bo2 = 1/M<=const
        reg2=2*(alpha*(1-alpha))**0.5*absval

        reg3=alpha* absval*(M+(1-alpha)/M)

        bo2=torch.logical_and(bo2, torch.logical_not(bo1))
        bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

        reg=(bo1*reg1+bo2*reg2+bo3*reg3).sum()
        if self.track_stats:
            self.stats['case1']+=bo1.sum().item()
            self.stats['case2']+=bo2.sum().item()
            self.stats['case3']+=bo3.sum().item()
                        
        return reg

    def compute_by_convs(self,net):
        reg = 0    

        for m in net.modules():
            if isinstance(m,torch.nn.Conv2d):
                group = m.weight
                reg+=self.berhu_computation(group.reshape(group.numel()), self.M[m])
                

        return reg

 