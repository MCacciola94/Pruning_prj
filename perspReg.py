import torch
import numpy as np

################################################################################
##################################### OPTIONS ##################################
################################################################################


class PerspReg: 
    def __init__(self,alpha,M, option= 'convs_and_batchnorm'):
        self.alpha=alpha
        self.M=M
        self.const=(torch.sqrt(torch.Tensor([alpha/(1-alpha)]))).cuda()
        self.option=option
    #Computation of the current factor
    def __call__(self,net, lamb = 0.1):

        if self.option=='single_convs':
            reg= self.single_convs(net)

        if self.option=='convs_and_batchnorm':
            reg= self.convs_and_batchnorm(net)

        return lamb* reg

    def compatible_group_computation(self,group, M):
        reg = 0 
        alpha=self.alpha
        const=self.const
        
        norminf=torch.norm(group,dim=1,p=np.inf)
        norm2= torch.norm(group,dim=1,p=2)
        num_el_struct=group.size(1)
        


        bo1 = torch.max(norminf/M,const*norm2)>=1
        reg1 = norm2**2+1-alpha

        bo2 = norminf/M<=const*norm2
        reg2=const*norm2*(1+alpha)

        eps=(torch.zeros(norminf.size())).cuda()
        eps=eps+1e-10
        reg3=norm2**2/(torch.max(eps,norminf))*M+(1-alpha)*norminf/M

        bo2=torch.logical_and(bo2, torch.logical_not(bo1))
        bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

        reg+=(bo1*reg1+bo2*reg2+bo3*reg3).sum()*num_el_struct
                        
        return reg

    def single_convs(self,net):
        reg = 0    
        tot=0
        for m in net.modules():
            if isinstance(m,torch.nn.Conv2d):
                group = m.weight
                reg+=self.compatible_group_computation(group.reshape(group.size(0),-1), self.M[m])
                tot+=group.numel()

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
            reg+=self.compatible_group_computation(group,self.M[block['id']])
            tot+=group.numel()

        return reg/tot


    
  
    
    #Computation of y variables gradients and values
    def yGrads(self,model):
        grads=[]
        Y=[]
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d): 
                g,y=self.yGrad(m)
                grads=grads+[(g)]
                Y=Y+[y]
        return grads, Y

    def yGrad(self,m):
        M=self.M[m]
        alpha=self.alpha
        out=[]
        ys=[]

        #Fully Connected layers
        if isinstance(m,torch.nn.Linear):
            
            for i in range(m.out_features):
                a=max(torch.abs(m.weight[i,:]).view(m.weight[i,:].numel()))
                b=torch.norm(m.weight[i,:])*np.sqrt(alpha/(1-alpha))
                y=min(max(a/M,b),torch.tensor(1).cuda())
                ys=ys+[y.item()]
                out=out+[(-alpha*(torch.norm(m.weight[i,:])**2)/y**2+1-alpha).item()]
                
        #Convolutional layers       
        if isinstance(m,torch.nn.Conv2d):
            
            for i in range(m.out_channels):
                a=max(torch.abs(m.weight[i,:,:,:]).view(m.weight[i,:,:,:].numel()))
                b=torch.norm(m.weight[i,:,:,:])*np.sqrt(alpha/(1-alpha))
                y=min(max(a/M,b),torch.tensor(1).cuda())
                ys=ys+[y.item()]
                out=out+[(-alpha*(torch.norm(m.weight[i,:,:,:])**2)/y**2+1-alpha).item()]
        return out,ys
    

#-------------------------------------------------------------------------------------------------

#Code to retrive Y variables information from log files
class yDatas:
  
    def __init__(self,logFile):
        self.y=[]
        self.grads=[]
        self.setupName=logFile
        f=open(logFile,"r")
        l=f.readline()
        while not("alpha" in l):
            l=f.readline()
        s=l.split() 
        self.alpha=s[1]
        
        while not("M" in l):
            l=f.readline()
        s=l.split()
        self.M=s[1]
        while not("Y gradients" in l):
            l=f.readline()
        s=l.split()
        temp=[]
        for el in s[slice(2,len(s))]:
            if "[" in el:
                self.grads.append(temp)
                temp=[]
            el=el.replace("[","")
            el=el.replace("]","")
            el=el.replace(",","")
            temp.append(float(el))
        self.grads.append(temp)
        l=f.readline()
        while not("Y" in l):
            l=f.readline()
        s=l.split()
        temp=[]
        for el in s[slice(2,len(s))]:
            if "[" in el:
                self.y.append(temp)
                temp=[]
            el=el.replace("[","")
            el=el.replace("]","")
            el=el.replace(",","")
            temp.append(float(el))
        self.y.append(temp)
        f.close()