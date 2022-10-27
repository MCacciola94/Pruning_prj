import torch
import numpy as np

class myTools: 
   def __init__(self,alpha,M):
       self.alpha=alpha
       self.M=M
    #Computation of the current factor
   def myReg(self,net, loss, lamb = 0.1):
        reg = 0 
        alpha=self.alpha
        const=(torch.sqrt(torch.Tensor([alpha/(1-alpha)]))).cuda()
        tot=0
        for m in net.modules():

            #Fully Connected layers
            if isinstance(m,torch.nn.Linear):
                continue
                M=self.M[m]
                norminf=torch.norm(m.weight,dim=1,p=np.inf)
                norm2= torch.norm(m.weight,dim=1,p=2)
                num_el=m.in_features
                tot+=m.in_features*m.out_features
                
            else:              
            #Convolutional layers               
                if isinstance(m,torch.nn.Conv2d):
                    M=self.M[m]
                    norminf=torch.norm(m.weight,dim=(1,2,3),p=np.inf)
                    norm2= torch.norm(m.weight,dim=(1,2,3),p=2)
                    num_el=m.kernel_size[0]*m.kernel_size[1]*m.in_channels
                    tot+=m.kernel_size[0]*m.kernel_size[1]*m.in_channels*m.out_channels
                else:
                    continue


            bo1 = torch.max(norminf/M,const*norm2)>=1
            reg1 = norm2**2+1-alpha

            bo2 = norminf/M<=const*norm2
            reg2=const*norm2*(1+alpha)

            eps=(torch.zeros(norminf.size())).cuda()
            eps=eps+1e-10
            reg3=norm2**2/(torch.max(eps,norminf))*M+(1-alpha)*norminf/M

            bo2=torch.logical_and(bo2, torch.logical_not(bo1))
            bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

            reg+=(bo1*reg1+bo2*reg2+bo3*reg3).sum()*num_el
                        
        loss = loss +lamb* reg/tot
        reg=(lamb* reg/tot).item()
        return loss, reg
    
    
    
  
    
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