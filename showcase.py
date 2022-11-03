import torch
import numpy as np

def l1l2norm(t,dim=1):
    # breakpoint()
    out=torch.norm(t,dim=dim)
    return out.sum()

def l2normsq(t):
    # breakpoint()

    out = torch.norm(t)**2
    return out

def l0norm(t,dim=1, thr=1e-6):
    # breakpoint()
    out=torch.norm(t,dim=dim, p=np.inf)
    return (out>thr).sum()

def perspreg(w,alpha,lamb,M, dim=1):
    t=w.view(w.size(0),-1)
    ninf= torch.norm(t,dim=dim,p=np.inf)
    norm2=torch.norm(t,dim=dim)
    const = (alpha/(1-alpha))**0.5

    bo1 = torch.max(ninf/M,const*norm2)>=1
    reg1 = alpha*norm2**2+1-alpha

    bo2 = ninf/M<=const*norm2
    reg2=2*(alpha*(1-alpha))**0.5*norm2

    eps=(torch.zeros(ninf.size()))
    eps=eps+1e-10
    reg3=alpha* norm2**2/(torch.max(eps,ninf))*M+(1-alpha)*ninf/M

    bo2=torch.logical_and(bo2, torch.logical_not(bo1))
    bo3=torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

    reg=(bo1*reg1+bo2*reg2+bo3*reg3).sum()
                    
    return lamb*reg

def sqmeanerr(t,target):
    out = torch.norm(t-target)**2/t.numel()
    return out

def real_obj_val(w,tar,alpha,lamb):
    return sqmeanerr(w,tar) + lamb* (alpha*l2normsq(w) + (1-alpha)*l0norm(w))

def l1l2_obj_fun(w,tar,alpha,lamb):
    return sqmeanerr(w,tar) + lamb* (alpha * l2normsq(w) + (1-alpha) * l1l2norm(w))

def persp_obj_fun(w,tar,alpha,lamb, M):
    return sqmeanerr(w,tar)+perspreg(w,alpha,lamb,M)

def standard_obj_fun(w,tar,alpha,lamb):    
    return sqmeanerr(w,tar) + lamb* (alpha * l2normsq(w))

def y_persp(w,alpha,M):
    t=w.view(w.size(0),-1)
    ninf= torch.norm(t,dim=1,p=np.inf).unsqueeze(0)
    norm2=torch.norm(t,dim=1).unsqueeze(0)
    const = (alpha/(1-alpha))**0.5

    tt =torch.cat((ninf/M,norm2*const),0)
    out,_=torch.max(tt,0)
    bo = out>1
    out[bo]=1

    return out.data

def y_standard(w,M):
    return (torch.norm(w,p=np.inf,dim=1)/M).data



tar=torch.Tensor([[0.5,0.8,0.1],[0.2,0.4,0.6]])
w_start = torch.Tensor([[-0.3,0.2,-0.7],[0.5,-0.7,0.8]])

lr=0.1
iterations=2000
M=2
print_freq=100000
mlst = [300,800,1200,1500]

for lamb in [0.3]:#,0.05,0.1,0.3,0.5,0.7,0.8,1.0]:
    for alpha in [0.5]:#,0.1,0.05,0.01,0.005,0.001,1e-4]:
        print('alpha= ' ,alpha, 'lambda= ', lamb )
        w=w_start
        w=torch.nn.Parameter(w)
        opt=torch.optim.SGD({w},lr=lr,weight_decay=0.0,momentum=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=mlst, last_epoch= - 1)
                        
        # print('-------------- l1l2 --------')
        for i in range(iterations):   
            
            loss = l1l2_obj_fun(w,tar,alpha,lamb)

            opt.zero_grad()
            loss.backward()

            if (i+1)%print_freq==0: 
                print('iter ',i,' loss ', loss.item())
                print('grad norm ', torch.norm(w.grad).item())

            opt.step()
            lr_scheduler.step()

        # breakpoint()
        # print('grad norm ', torch.norm(w.grad).item())
        print('L1L2 -> ', l1l2_obj_fun(w,tar,alpha,lamb).item(), 'real obj val ', real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))
        # print(' opt sol ', w)


        #################################################################################################################################
        w=w_start
        w=torch.nn.Parameter(w)
        opt=torch.optim.SGD({w},lr=lr,weight_decay=0.0,momentum=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=mlst, last_epoch= - 1)

        # print('-------------- persp --------')
        for i in range(iterations):   
            
            loss = persp_obj_fun(w,tar,alpha,lamb,M)

            opt.zero_grad()
            loss.backward()

            if (i+1)%print_freq==0: 
                print('iter ',i,' loss ', loss.item())
                print('grad norm ', torch.norm(w.grad).item())

            opt.step()
            lr_scheduler.step()

        # breakpoint()
        # print('grad norm ', torch.norm(w.grad).item())
        print('PERSP -> ', persp_obj_fun(w,tar,alpha,lamb,M).item(), ' real obj val ',real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))
        # print(' opt sol ', w)

#################################################################################################################################

        w=w_start
        w=torch.nn.Parameter(w)
        opt=torch.optim.SGD({w},lr=lr,weight_decay=0.0,momentum=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=mlst, last_epoch= - 1)

        # print('-------------- Real --------')
        for i in range(iterations):   
            
            loss =  standard_obj_fun(w,tar,alpha,lamb)

            opt.zero_grad()
            loss.backward()

            if (i+1)%print_freq==0: 
                print('iter ',i,' loss ', loss.item())
                print('grad norm ', torch.norm(w.grad).item())

            opt.step()
            lr_scheduler.step()

        # breakpoint()
        # print('grad norm ', torch.norm(w.grad).item())
        print('No pruned opt -> ', standard_obj_fun(w,tar,alpha,lamb).item(), ' real obj val ',real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))
        # print(' opt sol ', w)

#################################################################################################################################

        w=w_start
        w[0]=0
        w=torch.nn.Parameter(w)
        opt=torch.optim.SGD({w},lr=lr,weight_decay=0.0,momentum=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=mlst, last_epoch= - 1)

        # print('-------------- Real --------')
        for i in range(iterations):   
            
            loss = standard_obj_fun(w,tar,alpha,lamb)

            opt.zero_grad()
            loss.backward()

            if (i+1)%print_freq==0: 
                print('iter ',i,' loss ', loss.item())
                print('grad norm ', torch.norm(w.grad).item())

            w.grad[0]=0
            opt.step()
            lr_scheduler.step()

        # breakpoint()
        # print('grad norm ', torch.norm(w.grad).item())
        print('First pruned opt -> ', standard_obj_fun(w,tar,alpha,lamb).item(), ' real obj val ',real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))
        # print(' opt sol ', w)


#################################################################################################################################

        w=w_start
        w[1]=0
        w=torch.nn.Parameter(w)
        opt=torch.optim.SGD({w},lr=lr,weight_decay=0.0,momentum=0.0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=mlst, last_epoch= - 1)

        # print('-------------- Real --------')
        for i in range(iterations):   
            
            loss = standard_obj_fun(w,tar,alpha,lamb)

            opt.zero_grad()
            loss.backward()

            if (i+1)%print_freq==0: 
                print('iter ',i,' loss ', loss.item())
                print('grad norm ', torch.norm(w.grad).item())
            
            w.grad[1]=0
            opt.step()
            lr_scheduler.step()

        # breakpoint()
        # print('grad norm ', torch.norm(w.grad).item())
        print('Second pruned opt -> ', standard_obj_fun(w,tar,alpha,lamb).item(), ' real obj val ',real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))
        # print(' opt sol ', w)
        w=torch.zeros([2,3])
        print('ALL pruned opt -> ', standard_obj_fun(w,tar,alpha,lamb).item(), ' real obj val ',real_obj_val(w,tar,alpha,lamb).item(), ' l0 struct ', l0norm(w).item(),torch.norm(w,dim=1, p=np.inf).data)
        print(' y standard ',y_standard(w,M),' y persp ', y_persp(w,alpha,M))