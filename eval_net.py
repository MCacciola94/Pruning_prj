import torch
import torch.nn as nn
import argparse
import architectures as archs
import quik_pruning as qp
from trainer import Trainer
import data_loaders as dl
import aux_tools as at
import resnet
import torch.nn.utils.prune as prune

def pruned_par(model):
    
    tot_pruned=0
    for m in model.modules():
        #Convolutional layers 
        if isinstance(m,torch.nn.Conv2d):
            el= float((m.weight_mask[:,:,:,:]==0).sum())
            tot_pruned+=el

    return tot_pruned



def par_count(model):
    res = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            dims = m.weight.shape
            res += dims[0]*dims[1]*dims[2]*dims[3]
        if isinstance(m,nn.Linear):
            dims = m.weight.shape
            res += dims[0]*dims[1]
    return res

def get_unpruned_filters(m):
    idx = []
    if isinstance(m,nn.Conv2d):
        for i in range(m.out_channels):
            if m.weight_mask[i,:].sum()>0:
                idx.append(i)
    
    return idx

def compress_block(block):
    if isinstance(block,resnet.BasicBlock):
        c1 = block._modules["conv1"]
        c2 = block._modules["conv2"]
        d1 = c1.weight.data
        d2 = c2.weight.data
        idx = get_unpruned_filters(c1)
        prune.remove(c1,"weight")
        prune.remove(c2,"weight")
        print(idx)
        if len(idx) < c1.out_channels:
            if len(idx) == 0:
                idx = [0]
                print(c1.in_channels*c1.kernel_size[0]*c1.kernel_size[1]+c2.out_channels*c2.kernel_size[0]*c2.kernel_size[1])
            idx = torch.Tensor(idx).type(torch.int).cuda()

            d1 = torch.index_select(d1, dim = 0, index = idx)
            c1.weight = nn.Parameter(d1)
            c1.out_channels = c1.weight.shape[0]
            
           

            #block._modules["bn1"] = nn.BatchNorm2d(c1.out_channels).cuda()

            d2 = torch.index_select(d2, dim = 1, index = idx)
            c2.weight = nn.Parameter(d2)
            c2.in_channels = c2.weight.shape[1]
            
def prune_block_channels(block):
    if isinstance(block,resnet.BasicBlock):
        c1 = block._modules["conv1"]
        c2 = block._modules["conv2"]
        idx = get_unpruned_filters(c1)
        idx_c = [el for el in range(c2.in_channels)]
        [idx_c.remove(el) for el in idx]
        print(idx_c)
        if len(idx) < c1.out_channels:
            c2.weight_mask[:,idx_c,:,:] = 0

             
            

def go():

    parser = argparse.ArgumentParser(description='evaluation of pruned net')
    parser.add_argument('--name',
                        help="Name of checkpoint")

    args = parser.parse_args()
    model = archs.load_arch("resnet20", 10)
    qp.prune_thr(model,1.e-12)
    base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    model.load_state_dict(base_checkpoint['state_dict'])
    dataset = dl.load_dataset("Cifar10", 128)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5.e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[300], last_epoch= - 1)


    trainer = Trainer(model = model, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, 
                                            criterion =criterion, optimizer = optimizer, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer.validate(reg_on = False)
    _, a = at.sparsityRate(model)
    b = pruned_par(model)
    tot = par_count(model)

    for m in model.modules():
        prune_block_channels(m)

    #print(model)
    #new_tot = par_count(model)
    trainer.validate(reg_on = False)

    _, a_new = at.sparsityRate(model)
    b_new = pruned_par(model)
    print("tot ",tot)
    #print("new_tot ", new_tot)
    print(a)
    print(a_new)
    print(b)
    print(b_new)
    print((b_new-b)/tot)
