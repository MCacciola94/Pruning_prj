import torch
import torch.nn as nn
import argparse
import architectures as archs
import quik_pruning as qp
from trainer import Trainer
import data_loaders as dl
import aux_tools as at
import torch.nn.functional as F
import resnet, resnet_pruned
import torch.nn.utils.prune as prune

def pruned_par(model):
    
    tot_pruned=0
    for m in model.modules():
        #Convolutional layers 
        if isinstance(m,torch.nn.Conv2d):
            if hasattr(m,'weight_mask'):
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
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'
    if isinstance(block,resnet.BasicBlock) or isinstance(block,resnet_pruned.BasicBlock):
        c1 = block._modules["conv1"]
        c2 = block._modules["conv2"]
        b1= block._modules['bn1']
        b2= block._modules['bn2']

        d1 = c1.weight.data
        d2 = c2.weight.data

        db1=b1.weight.data
        db2=b1.bias.data
        db3=b1._buffers

        ddb1=b2.weight.data
        ddb2=b2.bias.data
        ddb3=b2._buffers
  
        idx = get_unpruned_filters(c1)
       
        idx2 = get_unpruned_filters(c2)
        

        if len(idx) < c1.out_channels:
            prune.remove(c1,"weight")
            prune.remove(b1,"bias")
            prune.remove(b1,"weight")

        if len(idx) < c1.out_channels or ( isinstance(block,resnet_pruned.BasicBlock)  and len(idx2) < c2.out_channels):
            prune.remove(c2,"weight")
        
        
        if isinstance(block,resnet_pruned.BasicBlock) and len(idx2) < c2.out_channels:
            prune.remove(b2,"weight")
            prune.remove(b2,"bias")

        # print(idx)
        if len(idx) < c1.out_channels:
            if len(idx) == 0:
                idx = [0]
                # print('All pruned', c1.in_channels*c1.kernel_size[0]*c1.kernel_size[1]+c2.out_channels*c2.kernel_size[0]*c2.kernel_size[1])
            
            idx = torch.Tensor(idx).type(torch.int).to(device)
            d1 = torch.index_select(d1, dim = 0, index = idx)
            c1.weight = nn.Parameter(d1)
            c1.out_channels = c1.weight.shape[0]
            
           
            db1= torch.index_select(db1, dim = 0, index = idx)
            db2= torch.index_select(db2, dim = 0, index = idx)
            db3['running_mean']=torch.index_select(db3['running_mean'], dim = 0, index = idx)
            db3['running_var']=torch.index_select(db3['running_var'], dim = 0, index = idx)
            b1.weight = nn.Parameter(db1)
            b1.bias = nn.Parameter(db2)
            b1._buffers=db3
            b1.num_features=b1.weight.shape[0]
            #works untill here
            d2 = torch.index_select(d2, dim = 1, index = idx)
            if isinstance(block,resnet_pruned.BasicBlock) and len(idx2) < c2.out_channels:
                if len(idx2) == 0:
                    idx2 = [0]
                    # print('All pruned')
                # breakpoint()
                idx2 = torch.Tensor(idx2).type(torch.int).to(device)
                d2 = torch.index_select(d2, dim = 0, index = idx2)
                pruned_idx=[i  for i in range(c2.out_channels) if i not in idx2]
                block.pruned_filters_conv2=pruned_idx
                block.bn2_biases={i:ddb2[i] for i in pruned_idx}

                ddb1= torch.index_select(ddb1, dim = 0, index = idx2)
                ddb2= torch.index_select(ddb2, dim = 0, index = idx2)
                ddb3['running_mean']=torch.index_select(ddb3['running_mean'], dim = 0, index = idx2)
                ddb3['running_var']=torch.index_select(ddb3['running_var'], dim = 0, index = idx2)
                b2.weight = nn.Parameter(ddb1)
                b2.bias = nn.Parameter(ddb2)
                b2._buffers=ddb3
                b2.num_features=b2.weight.shape[0]

            c2.weight = nn.Parameter(d2)
            c2.in_channels = c2.weight.shape[1]
            c2.out_channels = c2.weight.shape[0]
            

            return idx



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

             
            
def block_test():
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'

    model=resnet.resnet20(10).to(device)
    model.eval()
    bb=model.layer1[0]
    inp=torch.rand([2,16,2,2]).to(device)
    out0=bb(inp)
    
    print('0pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    qp.prune_thr(bb,1.e-12)
    outhalf=bb(inp)
    print('1pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    bb.conv1.weight_mask[2]=0
    print('2pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    out1=bb(inp)
    out1_c1=bb.conv1(inp)
    out1_b1=bb.bn1(out1_c1)
    # print(' is zero? ', ou)
    out1_rel=F.relu(out1_b1)
    out1_c2=bb.conv2(out1_rel)
    out1_b2=bb.bn2(out1_c2)
    out1_temp= F.relu(bb.shortcut(inp)+out1_b2)
    print('Diff sanity: ', torch.max(torch.abs(out1_temp-out1)).item())

    idx=compress_block(bb)
    #breakpoint()

    out2=bb(inp)
    out2_c1=bb.conv1(inp)
    out2_b1=bb.bn1(out2_c1)
    out2_rel=F.relu(out2_b1)
    out2_c2=bb.conv2(out2_rel)
    out2_b2=bb.bn2(out2_c2)
    out2_temp= F.relu(bb.shortcut(inp)+out2_b2)
    print('Diff sanity: ', torch.max(torch.abs(out2_temp-out2)).item())
    print('2pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    print('Diff: ', torch.max(torch.abs(out1-out2)).item(),' with start ', torch.max(torch.abs(out1-out0)).item(),' with half ', torch.max(torch.abs(outhalf-out0)).item())
    print('Diff c1: ', torch.max(torch.abs(torch.index_select(out1_c1, dim = 1, index = idx)-out2_c1)).item())
    print('Diff b1: ', torch.max(torch.abs(torch.index_select(out1_b1, dim = 1, index = idx)-out2_b1)).item())
    print('Diff rel: ', torch.max(torch.abs(torch.index_select(out1_rel, dim = 1, index = idx)-out2_rel)).item())
    print('Diff c2: ', torch.max(torch.abs(out1_c2-out2_c2)).item())#torch.index_select(out1_c2, dim = 1, index = idx)
    print('Diff b2: ', torch.max(torch.abs(out1_b2-out2_b2)).item())#torch.index_select(out1_b2, dim = 1, index = idx)


def block_test_pruned():
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'
    
    model=resnet_pruned.resnet20(10).to(device)
    model.eval()
    bb=model.layer1[0]
    inp=torch.rand([2,16,2,2]).to(device)
    out0=bb(inp)
    
    print('0pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    qp.prune_thr(bb,1.e-12)
    outhalf=bb(inp)
    print('1pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    bb.conv1.weight_mask[2]=0
    bb.conv2.weight_mask[:]=0
    print('2pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    out1=bb(inp)
    # out1_c1=bb.conv1(inp)
    # out1_b1=bb.bn1(out1_c1)
    # # print(' is zero? ', ou)
    # out1_rel=F.relu(out1_b1)
    # out1_c2=bb.conv2(out1_rel)
    # out1_b2=bb.bn2(out1_c2)
    # out1_temp= F.relu(bb.shortcut(inp)+out1_b2)
    # print('Diff sanity: ', torch.max(torch.abs(out1_temp-out1)).item())

    idx=compress_block(bb)
    #breakpoint()

    out2=bb(inp)
    # out2_c1=bb.conv1(inp)
    # out2_b1=bb.bn1(out2_c1)
    # out2_rel=F.relu(out2_b1)
    # out2_c2=bb.conv2(out2_rel)
    # out2_b2=bb.bn2(out2_c2)
    # out2_temp= F.relu(bb.shortcut(inp)+out2_b2)
    # print('Diff sanity: ', torch.max(torch.abs(out2_temp-out2)).item())
    print('2pars: ',par_count(bb),' pruned pars: ',pruned_par(bb))
    print('Diff: ', torch.max(torch.abs(out1-out2)).item(),' with start ', torch.max(torch.abs(out1-out0)).item(),' with half ', torch.max(torch.abs(outhalf-out0)).item())
    # print('Diff c1: ', torch.max(torch.abs(torch.index_select(out1_c1, dim = 1, index = idx)-out2_c1)).item())
    # print('Diff b1: ', torch.max(torch.abs(torch.index_select(out1_b1, dim = 1, index = idx)-out2_b1)).item())
    # print('Diff rel: ', torch.max(torch.abs(torch.index_select(out1_rel, dim = 1, index = idx)-out2_rel)).item())
    # print('Diff c2: ', torch.max(torch.abs(out1_c2-out2_c2)).item())#torch.index_select(out1_c2, dim = 1, index = idx)
    # print('Diff b2: ', torch.max(torch.abs(out1_b2-out2_b2)).item())#torch.index_select(out1_b2, dim = 1, index = idx)




def go():

    parser = argparse.ArgumentParser(description='evaluation of pruned net')
    parser.add_argument('--name',
                        help="Name of checkpoint")

    args = parser.parse_args()
    model = archs.load_arch("resnet20", 10).cuda()
    model_pruned=torch.nn.DataParallel(resnet_pruned.__dict__['resnet20'](10))
    qp.prune_thr(model,1.e-12)
    qp.prune_thr(model_pruned,1.e-12)
    # base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    base_checkpoint=torch.load("/local1/caccmatt/Pruning_prj/saves/save_V0.0.2-_resnet20_Cifar10_lr0.1_l1.8_a0.001_e300+200_bs128_t0.0001_m0.9_wd0.0005_mlstemp3_Mscl1.0/checkpoint.th")

    model.load_state_dict(base_checkpoint['state_dict'])
    model_pruned.load_state_dict(base_checkpoint['state_dict'])
    dataset = dl.load_dataset("Cifar10", 128)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5.e-4)
    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5.e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[300], last_epoch= - 1)


    trainer = Trainer(model = model, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, 
                                            criterion =criterion, optimizer = optimizer, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)
    trainer_pruned = Trainer(model = model_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, 
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)
    _, sp_rate0 = at.sparsityRate(model)
    pr_par0 = pruned_par(model)
    tot0 = par_count(model)

    _, sp_rate0_pr = at.sparsityRate(model_pruned)
    pr_par0_pr = pruned_par(model_pruned)
    tot0_pr = par_count(model_pruned)

    for m in model.modules():
        compress_block(m)
        # prune_block_channels(m)
    for m in model_pruned.modules():
        compress_block(m)

    breakpoint()

    #print(model)
    #new_tot = par_count(model)
    trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)

    _, sp_rate1 = at.sparsityRate(model)
    pr_par1 = pruned_par(model)
    tot1 = par_count(model)

    _, sp_rate1_pr = at.sparsityRate(model_pruned)
    pr_par1_pr = pruned_par(model_pruned)
    tot1_pr = par_count(model_pruned)
    print("tot berfore and after ",tot0, ' ', tot1, ' pr ', tot0_pr,' ', tot1_pr)
    #print("new_tot ", new_tot)
    print("sparsity before and after ", sp_rate0, ' ', sp_rate1, ' pr ', sp_rate0_pr,' ', sp_rate1_pr )
    print("pruned par before and after ", pr_par0, ' ', pr_par1, ' pr ', pr_par0_pr,' ', pr_par1_pr )

    # print((b_new-b)/tot)
    return model, model_pruned, dataset