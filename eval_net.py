import torch
import torch.nn as nn
import argparse
import os
import architectures as archs
import quik_pruning as qp
from trainer import Trainer
import data_loaders as dl
import aux_tools as at
import torch.nn.functional as F
import resnet, resnet_pruned, resnetBig_pruned
import torch.nn.utils.prune as prune
import copy
import vgg_pruned, resnetBig_imgNet_pruned

DEBUG =False
eps_debug =1e-5
out_old={}
#compute number of zero in weight masks of convolutional layers
def pruned_par(model):
    
    tot_pruned=0
    for m in model.modules():
        #Convolutional layers 
        if isinstance(m,torch.nn.Conv2d):
            if hasattr(m,'weight_mask'):
                el= float((m.weight_mask[:,:,:,:]==0).sum())
                tot_pruned+=el  

    return tot_pruned

#compute number of parmters of model, default takes into account all params from conv, linear and batchnorm
def par_count(model,conv=True,bias_conv=True, linear=True, bias_linear=True, batchnorm=True, bias_batchnorm=True, all_modules=True):
    res = 0
    for m in model.modules():
        
        if isinstance(m,nn.Conv2d) and (conv or all_modules):
            res+=par_count_module(m, bias=(bias_conv or all_modules))

        if isinstance(m,nn.Linear) and (linear or all_modules):
            res+=par_count_module(m, bias=(bias_linear or all_modules))

        if isinstance(m,nn.BatchNorm2d) and (batchnorm or all_modules):
            res+=par_count_module(m, bias=(bias_batchnorm or all_modules))

    return res

#count the number of a params of a module
def par_count_module(module, bias):
    res=0
    for name, par in module.named_parameters():
        if par.requires_grad and (bias or name!='bias'):
            res+= par.numel()
    return res

#count the overall total number of a params of a model
def par_count_all(model):
    res=0
    for par in model.parameters():
        if par.requires_grad:
            res+= par.numel()
    return res


#compare two model printing different params
def find_diff_params(model1,model2, equal_vals= False):
    params1=model1.state_dict()
    params2=model2.state_dict()
    
    for par in params1.keys():
        if hasattr(par,'requires_grad') and par.requires_grad and par not in params2.keys():
            print('not in mod2 ', par)
        elif equal_vals and torch.is_tensor(params1[par]):
            if torch.norm(params1[par].to(torch.float)-params2[par].to(torch.float))!=0:
                print('different vals', par)


    for par in params2.keys():
        if hasattr(par,'requires_grad') and par.requires_grad and par not in params1.keys():
            print('not in mod1 ', par)
    

# return the indexes of filters that are not completely pruned maskwise
def get_unpruned_filters(m):
    idx = []
    if isinstance(m,nn.Conv2d):
        for i in range(m.out_channels):
            if m.weight_mask[i,:].sum()>0:
                idx.append(i)
    
    return idx

#remove parameters created by pytorch pruning function in conv layers
def remove_additional_pars_conv(conv):

    weight_bkp = conv.weight.data
    prune.remove(conv,'weight')
    conv.weight=nn.Parameter(weight_bkp)
    
    if hasattr(conv,'bias') and conv.bias is not None:
        bias_bkp = conv.bias.data
        prune.remove(conv,'bias')
        conv.bias=nn.Parameter(bias_bkp)

#remove parameters created by pytorch pruning function in batchnorm layers    
def remove_additional_pars_bn(bn):
    weight_bkp = bn.weight.data
    bias_bkp = bn.bias.data
    # buffers_bkp = {'running_mean': bn._buffers['running_mean'],'running_var': bn._buffers['running_var']}

    prune.remove(bn,'weight')
    prune.remove(bn,'bias')

    bn.weight=nn.Parameter(weight_bkp)
    bn.bias=nn.Parameter(bias_bkp)
    # bn._buffers=buffers_bkp
    
# compress a conv layer selecting the desired output filters
def compress_conv_filters(conv, idx_not_pr, debug =DEBUG):
    if DEBUG:
        if conv not in out_old.keys():
            dims= conv.weight.shape
            ones =torch.ones(dims[1:])
            out_old[conv]=conv(ones.cuda())

    weight = conv.weight.data
    weight = torch.index_select(weight, dim = 0, index = idx_not_pr)
    conv.weight = nn.Parameter(weight)

    if conv.bias is not None:
        bias = conv.bias.data
        bias = torch.index_select(bias, dim = 0, index = idx_not_pr)
        conv.bias = nn.Parameter(bias)

    conv.out_channels = conv.weight.shape[0]

# compress a conv layer selecting the desired input channels
def compress_conv_channels(conv, idx_not_pr):
    if DEBUG:
        if conv not in out_old.keys():
            dims= conv.weight.shape
            ones =torch.ones(dims[1:])
            out_old[conv]=conv(ones.cuda())

    weight = conv.weight.data
    weight = torch.index_select(weight, dim = 1, index = idx_not_pr)
    conv.weight = nn.Parameter(weight)
    conv.in_channels = conv.weight.shape[1]

# compress a linear layer selecting the desired input features
def compress_linear_in(lin, idx_not_pr):
    weight = lin.weight.data
    weight = torch.index_select(weight, dim = 1, index = idx_not_pr)
    lin.weight = nn.Parameter(weight)
    lin.in_features = lin.weight.shape[1]


# compress a batchnorm layer selecting the desired features
def compress_batchnorm(bn,idx_not_pr):
    weight = bn.weight.data
    bias = bn.bias.data
    buffers = bn._buffers

    weight= torch.index_select(weight, dim = 0, index = idx_not_pr)
    bias= torch.index_select(bias, dim = 0, index = idx_not_pr)
    buffers['running_mean']=torch.index_select(buffers['running_mean'], dim = 0, index = idx_not_pr)
    buffers['running_var']=torch.index_select(buffers['running_var'], dim = 0, index = idx_not_pr)

    bn.weight = nn.Parameter(weight)
    bn.bias = nn.Parameter(bias)
    bn._buffers=buffers
    bn.num_features=bn.weight.shape[0]

def compress_shortcut_inp(block,idx_sc_pr):
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'

    if hasattr(block, 'shortcut'):
        sc=block.shortcut
        sc_name = 'pruned_shortcut'
    else: 
        sc=block.downsample
        sc_name = 'pruned_downsample'

    if isinstance(sc, nn.Sequential):
        if  len(sc)>0:
            idx_not_pr=[i for i in range(sc[0].weight.size(1)) if i not in idx_sc_pr]
            idx_not_pr = torch.Tensor(idx_not_pr).type(torch.int).to(device)
            compress_conv_channels(sc[0],idx_not_pr)
        else: block.__dict__[sc_name] += idx_sc_pr
    elif 'LambdaLayer' in str(type(sc)):
        block.__dict__[sc_name]+= [i+block.numb_added_planes//2 for i in idx_sc_pr]


def compress_conv_bn_linear(conv1,bn,lin):
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'

    original_out_channles=conv1.weight.size(0)

#Compress conv1 and batchnorm
    idx_not_pr = get_unpruned_filters(conv1)
    
    if len(idx_not_pr) < conv1.out_channels:
        if len(idx_not_pr) == 0:
                    idx_not_pr = [0]
        idx_not_pr = torch.Tensor(idx_not_pr).type(torch.int).to(device)
        compress_conv_filters(conv1,idx_not_pr)
        compress_batchnorm(bn,idx_not_pr)

        channel_size = lin.in_features/original_out_channles
        # print(channel_size)
        idx_aux=[]
        for i in  idx_not_pr:
            idx_aux += [i*channel_size + j for j in range(int(channel_size))]

        idx_aux = torch.Tensor(idx_aux).type(torch.int).to(device)

        compress_linear_in(lin,idx_aux)      
    
    return idx_not_pr




# Copress an isolated sequence conv1->batchnorm->conv2, where the compression is consequence of the maskwise pruned filters of conv1
def compress_conv_bn_conv(conv1,bn1,conv2):
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'
    
    idx_not_pr = get_unpruned_filters(conv1)
    
    if len(idx_not_pr) < conv1.out_channels:
        if len(idx_not_pr) == 0:
            idx_not_pr = [0]
            
        idx_not_pr = torch.Tensor(idx_not_pr).type(torch.int).to(device)
        compress_conv_filters(conv1,idx_not_pr)
        compress_batchnorm(bn1,idx_not_pr)
        compress_conv_channels(conv2,idx_not_pr)

    return idx_not_pr

# Copress an isolated sequence [(...->conv1->batchnorm)+(shortcut1)]->[(conv->...)+(shortcut2)], 
# where the compression is consequence of the maskwise pruned filters of conv1 and shortcut1
def compress_conv_bn_shortcut(conv1,bn,shortcut,pruned_shortcut,next_block):
    device='cpu'
    if torch.cuda.is_available():
        device= 'cuda:0'

    original_out_channles=conv1.weight.size(0)

#Compress conv1 and batchnorm
    idx_not_pr = get_unpruned_filters(conv1)
    
    if len(idx_not_pr) < conv1.out_channels:
        if len(idx_not_pr) == 0:
                    idx_not_pr = [0]
        idx_not_pr = torch.Tensor(idx_not_pr).type(torch.int).to(device)
        compress_conv_filters(conv1,idx_not_pr)
        compress_batchnorm(bn,idx_not_pr)
      
    # Deal with shortcut1. There are 3 cases: Sequential but empty, Sequential with conv-bn and LambdaLayer
    if isinstance(shortcut,nn.Sequential) and len(shortcut)>0:
        # breakpoint()
        
        conv_sc =shortcut[0]
        bn_sc =shortcut[1]

        idx_sc_not_pr = get_unpruned_filters(conv_sc)

        if len(idx_sc_not_pr) <original_out_channles:
            if len(idx_sc_not_pr) == 0:
                idx_sc_not_pr = [0]
            idx_sc_not_pr = torch.Tensor(idx_sc_not_pr).type(torch.int).to(device)
            compress_conv_filters(conv_sc,idx_sc_not_pr)
            compress_batchnorm(bn_sc,idx_sc_not_pr)

    else:
        idx_sc_not_pr = [i for i in range(original_out_channles) if i not in pruned_shortcut]      

    # Adjust input of the following block. Again there are 3 cases for the shortcut (of next block this time)  

    if hasattr(next_block,'conv1'):
        idx_channels_next_layer_not_pr= [i for i in range(next_block.conv1.in_channels) if (i in idx_not_pr or i in idx_sc_not_pr)]
        idx_channels_next_layer_not_pr = torch.Tensor(idx_channels_next_layer_not_pr).type(torch.int).to(device)

        compress_conv_channels(next_block.conv1,idx_channels_next_layer_not_pr) 
        compress_shortcut_inp(next_block,[i for i in range(original_out_channles) if i not in idx_channels_next_layer_not_pr])   
  
    
    if isinstance(next_block,nn.Linear):
        idx_channels_next_layer_not_pr = [i for i in range(original_out_channles) if i in idx_not_pr or i in idx_sc_not_pr]
        channel_size = next_block.in_features/original_out_channles
        # print(channel_size)
        idx_aux=[]
        for i in  idx_channels_next_layer_not_pr:
            idx_aux += [i*channel_size + j for j in range(int(channel_size))]
        
        idx_channels_next_layer_not_pr = idx_aux
    
        idx_channels_next_layer_not_pr = torch.Tensor(idx_channels_next_layer_not_pr).type(torch.int).to(device)
        
        compress_linear_in(next_block,idx_channels_next_layer_not_pr)      
    
    return [i for i in range(original_out_channles) if i not in idx_not_pr], [i for i in range(original_out_channles) if i not in idx_sc_not_pr]

#Compress the classic resnet basicblok (2 conv, 2 batchnorm and a shortcut)
def compress_basicblock(block,next_block):

    compress_conv_bn_conv(block.conv1,block.bn1,block.conv2)

    if hasattr(block, 'shortcut'):
        sc=block.shortcut
        sc_name = 'pruned_shortcut'
    else: 
        sc=block.downsample
        sc_name = 'pruned_downsample'
    
    idx_conv2_pruned, idx_sc_pruned= compress_conv_bn_shortcut(block.conv2, block.bn2, sc, block.__dict__[sc_name], next_block)

    block.pruned_filters_conv2=idx_conv2_pruned
    block.__dict__[sc_name] = idx_sc_pruned

#Compress the classic resnet bottleneck block (3 conv, 3 batchnorm and a shortcut)
def compress_bottleneck(block,next_block):
    if hasattr(block, 'shortcut'):
        sc=block.shortcut
        sc_name = 'pruned_shortcut'
    else: 
        sc=block.downsample
        sc_name = 'pruned_downsample'

    compress_conv_bn_conv(block.conv1,block.bn1,block.conv2)

    compress_conv_bn_conv(block.conv2,block.bn2,block.conv3)
    
    idx_conv3, idx_sc= compress_conv_bn_shortcut(block.conv3,block.bn3,sc, block.__dict__[sc_name],next_block)

    block.pruned_filters_conv3 = idx_conv3
    block.__dict__[sc_name] = idx_sc


#Compress a resnet model that already have a pruning mask
def compress_resnet(net):
    if isinstance(net,nn.DataParallel):
        net=net.module

    # breakpoint()

    #First part of the net that is outside the main layers
    original_num_filters= net.conv1.weight.size(0)
    idx_sc_not_pr=compress_conv_bn_conv(net.conv1,net.bn1,net.layer1[0].conv1)
    compress_shortcut_inp(net.layer1[0],[i for i in range(original_num_filters) if i not in idx_sc_not_pr])

# Compress block by block, info about the next block is necessary
    for block,next_block in net.blocks_sequence():

        if 'BasicBlock' in str(type(block)):
            compress_basicblock(block, next_block)
        elif 'Bottleneck' in str(type(block)):
            compress_bottleneck(block, next_block)

# Remove masks and other params
    for module in net.modules():
        if isinstance(module,nn.Conv2d):
            remove_additional_pars_conv(module)
        elif isinstance(module,nn.BatchNorm2d):
            remove_additional_pars_bn(module)
    # breakpoint()
    if DEBUG:
        for conv in out_old.keys():
            dims= conv.weight.shape
            ones =torch.ones(dims[1:])
            out_new=conv(ones.cuda())
            out_old_aux=out_old[conv]
            breakpoint()
            idx= torch.norm(out_old_aux,dim=[1,2],p=2)!=0
            out_old_aux =out_old_aux[idx]
            if torch.norm(out_old_aux-out_new,p=2)>eps_debug:
                print('Not equivalent ',conv)

#Compress a vgg model that already have a pruning mask
def compress_vgg(net):
    if isinstance(net,nn.DataParallel):
        net=net.module

# breakpoint()
    j=0
    k=0
    while k<len(net.features):
        # breakpoint()

        k=j+2
        while  k<len(net.features) and (not isinstance(net.features[k], nn.Conv2d)):
            k+=1

        if k<len(net.features) and isinstance(net.features[k], nn.Conv2d):
            compress_conv_bn_conv(net.features[j],net.features[j+1],net.features[k])
        else: 
            compress_conv_bn_linear(net.features[j],net.features[j+1],net.classifier)

        j=k
            

# Remove masks and other params
    for module in net.modules():
        if isinstance(module,nn.Conv2d):
            remove_additional_pars_conv(module)
        elif isinstance(module,nn.BatchNorm2d):
            remove_additional_pars_bn(module)
    # breakpoint()
    if DEBUG:
        for conv in out_old.keys():
            dims= conv.weight.shape
            ones =torch.ones(dims[1:])
            out_new=conv(ones.cuda())
            out_old_aux=out_old[conv]
            # breakpoint()
            idx= torch.norm(out_old_aux,dim=[1,2],p=2)!=0
            out_old_aux =out_old_aux[idx]
            if torch.norm(out_old_aux-out_new,p=2)>eps_debug:
                print('Not equivalent ',conv)


#Prune resnet20 starting froma checkpoint
def unit_test_2010(name):
    
    name='saves/save_V1.1.0-_resnet20_Cifar10_lr0.10001_l10.0_a0.1_e3+1_bs128_t0.05_tstr0.05_m0.9_wd0.0005_mlstemp3_Mscl1.0_structconvs_and_batchnorm/model_best_val.th'
    model_pruned=torch.nn.DataParallel(resnet_pruned.__dict__['resnet20'](10))
    qp.prune_thr(model_pruned,1.e-30)

    # base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    base_checkpoint=torch.load(name)
    model_pruned.load_state_dict(base_checkpoint['state_dict'])

    model_pruned.eval()
    model_pruned(torch.rand([1,3,32,32]))
    dataset = dl.load_dataset("Cifar10", 128)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()



    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5.e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pruned,
                                                        milestones=[300], last_epoch= - 1)


    trainer_pruned = Trainer(model = model_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer_pruned.validate(reg_on = False)


 

    _, sp_rate0_pr = at.sparsityRate(model_pruned)
    pr_par0_pr = pruned_par(model_pruned)
    tot0_pr = par_count(model_pruned)

    
    compress_resnet(model_pruned)

    # breakpoint()

    #print(model)
    # new_tot = par_count(model)
    # trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)



    _, sp_rate1_pr = at.sparsityRate(model_pruned)
    pr_par1_pr = pruned_par(model_pruned)
    tot1_pr = par_count(model_pruned)
    # print("tot berfore and after ",tot0, ' ', tot1, ' pr ', tot0_pr,' ', tot1_pr)
    #print("new_tot ", new_tot)
    # print("sparsity before and after ", sp_rate0, ' ', sp_rate1, ' pr ', sp_rate0_pr,' ', sp_rate1_pr )
    # print("pruned par before and after ", pr_par0, ' ', pr_par1, ' pr ', pr_par0_pr,' ', pr_par1_pr )

    print('New stats ', tot0_pr-tot1_pr,' perc ', 100*(tot0_pr-tot1_pr)/tot0_pr )
    fully_pruned= resnet_pruned.FullyCompressedResNet(copy.deepcopy(model_pruned))
    # breakpoint()


    trainer_fully_pruned = Trainer(model = fully_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)
    print('Fully pruned model acc ')
    trainer_fully_pruned.validate(reg_on = False)


 

    _, sp_rate_fll = at.sparsityRate(fully_pruned)
    pr_par_fll = pruned_par(fully_pruned)
    tot_fll = par_count(fully_pruned)

    print('Final stats ', tot0_pr-tot_fll,' perc ', 100*(tot0_pr-tot_fll)/tot0_pr )


    return model_pruned, dataset

def eval_again():
    name_list = os.listdir('saves')
    for file_name in name_list:
        if '_resnet20_Cifar10_'in file_name and 'original' not in file_name:
            print(file_name)
            go('saves/'+file_name+'/checkpoint.th')
            breakpoint()

#Prune resnet20 starting froma checkpoint
def unit_test_5010(name):
    
    # name='saves/save_V1.1.CP-_resnet50_Cifar10_lr0.1_l1.0_a0.1_e300+200_bs64_t0.05_tstr0.05_m0.9_wd0.0005_mlstemp3_Mscl1.0_structconvs_and_batchnorm_id1667581732/checkpoint.th'
    # model_pruned=torch.nn.DataParallel(resnetBig_pruned.__dict__['resnet50'](10))
    model_pruned= resnetBig_pruned.resnet50(10)

    qp.prune_thr(model_pruned,1.e-30)

    # base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    base_checkpoint=torch.load(name)
    if 'linear.weight' not in base_checkpoint['state_dict'].keys():
        dict_temp=base_checkpoint['state_dict']
        dict_temp['linear.weight']=dict_temp['linear.weight_orig']
        dict_temp['linear.bias']=dict_temp['linear.bias_orig']
        dict_temp.pop('linear.weight_orig')
        dict_temp.pop('linear.bias_orig')
        dict_temp.pop('linear.weight_mask')
        dict_temp.pop('linear.bias_mask')
        base_checkpoint['state_dict']=dict_temp

    model_pruned.load_state_dict(base_checkpoint['state_dict'])

    model_pruned.eval()
    model_pruned(torch.rand([1,3,32,32]))

    dataset = dl.load_dataset("Cifar10", 128)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model_pruned.cuda()



    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pruned,
                                                        milestones=[300], last_epoch= - 1)


    trainer_pruned = Trainer(model = model_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer_pruned.validate(reg_on = False)

    # breakpoint()
    model_aux= copy.deepcopy(model_pruned)
 

    _, sp_rate0_pr = at.sparsityRate(model_pruned)
    pr_par0_pr = pruned_par(model_pruned)
    tot0_pr = par_count(model_pruned)

    compress_resnet(model_pruned)

    breakpoint()

    #print(model)
    # new_tot = par_count(model)
    # trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)



    _, sp_rate1_pr = at.sparsityRate(model_pruned)
    pr_par1_pr = pruned_par(model_pruned)
    tot1_pr = par_count(model_pruned)
    # print("tot berfore and after ",tot0, ' ', tot1, ' pr ', tot0_pr,' ', tot1_pr)
    #print("new_tot ", new_tot)
    # print("sparsity before and after ", sp_rate0, ' ', sp_rate1, ' pr ', sp_rate0_pr,' ', sp_rate1_pr )
    # print("pruned par before and after ", pr_par0, ' ', pr_par1, ' pr ', pr_par0_pr,' ', pr_par1_pr )

    print('New stats ', tot0_pr-tot1_pr,' perc ', 100*(tot0_pr-tot1_pr)/tot0_pr )
    # fully_pruned= resnet_pruned.FullyCompressedResNet(copy.deepcopy(model_pruned))
    # breakpoint()


    # trainer_fully_pruned = Trainer(model = fully_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
    #                                         criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)
    # print('Fully pruned model acc ')
    # trainer_fully_pruned.validate(reg_on = False)


 

    # _, sp_rate_fll = at.sparsityRate(fully_pruned)
    # pr_par_fll = pruned_par(fully_pruned)
    # tot_fll = par_count(fully_pruned)

    # print('Final stats ', tot0_pr-tot_fll,' perc ', 100*(tot0_pr-tot_fll)/tot0_pr )


    return model_pruned, dataset



#Prune resnet20 starting froma checkpoint
def unit_test_1610(name):
    
    name='saves/save_V1.1.CP-_vgg16_Cifar10_lr0.1_l23.0_a0.5_e1+1_bs128_t0.05_tstr0.05_m0.9_wd0.0005_mlstemp1_Mscl1.0_structconvs_and_batchnorm_id1667857294/model_best_val.th'
    # model_pruned=torch.nn.DataParallel(resnetBig_pruned.__dict__['resnet50'](10))
    model_pruned= vgg_pruned.vgg16()

    qp.prune_thr(model_pruned,1.e-4)

    # base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    base_checkpoint=torch.load(name)

    model_pruned.load_state_dict(base_checkpoint['state_dict'])

    model_pruned.eval()
    model_pruned(torch.rand([1,3,32,32]))

    dataset = dl.load_dataset("Cifar10", 128)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model_pruned.cuda()



    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pruned,
                                                        milestones=[300], last_epoch= - 1)


    trainer_pruned = Trainer(model = model_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer_pruned.validate(reg_on = False)

    breakpoint()
    model_aux= copy.deepcopy(model_pruned)
 

    _, sp_rate0_pr = at.sparsityRate(model_pruned)
    pr_par0_pr = pruned_par(model_pruned)
    tot0_pr = par_count(model_pruned)

    compress_vgg(model_pruned)

    breakpoint()

    #print(model)
    # new_tot = par_count(model)
    # trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)



    _, sp_rate1_pr = at.sparsityRate(model_pruned)
    pr_par1_pr = pruned_par(model_pruned)
    tot1_pr = par_count(model_pruned)


    print('New stats ', tot0_pr-tot1_pr,' perc ', 100*(tot0_pr-tot1_pr)/tot0_pr )


    return model_pruned, dataset



def unit_test_501000(name):
    
    name='saves/save_V0.0.1-_resnet50_Imagenet_lr0.1_l0.6_a0.001_e150+50_bs256_t0.0001_m0.9_wd0.0001_mlstemp2_Mscl1.0/checkpoint.th'
    # model_pruned=torch.nn.DataParallel(resnetBig_pruned.__dict__['resnet50'](10))
    model_pruned= resnetBig_imgNet_pruned.resnet50()

    qp.prune_thr(model_pruned,1.e-30)

    # base_checkpoint=torch.load("saves/save_" + args.name +"/checkpoint.th")
    base_checkpoint=torch.load(name)
    if 'fc.weight' not in base_checkpoint['state_dict'].keys():
        dict_temp=base_checkpoint['state_dict']
        dict_temp['fc.weight']=dict_temp['fc.weight_orig']
        dict_temp['fc.bias']=dict_temp['fc.bias_orig']
        dict_temp.pop('fc.weight_orig')
        dict_temp.pop('fc.bias_orig')
        dict_temp.pop('fc.weight_mask')
        dict_temp.pop('fc.bias_mask')
        base_checkpoint['state_dict']=dict_temp

    model_pruned.load_state_dict(base_checkpoint['state_dict'])

    model_pruned.eval()
    model_pruned(torch.rand([1,3,256,256]))

    dataset = dl.load_dataset("Imagenet", 256)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    model_pruned.cuda()
    model_pruned(torch.rand([1,3,256,256]).cuda())



    optimizer_pruned = torch.optim.SGD(model_pruned.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=5e-4)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pruned,
                                                        milestones=[300], last_epoch= - 1)


    trainer_pruned = Trainer(model = model_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
                                            criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)

    trainer_pruned.validate(reg_on = False)

    # breakpoint()
    model_aux= copy.deepcopy(model_pruned)
 

    _, sp_rate0_pr = at.sparsityRate(model_pruned)
    pr_par0_pr = pruned_par(model_pruned)
    tot0_pr = par_count(model_pruned)

    compress_resnet(model_pruned)

    # breakpoint()

    #print(model)
    # new_tot = par_count(model)
    # trainer.validate(reg_on = False)
    trainer_pruned.validate(reg_on = False)



    _, sp_rate1_pr = at.sparsityRate(model_pruned)
    pr_par1_pr = pruned_par(model_pruned)
    tot1_pr = par_count(model_pruned)
    # print("tot berfore and after ",tot0, ' ', tot1, ' pr ', tot0_pr,' ', tot1_pr)
    #print("new_tot ", new_tot)
    # print("sparsity before and after ", sp_rate0, ' ', sp_rate1, ' pr ', sp_rate0_pr,' ', sp_rate1_pr )
    # print("pruned par before and after ", pr_par0, ' ', pr_par1, ' pr ', pr_par0_pr,' ', pr_par1_pr )

    print('New stats ', tot0_pr-tot1_pr,' perc ', 100*(tot0_pr-tot1_pr)/tot0_pr )
    # fully_pruned= resnet_pruned.FullyCompressedResNet(copy.deepcopy(model_pruned))
    # breakpoint()


    # trainer_fully_pruned = Trainer(model = fully_pruned, dataset = dataset, reg = None, lamb = 1.0, threshold = 0.05, threshold_str=1e-4,
    #                                         criterion =criterion, optimizer = optimizer_pruned, lr_scheduler = lr_scheduler, save_dir = "./delete_this_folder", save_every = 44, print_freq = 100)
    # print('Fully pruned model acc ')
    # trainer_fully_pruned.validate(reg_on = False)


 

    # _, sp_rate_fll = at.sparsityRate(fully_pruned)
    # pr_par_fll = pruned_par(fully_pruned)
    # tot_fll = par_count(fully_pruned)

    # print('Final stats ', tot0_pr-tot_fll,' perc ', 100*(tot0_pr-tot_fll)/tot0_pr )


    return model_pruned, dataset
