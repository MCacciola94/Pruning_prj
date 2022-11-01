'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from aux_tools import mask_idx_list
import eval_net as en

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.pruned_filters_conv2=[]
        self.pruned_shortcut=[]
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # self.shortcut = LambdaLayer(lambda x:
                                            # F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x: x[:, :, ::2, ::2])
                self.pruned_shortcut = [i for i  in range(planes//4)]+ [i+3*(planes//4) for i  in range(planes//4)]
                self.numb_added_planes = (planes//4)*2
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip_x = self.shortcut(x)
        out=self.incompatible_sum(skip_x,out)

        out = F.relu(out)
        return out

    def incompatible_sum(self,skip_x,out):
        if self.pruned_filters_conv2!=[] or self.pruned_shortcut !=[]:
            device='cpu'
            if torch.cuda.is_available():
                device= 'cuda:0'
            # breakpoint()
            original_num_filters=len(self.pruned_filters_conv2)+self.conv2.weight.size(0)
            combined_unpruned= [i for i in range(original_num_filters) if i not in self.pruned_filters_conv2 and i not in self.pruned_shortcut ]
            combined_pruned= [i for i in range(original_num_filters) if i in self.pruned_filters_conv2 and i in self.pruned_shortcut ]
            shortcut_only_unpruned= [i for i in range(original_num_filters) if i in self.pruned_filters_conv2 and i not in self.pruned_shortcut ]
            conv2_only_unpruned= [i for i in range(original_num_filters) if i not in self.pruned_filters_conv2 and i in self.pruned_shortcut ]
            
            if combined_unpruned !=[]:
                comb_unpr_conv2 = mask_idx_list(combined_unpruned,self.pruned_filters_conv2)
                comb_unpr_shortcut = mask_idx_list(combined_unpruned,self.pruned_shortcut)

                out_aux = out[:,comb_unpr_conv2,:]
                skip_x_aux = skip_x[:,comb_unpr_shortcut,:]
                out_aux += skip_x_aux
            
                final_idxs_aux = mask_idx_list(combined_unpruned,combined_pruned)

            if shortcut_only_unpruned !=[]:
                final_idxs_skip_x = mask_idx_list(shortcut_only_unpruned,combined_pruned)
                shortcut_only_unpruned = mask_idx_list(shortcut_only_unpruned,self.pruned_shortcut)

            if conv2_only_unpruned !=[]:
                final_idxs_out = mask_idx_list(conv2_only_unpruned,combined_pruned)
                conv2_only_unpruned = mask_idx_list(conv2_only_unpruned,self.pruned_filters_conv2)
 
            final_shape= [out.size(0),original_num_filters-len(combined_pruned),out.size(2),out.size(3)]
            t_aux=torch.zeros(final_shape).to(device)
            
            if combined_unpruned !=[]:
                t_aux[:,final_idxs_aux,:]=out_aux

            if conv2_only_unpruned !=[]:
                t_aux[:,final_idxs_out,:]=out[:,conv2_only_unpruned,:]

            if shortcut_only_unpruned !=[]:
                t_aux[:,final_idxs_skip_x,:]=skip_x[:,shortcut_only_unpruned,:]

            return t_aux

        else: return out+skip_x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out

    def conv_batchnorm_blocks(self):
        blocks_list=[]
        blocks_list.append( {'conv':self.conv1,'batchnorm':self.bn1, 'id':0})
        i=1
        for b in self.modules():
            if isinstance(b, BasicBlock):
                blocks_list.append( {'conv':b.conv1,'batchnorm':b.bn1, 'id':i})
                blocks_list.append( {'conv':b.conv2,'batchnorm':b.bn2, 'id':i+1})
                i+=2
                if isinstance(b.shortcut,nn.Sequential) and len(b.shortcut)>0 :
                    blocks_list.append( {'conv':b.shortcut[0],'batchnorm':b.shortcut[1], 'id':i})
                    i+=1
        yield from blocks_list

    def block_layer_sequence(self):
       
        
        for  i in range(len(self.layer1)-1):
                yield self.layer1[i], self.layer1[i+1].conv1
        yield self.layer1[-1], self.layer2[0].conv1

        
        for  i in range(len(self.layer2)-1):
                yield self.layer2[i], self.layer2[i+1].conv1
        yield self.layer2[-1], self.layer3[0].conv1

        
        for  i in range(len(self.layer3)-1):
                yield self.layer3[i], self.layer3[i+1].conv1
        yield self.layer3[-1], self.linear



    def blocks_sequence(self):
       
        for  i in range(len(self.layer1)-1):
                yield self.layer1[i], self.layer1[i+1]
        yield self.layer1[-1], self.layer2[0]

        
        for  i in range(len(self.layer2)-1):
                yield self.layer2[i], self.layer2[i+1]
        yield self.layer2[-1], self.layer3[0]

        
        for  i in range(len(self.layer3)-1):
                yield self.layer3[i], self.layer3[i+1]
        yield self.layer3[-1], self.linear

def resnet20(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes = num_classes)


def resnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes = num_classes)


def resnet44(num_classes):
    return ResNet(BasicBlock, [7, 7, 7],  num_classes = num_classes)


def resnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9],  num_classes = num_classes)


def resnet110(num_classes):
    return ResNet(BasicBlock, [18, 18, 18],  num_classes = num_classes)


def resnet1202(num_classes):
    return ResNet(BasicBlock, [200, 200, 200],  num_classes = num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


class FullyCompressedBlock(nn.Module):
    
    def __init__(self, old_block,mod_type):
        super(FullyCompressedBlock, self).__init__()

        self.module_type=mod_type

        if self.module_type =='shortcut':
            self.shortcut = old_block.shortcut
            
        elif self.module_type =='conv-bn':
            self.conv1 = old_block.conv1
            self.conv2 = old_block.conv2
            self.bn1 = old_block.bn1
            self.bn2 = old_block.bn2
        else:
             print('NOTHUNG TO COMPRESS')
             
             
    def forward(self, x):
        # print('x ', x.shape)


        if self.module_type =='shortcut':
            out = F.relu(self.shortcut(x))
            # print('out ',out.shape)

            return out
        
        if self.module_type =='conv-bn':
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            # print('out ',out.shape)

            return out
        
        print('WARNING: Empty block')
        return 0


class FullyCompressedResNet(nn.Module):
    def __init__(self, net):
        super(FullyCompressedResNet, self).__init__()
        if isinstance(net, nn.DataParallel):
            net=net.module

        device='cpu'
        if torch.cuda.is_available():
            device= 'cuda:0'

        self.conv1=net.conv1
        self.bn1=net.bn1
        self.layers=[]
        compress_later={}
        idx_pr_old=[]

        for block,next_block in net.blocks_sequence():

            original_out_channles=len(block.pruned_filters_conv2)+block.conv2.out_channels
            idx_already_pr=[i for i in range(original_out_channles) if (i in block.pruned_filters_conv2 and i in block.pruned_shortcut)]
            curr_out_channel =original_out_channles-len(idx_already_pr)

            if torch.all(block.conv1.weight==0) or torch.all(block.conv2.weight==0):
                # breakpoint()
                idx_conv=[i for i in range(original_out_channles) if (i not in block.pruned_filters_conv2 and i in block.pruned_shortcut)]

                if isinstance(block.shortcut, LambdaLayer):
                        idx_pr_old= [i+block.numb_added_planes//2 for i in idx_pr_old]

                idx_pr_old=mask_idx_list(idx_pr_old,idx_already_pr)
                idx_conv+=[i for i in idx_pr_old if i not in idx_conv]
                idx_pr=mask_idx_list(idx_conv,idx_already_pr)
                idx_pr_old=idx_conv 

                if not(isinstance(block.shortcut,nn.Sequential) and len(block.shortcut)==0):
                    self.layers.append(FullyCompressedBlock(block,'shortcut'))
                    

            elif len(block.pruned_filters_conv2)+block.conv2.out_channels==len(block.pruned_shortcut):
                idx_sc=[i for i in range(original_out_channles) if (i not in block.pruned_shortcut and i in block.pruned_filters_conv2)]
                idx_sc+=[i for i in idx_pr_old if i not in idx_sc]
                idx_pr=mask_idx_list(idx_sc,idx_already_pr)
                self.layer.append(FullyCompressedBlock(block,'conv-bn'))
                idx_pr_old=idx_sc 
            
            else:
                self.layers.append(block)
                idx_pr_old=[]
                continue

                
            if hasattr(next_block,'conv1'):
                idx_channels_next_layer_not_pr= [i for i in range(next_block.conv1.in_channels) if (i not in idx_pr)]
                idx_channels_next_layer_not_pr = torch.Tensor(idx_channels_next_layer_not_pr).type(torch.int).to(device)

                en.compress_conv_channels(next_block.conv1,idx_channels_next_layer_not_pr) 
                compress_later[next_block]=idx_pr   
                # en.compress_shortcut_inp(next_block,idx_pr)
                
            
    
            if isinstance(next_block,nn.Linear):
                idx_channels_next_layer_not_pr = [i for i in range(curr_out_channel) if (i not in idx_pr )]
                channel_size = next_block.in_features/curr_out_channel
                # print(channel_size)
                idx_aux=[]
                for i in  idx_channels_next_layer_not_pr:
                    idx_aux += [i*channel_size + j for j in range(int(channel_size))]
                
                idx_channels_next_layer_not_pr = idx_aux
            
                idx_channels_next_layer_not_pr = torch.Tensor(idx_channels_next_layer_not_pr).type(torch.int).to(device)
                
                en.compress_linear_in(next_block,idx_channels_next_layer_not_pr)    
                
                linear_layer=next_block

        for key in compress_later.keys():
            en.compress_shortcut_inp(key,compress_later[key])
        self.layers=nn.Sequential(*(self.layers))
        self.linear=linear_layer



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out







if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name](num_classes = 10))
            print()