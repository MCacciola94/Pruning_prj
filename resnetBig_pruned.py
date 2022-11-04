
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:45:26 2019
@author: Souvik
"""


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import numpy as np
import torch.nn as nn
from aux_tools import mask_idx_list
import torch.nn.functional as F



class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.pruned_filters_conv2=[]
        self.bn2_biases={}

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes)

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



class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, in_planes, planes, stride=1):

        super(Bottleneck, self).__init__()

        self.pruned_filters_conv3=[]
        self.pruned_shortcut=[]

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(self.expansion*planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes)

            )

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        skip_x = self.shortcut(x)

        out= self.incompatible_sum(skip_x,out)

        out = F.relu(out)

        return out

    def incompatible_sum(self,skip_x,out):
        if self.pruned_filters_conv3!=[] or self.pruned_shortcut !=[]:
            
            device='cpu'
            if torch.cuda.is_available():
                device= 'cuda:0'
            # breakpoint()
            original_num_filters=len(self.pruned_filters_conv3)+self.conv3.weight.size(0)
            combined_unpruned= [i for i in range(original_num_filters) if i not in self.pruned_filters_conv3 and i not in self.pruned_shortcut ]
            combined_pruned= [i for i in range(original_num_filters) if i in self.pruned_filters_conv3 and i in self.pruned_shortcut ]
            shortcut_only_unpruned= [i for i in range(original_num_filters) if i in self.pruned_filters_conv3 and i not in self.pruned_shortcut ]
            conv3_only_unpruned= [i for i in range(original_num_filters) if i not in self.pruned_filters_conv3 and i in self.pruned_shortcut ]
            
            if combined_unpruned !=[]:
                comb_unpr_conv3 = mask_idx_list(combined_unpruned,self.pruned_filters_conv3)
                comb_unpr_shortcut = mask_idx_list(combined_unpruned,self.pruned_shortcut)

                out_aux = out[:,comb_unpr_conv3,:]
                skip_x_aux = skip_x[:,comb_unpr_shortcut,:]
                out_aux += skip_x_aux
            
                final_idxs_aux = mask_idx_list(combined_unpruned,combined_pruned)

            if shortcut_only_unpruned !=[]:
                final_idxs_skip_x = mask_idx_list(shortcut_only_unpruned,combined_pruned)
                shortcut_only_unpruned = mask_idx_list(shortcut_only_unpruned,self.pruned_shortcut)

            if conv3_only_unpruned !=[]:
                final_idxs_out = mask_idx_list(conv3_only_unpruned,combined_pruned)
                conv3_only_unpruned = mask_idx_list(conv3_only_unpruned,self.pruned_filters_conv3)

            # breakpoint()
            final_shape= [out.size(0),original_num_filters-len(combined_pruned),out.size(2),out.size(3)]
            t_aux=torch.zeros(final_shape).to(device)
            
            if combined_unpruned !=[]:
                t_aux[:,final_idxs_aux,:]=out_aux

            if conv3_only_unpruned !=[]:
                t_aux[:,final_idxs_out,:]=out[:,conv3_only_unpruned,:]

            if shortcut_only_unpruned !=[]:
                t_aux[:,final_idxs_skip_x,:]=skip_x[:,shortcut_only_unpruned,:]

            return t_aux

        else: return out+skip_x



class ResNet(nn.Module):

    def __init__(self, block, num_blocks, init_weight = True, num_classes=100):

        super(ResNet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)


        if init_weight:
            self._initialize_weights()


    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain = 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

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

                if isinstance(b, Bottleneck):
                    blocks_list.append( {'conv':b.conv1,'batchnorm':b.bn1, 'id':i})
                    blocks_list.append( {'conv':b.conv2,'batchnorm':b.bn2, 'id':i+1})
                    blocks_list.append( {'conv':b.conv3,'batchnorm':b.bn1, 'id':i+2})
                    i+=3
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
        yield self.layer3[-1], self.layer4[0].conv1


        for  i in range(len(self.layer4)-1):
                yield self.layer4[i], self.layer4[i+1].conv1
        yield self.layer4[-1], self.linear

    def blocks_sequence(self):
       
        for  i in range(len(self.layer1)-1):
                yield self.layer1[i], self.layer1[i+1]
        yield self.layer1[-1], self.layer2[0]

        
        for  i in range(len(self.layer2)-1):
                yield self.layer2[i], self.layer2[i+1]
        yield self.layer2[-1], self.layer3[0]

        for  i in range(len(self.layer3)-1):
                yield self.layer3[i], self.layer3[i+1]
        yield self.layer3[-1], self.layer4[0]

        
        for  i in range(len(self.layer4)-1):
                yield self.layer4[i], self.layer4[i+1]
        yield self.layer4[-1], self.linear

def resnet18(num_classes=10):

    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34():

    return ResNet(BasicBlock, [3,4,6,3])

def resnet50(num_classes=10):

    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():

    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():

    return ResNet(Bottleneck, [3,8,36,3])



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        print(p.size())
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

net = resnet18()