#Import pkgs
import sys
import os
import configparser
import argparse

#Pytorch pkgs
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import prune

#My pkgs
import aux_tools as at
import perspReg as pReg
import otherReg as oReg
import architectures as archs
import data_loaders as dl
from trainer import Trainer
import resnet_pruned
import eval_net as en
import quik_pruning as qp
from otherReg import SCALED
from trainer import BN_SRCH_ITER

from time import time


##########################################################################################
milestones_dict = {"emp1": [120, 200, 230, 250, 350, 400, 450], 
                    "emp2": [35, 70, 105, 140, 175, 210, 245, 280, 315],
                    "emp3": [100, 250, 350, 400, 450], 
                    "emp4": [200, 250, 350, 400, 450],
                    "emp5": [200, 400],
                    "emp6": [75, 150, 250],
                    "emp7": [100, 150],
                    "emp8": [120, 200, 230, 250, 400, 450,485],
                    "emp9": [120, 200, 230, 250, 450]} 

parser = argparse.ArgumentParser(description='Pruning using SPR term')
parser.add_argument('--config', '-c',
                    help="Name of the configuration file")
parser.add_argument('--resume_path', '-rp',
                    help="resume path")
############################################################################################



class Runner():
    def __init__(self, args):
        self.config_file = args.config
        self.resume_path = args.resume_path
    
    def run(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file)
            #Parameters setting
        ##############################################################
        cudnn.benchmark = True
        conf1 = conf["conf1"]
        arch = conf1.get("arch")
        dset = conf1.get("dset")
        batch_size = conf1.getint("batch_size")



        resume_path = self.resume_path

        ##############################################################

        ################################################################
        #       Main triple loop on configurations
        ################################################################



   
        save_dir = resume_path
        
        if dset == "Cifar10": num_classes = 10
        elif dset == "Cifar100": num_classes = 100
        elif dset == "Imagenet": num_classes = 1000
        if 'Cifar' in dset:
            img_len =32
        else: 
            img_len = 256

        model = archs.load_arch(arch, num_classes, resume = resume_path)#,already_pruned = False)
        dataset = dl.load_dataset(dset, batch_size)


        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()


        optimizer = torch.optim.SGD(model.parameters(), 0.0,
                                    momentum=0.0,
                                    weight_decay=0.0)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[1e+100], last_epoch= - 1)


        reg = at.noReg

        trainer = Trainer(model = model, dataset = dataset, reg = reg, lamb = 0.0, threshold = 0.0, threshold_str = 0.0, 
                            criterion  =criterion, optimizer = None, lr_scheduler = lr_scheduler, save_dir = "del_this", save_every = 1e+10, print_freq = 100)


        if dset == "Imagenet":
            trainer.top5_comp = True

        # trainer.validate(reg_on = False)
        model(torch.rand([1,3,img_len,img_len]).cuda())



        #creating the actual pruned model

        model_pruned=archs.load_arch_pruned(arch, num_classes)
        qp.prune_thr(model_pruned,1.e-12)
        base_checkpoint=torch.load(save_dir)
        model_pruned.load_state_dict(base_checkpoint['state_dict'])

        model_pruned.eval()

        trainer_pr = Trainer(model = model_pruned, dataset = dataset, reg = reg, lamb = 0.0, threshold = 0.0, threshold_str = 0.0, 
                            criterion =criterion, optimizer = None, lr_scheduler = lr_scheduler, save_dir = "del_this", save_every = 1e+100, print_freq = 100)


        
        model_pruned(torch.rand([1,3,img_len,img_len]).cuda())
        

        if 'resnet' in arch:
            en.compress_resnet(model_pruned)
        else:
            en.compress_vgg(model_pruned)

        # trainer_pr.validate(reg_on = False)
        model_pruned(torch.rand([1,3,img_len,img_len]).cuda())

        
        print(' Real pruned parameter ',en.par_count(model)-en.par_count(model_pruned))
        import get_flops
        flops_orig, params= get_flops.measure_model(model,'cuda:0',3,img_len,img_len, False)
        print('flops ', flops_orig, 'params', params)
        flops, params= get_flops.measure_model(model_pruned,'cuda:0',3,img_len,img_len, False)
        print('Pruned: flops ', flops, 'params', params, ' perc flops ',flops/flops_orig*100 )
        print(en.par_count(model))
        # breakpoint()

        
                    
                    
def to_list_of_float(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [float(el) for el in list_string]

def to_list_of_int(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [int(el) for el in list_string]



def main():
    args = parser.parse_args()
    runner = Runner(args)
    runner.run()

if __name__ == "__main__":
    main()
                    
