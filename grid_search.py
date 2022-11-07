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

from time import time


##########################################################################################
milestones_dict = {"emp1": [120, 200, 230, 250, 350, 400, 450], 
                    "emp2": [35, 70, 105, 140, 175, 210, 245, 280, 315],
                    "emp3": [100, 250, 350, 400, 450], 
                    "emp4": [200, 250, 350, 400, 450],
                    "emp5": [200, 400],
                    "emp6": [75, 150, 250],
                    "emp7": [100, 150]} 

parser = argparse.ArgumentParser(description='Pruning using SPR term')
parser.add_argument('--config', '-c',
                    help="Name of the configuration file")
############################################################################################



class Grid_Search():
    def __init__(self, config_file):
        self.config_file = config_file
    
    def run(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file)
            #Parameters setting
        ##############################################################
        cudnn.benchmark = True
        conf1 = conf["conf1"]
        LRS = to_list_of_float(conf1.get("LRS"))
        LAMBS = to_list_of_float(conf1.get("LAMBS"))
        ALPHAS = to_list_of_float(conf1.get("ALPHAS"))
        arch = conf1.get("arch")
        dset = conf1.get("dset")
        epochs = conf1.getint("epochs")
        finetuning_epochs = conf1.getint("finetuning_epochs")
        batch_size = conf1.getint("batch_size")
        threshold = conf1.getfloat("threshold")
        threshold_str = conf1.getfloat("threshold_str")
        momentum = conf1.getfloat("momentum")
        weight_decay = conf1.getfloat("weight_decay")
        milestones = conf1.get("milestones")
        evaluate = conf1.getboolean("evaluate", "False")
        save_every = conf1.getint("save_every", str((epochs+finetuning_epochs)*0.2))
        print_freq = conf1.getint("print_freq", "100")
        M_scale = conf1.getfloat("M_scale", 1.0)
        structs = conf1.get("structs", 'convs_and_batchnorm')
        base_name = conf1.get("base_name")
        reg_type = conf1.get("reg_type", 'perspReg')
        track_stats= conf1.get("track_stats", False)
        ##############################################################

        ################################################################
        #       Main triple loop on configurations
        ################################################################

        for lr in LRS:
            for lamb in LAMBS:
                for alpha in ALPHAS:

                    name = (base_name + "_" +((reg_type + '_') if reg_type!='perspReg' else'')+ arch + "_" + dset + "_lr" + str(lr) + "_l" + str(lamb) + "_a" + 
                            str(alpha) + "_e" + str(epochs) + "+" + str(finetuning_epochs) + "_bs" + str(batch_size) +
                            "_t" + str(threshold)+ "_tstr" + str(threshold_str) + "_m" + str(momentum) + "_wd" + str(weight_decay) + "_mlst" + milestones + "_Mscl" + str(M_scale)+ "_struct" + structs+'_id'+str(int(time())))

                    save_dir = "saves/save_" + name
                    log_file = open("temp_logs/" + name, "w")
                    sys.stdout = log_file
                    sys.stderr = sys.stdout
                    print(name)
                    
                    if dset == "Cifar10": num_classes = 10
                    elif dset == "Cifar100": num_classes = 100
                    elif dset == "Imagenet": num_classes = 1000

                    model = archs.load_arch(arch, num_classes)
                    dataset = dl.load_dataset(dset, batch_size)


                    # define loss function (criterion) and optimizer
                    criterion = nn.CrossEntropyLoss().cuda()


                    optimizer = torch.optim.SGD(model.parameters(), lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)

                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                        milestones=milestones_dict[milestones], last_epoch= - 1)

                    if lamb == 0.0 and alpha == 0.0:
                        reg = at.noReg
                    elif reg_type == 'perspReg':
                        #Creating the perspective regualriation function
                        #Compute M values for each layer using a trained model 
                        torch.save(model.state_dict(),name + "rand_init.ph")
                        base_checkpoint=torch.load("saves/save_" + arch + "_" + dset + "_first_original/checkpoint.th")
                        model.load_state_dict(base_checkpoint['state_dict'])
                        if structs == 'single_convs':
                            M=at.layerwise_M(model, scale = M_scale) #a dictionary withe hte value of M for each layer of the model
                        elif structs == 'convs_and_batchnorm':
                             M=at.blockwise_M(model, scale = M_scale) 

                        model.load_state_dict(torch.load(name  + "rand_init.ph"))
                        os.remove(name + "rand_init.ph")

                        print("M values:\n",M)
                        
                        reg = pReg.PerspReg(alpha=alpha,M=M, option =structs, track_stats=track_stats)
                    else: reg = oReg.__dict__[reg_type](alpha, scaled=False,option=structs)




                    trainer = Trainer(model = model, dataset = dataset, reg = reg, lamb = lamb, threshold = threshold, threshold_str = threshold_str, 
                                        criterion =criterion, optimizer = optimizer, lr_scheduler = lr_scheduler, save_dir = save_dir, save_every = save_every, print_freq = print_freq)


                    if dset == "Imagenet":
                        trainer.top5_comp = True

                    if evaluate:
                        trainer.validate()
                    else:
                        trainer.train(epochs, finetuning_epochs)

                    #creating the actual pruned model

                    model_pruned=torch.nn.DataParallel(resnet_pruned.__dict__[arch](num_classes))
                    qp.prune_thr(model_pruned,1.e-12)
                    base_checkpoint=torch.load(save_dir+'/model_best_val.th')
                    model_pruned.load_state_dict(base_checkpoint['state_dict'])

                    model_pruned.eval()

                    trainer_pr = Trainer(model = model_pruned, dataset = dataset, reg = reg, lamb = lamb, threshold = threshold, threshold_str = threshold_str, 
                                        criterion =criterion, optimizer = None, lr_scheduler = lr_scheduler, save_dir = save_dir, save_every = save_every, print_freq = print_freq)

                    model_pruned(torch.rand([1,3,32,32]))

                    en.compress_resnet(model_pruned)

                    trainer_pr.validate(reg_on = False)
                    
                    print(' Real pruned parameter ',en.par_count(model)-en.par_count(model_pruned))
                    if track_stats:
                        print(' Reg cases stats ', reg.stats)
                    
                    log_file.close()
                    
                    
def to_list_of_float(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [float(el) for el in list_string]

def to_list_of_int(list_string, sep = ","):
    list_string = list_string.split(sep)
    return [int(el) for el in list_string]



def main():
    args = parser.parse_args()
    grid_search = Grid_Search(args.config)
    grid_search.run()

if __name__ == "__main__":
    main()
                    
