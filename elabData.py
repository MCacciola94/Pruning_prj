import os
from collections import defaultdict
import pandas as pd

########################################################################################
param_dict= {"lr": "lr", "lambda": "l", "alpha": "a", "epochs": "e", "batch_size": "bs",
             "threshold": "t", "momentum": "m", "weight_decay": "wd", "M_scale": "Mscl"}
########################################################################################

def create_csv(path = "logs/resnet20"):
    name_list = os.listdir(path +"/raw_logs")
    
    table_cols = defaultdict(lambda: [])
    
    for log_name in name_list:
        print(log_name)
        f = open(path+'/raw_logs/'+log_name,'r')
        s = f.readline()
        s = s.replace("\n", "")
        s = s.split("_")
        par_values = {}
        for el in s:
            par_name, par_val = take_par_and_val(el)
            if par_name != "not_found":
                if par_name == "epochs":
                    e1, e2 = par_val.split("+")
                    table_cols["epochs"].append(int(e1))
                    table_cols["ft_epochs"].append(int(e2))
                else:
                    table_cols[par_name].append(float(par_val))
            else: print(el)

      

        ls=f.readlines()


        i = 1
        while not('Best accuracy:' in ls[-i]):
            i = i+1

        s = ls[-i].split()
        table_cols["Final_Accuracy"].append(s[2])

        while not('Total parameter pruned:' in ls[-i]):
            i = i+1

        s = ls[-i].split()
        table_cols["Sparsity"].append(int(s[5]))

        while not('Elapsed time' in ls[-i]):
            i = i+1
        s = ls[-i].split()
        table_cols["Tot_time"].append(s[4])



        while not('FINETUNING' in ls[-i]):
            i = i+1

        while not('elapsed time' in ls[-i]):
            i = i+1
        s = ls[-i].split()
        table_cols["Pr_time"].append(s[3])

        while not('Prec@1' in ls[-i]):
            i = i+1

        s = ls[-i].split()
        table_cols["Accuracy"].append(float(s[2]))

        while not('Epoch:' in ls[-i]):
            i = i+1

        s = ls[-i].split()
        s[10] = s[10].replace("(","")
        s[10] = s[10].replace(")","")
        table_cols["Train_loss"].append(float(s[10]))
      

        f.close()


    table_cols = dict(table_cols)
    #breakpoint()
    table_cols_df = pd.DataFrame(table_cols)
    table_cols_df = table_cols_df.astype({"batch_size": int})
    table_cols_df.to_csv(path + "/" + "Full_tab.csv", sep = "\t", index = False)
    
def write_small_tab(path, name, cols = ["lambda", "alpha", "Final_Accuracy", "Sparsity"]):
    full_tab = pd.read_csv(path + "/Full_tab.csv", sep = "\t")
    small_tab = full_tab[cols]
    small_tab.to_csv(path+"/" + name, sep = "\t", index = False)
    return small_tab

def take_par_and_val(s):
    for k, v in param_dict.items():
        n = len(v)
        if s[:n] == v and ((s[n:].replace(".", "")).isnumeric() or (v == "e" and "+" in s[n:]) or ("1e" in s[n:])):
            return k, s[n:]

    return "not_found", None

