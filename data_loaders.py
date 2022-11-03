import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


###########################################################
available_datasets = ["Cifar10", "Cifar100", "Imagenet"]
Default_data_path = "/local1/caccmatt/imageNet/try2/ILSVRC/Data/CLS-LOC/"
###########################################################

def is_available(name):
    return name in available_datasets

def load_dataset(name, batch_size, data_path = None):

    if not(is_available(name)):
        print("Dataset requested not avilable")
        return None

    if name == "Cifar10":
        return cifar10_loader(batch_size)

    if name == "Cifar100":
        return cifar100_loader(batch_size)
    
    if name == "Imagenet":
        if data_path is None:
            data_path = Default_data_path

        return imagenet_loader(batch_size, data_path)


def cifar10_loader(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download = True),
        batch_size = batch_size, shuffle = True,
        num_workers = 4, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train = False, transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)
    
    stable_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train = True, transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)
    return {"train_loader": train_loader, "valid_loader": val_loader, "stable_train_loader": stable_train_loader}





def cifar100_loader(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download = True),
        batch_size = batch_size, shuffle = True,
        num_workers = 4, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train = False, transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)
    
    stable_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train = True, transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)
        
    return {"train_loader": train_loader, "valid_loader": val_loader, "stable_train_loader": stable_train_loader}








def imagenet_loader(batch_size, data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True,
    num_workers = 4, pin_memory = True, sampler = None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)

    stable_train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = 128, shuffle = False,
        num_workers = 4, pin_memory = True)

    return {"train_loader": train_loader, "valid_loader": val_loader, "stable_train_loader": stable_train_loader}