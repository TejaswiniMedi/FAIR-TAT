import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import Subset


# PATH = 'data/cifar-10-batches-py'
PATH = "data"
def dev(id):
    return f'cuda:{id}'

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std =  (0.2675, 0.2565, 0.2761)

mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

mu_100 = torch.tensor(cifar100_mean).view(3,1,1)
std_100 = torch.tensor(cifar100_std).view(3,1,1)

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

def normalize_cifar_100(x):
    return (x - mu_100.to(x.device))/(std_100.to(x.device))

def get_dataset(
        dataset = "cifar10",
        batch_size_train = 128,
        batch_size_valid = 128,
        batch_size_test  = 128,
        num_workers_train = 6,
        num_workers_valid = 3,
        num_workers_test  = 3,
    ):
    if not os.path.exists("data"):
        os.mkdir("data")
    
    fn_ds = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_set = fn_ds(
        PATH,
        train = True,
        download = True,
        transform = train_transform
    )
    valid_set = fn_ds(
        PATH,
        train = True,
        download = True,
        transform = test_transform
    )
    
    ori_label = torch.tensor(train_set.targets)
    num_labels = len(np.unique(train_set.targets))
    # n = 100 # for each classes (2% of 5000)
    n = int(50_000 / num_labels  / 50)
    valid_index, train_index = [], []
    for i in range(num_labels):
        valid_index_i = (ori_label==i).nonzero()[:n]
        train_index_i = (ori_label==i).nonzero()[n:]
        valid_index.append(valid_index_i)
        train_index.append(train_index_i)
    valid_index = torch.cat(valid_index, dim=0).flatten()
    train_index = torch.cat(train_index, dim=0).flatten()
    print(f"valid_index: {len(valid_index)}.")
    print(f"train_index: {len(train_index)}.")
    
    N = len(train_index)
    order = np.random.permutation(N)
    train_index = train_index[order]
    
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size = batch_size_train,
        shuffle = False,
        sampler = train_sampler,
        pin_memory = True,
        persistent_workers = num_workers_train > 1,
        num_workers = num_workers_train
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = batch_size_valid,
        shuffle = False,
        sampler = valid_sampler,
        pin_memory = True,
        persistent_workers = num_workers_valid > 1,
        num_workers = num_workers_valid
    )
    test_loader = torch.utils.data.DataLoader(
        fn_ds(
            PATH,
            train = False,
            download = True,
            transform = test_transform
        ),
        batch_size = batch_size_test,
        shuffle = False,
        pin_memory = True,
        persistent_workers = num_workers_test > 1,
        num_workers = num_workers_test
    )
    
    return train_loader, valid_loader, test_loader

# def load_dataset(dataset='cifar10', batch_size=128):
#     if dataset == 'cifar10':
#         transform_ = transforms.Compose([transforms.ToTensor()])
#         train_transform_ = transforms.Compose([
#             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()])
#         train_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(PATH, train=True, download=True, transform=train_transform_),
#             batch_size=batch_size, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
#             batch_size=batch_size, shuffle=False)

#         return train_loader, test_loader
#     elif dataset == 'cifar100':
#         transform_ = transforms.Compose([transforms.ToTensor()])

#         train_transform_ = transforms.Compose([
#             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()])
#         train_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR100(PATH, train=True, download=True, transform=train_transform_),
#             batch_size=batch_size, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR100(PATH, train=False, download=True, transform=transform_),
#             batch_size=batch_size, shuffle=False)

#         return train_loader, test_loader

# def load_valid_dataset(dataset='cifar10', batch_size=128):
#     if dataset == 'cifar10':
#         train_loader, valid_loader = torch.load('data/split_dataset.pth')

#         # test loader
#         transform_ = transforms.Compose([transforms.ToTensor()])
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
#             batch_size=batch_size, shuffle=False)
#         return train_loader, valid_loader, test_loader
    
#     elif dataset == 'cifar100':
#         train_loader, valid_loader = torch.load('data/split_dataset_100.pth')

#         # test loader
#         transform_ = transforms.Compose([transforms.ToTensor()])
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR100(PATH, train=False, download=True, transform=transform_),
#             batch_size=batch_size, shuffle=False)
#         return train_loader, valid_loader, test_loader

# def load_cw_dataset(dataset='cifar10', batch_size=128, valid=True):
#     if dataset == 'cifar10':
#         if valid:
#             train_loader, valid_loader = torch.load('data/split_dataset.pth')
#         else:
#             train_transform_ = transforms.Compose([
#             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()])
#             train_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(PATH, train=True, download=True, transform=train_transform_),
#             batch_size=batch_size, shuffle=True)
        
#         # test loader
#         transform_ = transforms.Compose([transforms.ToTensor()])
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(PATH, train=False, download=True, transform=transform_),
#             batch_size=batch_size, shuffle=False)
#         data = torch.cat([x for (x,y) in test_loader], dim=0)
#         label = torch.cat([y for (x,y) in test_loader], dim=0)

#         cw_test_loader = []
#         for i in range(10):
#             index = (label==i).nonzero().flatten()
#             loader = []
#             for j in range(10):
#                 curr_index = index[j*100:(j+1)*100]
#                 loader.append((data[curr_index], label[curr_index]))
#             cw_test_loader.append(loader)

#         if valid:
#             return train_loader, valid_loader, cw_test_loader
#         else:
#             return train_loader, cw_test_loader

def weight_average(model, new_model, decay_rate, init=False):
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    if init:
        decay_rate = 0
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict)

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]


corruptions = load_txt('/home/tejaswini/Experiments_Teja/Targeted-Adversarial-Training/corruptions.txt') # path to corruptions

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)

