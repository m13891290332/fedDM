import torch
import numpy as np
from torchvision import datasets, transforms
import os
def get_dataset(dataset, dataset_root, batch_size):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.MNIST(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.MNIST(dataset_root, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)
        class_names = trainset.classes
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR100(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR100(dataset_root, train=False, download=True, transform=transform)
        class_names = trainset.classes
    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)  # TinyImageNet 图像尺寸为 64x64
        num_classes = 200
        mean = [0.4802, 0.4481, 0.3975]  # TinyImageNet 的均值
        std = [0.2302, 0.2265, 0.2262]   # TinyImageNet 的标准差
        
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 假设数据集目录结构符合 ImageFolder 的要求
        trainset = datasets.ImageFolder(root=os.path.join(dataset_root, 'tiny-imagenet-200/train'), 
                                    transform=transform)
        testset = datasets.ImageFolder(root=os.path.join(dataset_root, 'tiny-imagenet-200/val'), 
                                    transform=transform)
        class_names = [str(c) for c in range(num_classes)]  # 或用实际类别名
    else:
        exit(f'unknown dataset: {dataset}')
    
    dataset_info = {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'classes_names': class_names,
        'mean': mean,
        'std': std,
    }
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                  num_workers=2)  # pin memory

    return dataset_info, trainset, testset, testloader

class PerLabelDatasetNonIID():
    def __init__(self, dst_train, classes, channel, device):  # images: n x c x h x w tensor
        self.images_all = []
        labels_all = []
        self.indices_class = {c: [] for c in classes}

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    def __len__(self):
        return self.images_all.shape[0]

    def get_random_images(self, n):  # get n random images
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]

    def get_images(self, c, n, avg=False):  # get n random images from class c
        if not avg:
            if len(self.indices_class[c]) == 0:
                # If no samples for this class, return zero tensor with correct shape
                return torch.zeros(n, *self.images_all.shape[1:], device=self.images_all.device)
            elif len(self.indices_class[c]) >= n:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            else:
                sampled_idx = np.random.choice(self.indices_class[c], n - len(self.indices_class[c]), replace=True)
                idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
            return self.images_all[idx_shuffle]
        else:
            if len(self.indices_class[c]) == 0:
                # If no samples for this class, return zero tensor with correct shape
                return torch.zeros(n, *self.images_all.shape[1:], device=self.images_all.device)
            
            sampled_imgs = []
            for _ in range(n):
                if len(self.indices_class[c]) >= 5:
                    idx = np.random.choice(self.indices_class[c], 5, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 5, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
            sampled_imgs = torch.cat(sampled_imgs, dim=0).cuda()
            return sampled_imgs
