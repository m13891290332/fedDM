import argparse
import json
import os

import numpy as np
from torchvision import datasets, transforms

def partition(args):
    np.random.seed(args.seed)

    # prepare datasets for then partition latter
    if args.dataset == 'MNIST':
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.MNIST(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = [str(c) for c in range(num_classes)]
    elif args.dataset == 'CIFAR10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR100(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    else:
        exit(f'unknown dataset: f{args.dataset}')

    K = num_classes
    labels = np.array(dataset.targets, dtype='int64')
    N = labels.shape[0]

    dict_users = {}
    
    # 初始化每个客户端的数据索引列表
    for client_id in range(args.client_num):
        dict_users[client_id] = []
    
    # 直接对每个类使用狄利克雷分布分配数据给所有客户端
    for class_id in range(num_classes):
        # 获取该类的所有数据索引
        idx_k = np.where(labels == class_id)[0]
        np.random.shuffle(idx_k)
        
        # 使用狄利克雷分布为所有客户端分配该类的数据
        proportions = np.random.dirichlet(np.repeat(args.alpha, args.client_num))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # 将数据分配给相应的客户端
        idx_splits = np.split(idx_k, proportions)
        for i, client_id in enumerate(range(args.client_num)):
            dict_users[client_id].extend(idx_splits[i].tolist())
    
    # 为每个客户端打乱数据顺序
    for client_id in range(args.client_num):
        np.random.shuffle(dict_users[client_id])

    # 计算每个客户端的类别分布
    net_cls_counts = {}
    dict_classes = {}

    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        
        # 记录每个客户端拥有的类别（数据量大于0的类别）
        dict_classes[net_i] = [int(cls) for cls in unq]

    print('Data statistics: %s' % str(net_cls_counts))
    print('Client class assignments:')
    for client_id, classes in dict_classes.items():
        print(f'Client {client_id}: classes {classes} (total samples: {len(dict_users[client_id])})')

    save_path = os.path.join(os.path.dirname(__file__), '../', 'split_file')
    file_name = f'{args.dataset}_only_dirichlet_client_num={args.client_num}_alpha={args.alpha}.json'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, file_name), 'w') as json_file:
        json.dump({
            "client_idx": [[int(idx) for idx in dict_users[i]] for i in range(args.client_num)],
            "client_classes": [[int(cls) for cls in dict_classes[i]] for i in range(args.client_num)],
        }, json_file, indent=4)

if __name__ == "__main__":
    partition_parser = argparse.ArgumentParser()

    partition_parser.add_argument("--dataset", type=str, default='CIFAR10')
    partition_parser.add_argument("--client_num", type=int, default=50)
    partition_parser.add_argument("--alpha", type=float, default=0.1)
    partition_parser.add_argument("--dataset_root", type=str, default='/home/MaCS/fedDM/datasets/torchvision')
    partition_parser.add_argument("--seed", type=int, default=19260817)
    args = partition_parser.parse_args()
    partition(args)