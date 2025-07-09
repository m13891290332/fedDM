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
    dict_classes = {}

    # 确保每个客户端分配到恰好2个类
    classes_per_client = 2
    
    # 检查客户端数量是否合理
    if args.client_num * classes_per_client > num_classes * 10:
        raise ValueError(f"Too many clients ({args.client_num}) for {num_classes} classes with {classes_per_client} classes per client")
    
    # 为每个客户端分配类别
    all_classes = list(range(num_classes))
    client_class_assignment = {}
    
    # 创建一个列表，每个类出现在恰好两个客户端中
    class_assignments = []
    for class_id in range(num_classes):
        class_assignments.extend([class_id] * 10)  # 每个类分配给2个客户端
    
    # 随机打乱类别分配
    np.random.shuffle(class_assignments)
    
    # 为每个客户端分配类别
    for client_id in range(args.client_num):
        dict_classes[client_id] = []
    
    # 分配类别给客户端，确保每个客户端有2个类
    class_idx = 0
    for client_id in range(args.client_num):
        assigned_classes = []
        while len(assigned_classes) < classes_per_client and class_idx < len(class_assignments):
            candidate_class = class_assignments[class_idx]
            if candidate_class not in assigned_classes:
                assigned_classes.append(candidate_class)
            class_idx += 1
        
        # 如果仍然没有足够的类，从剩余类中随机选择
        if len(assigned_classes) < classes_per_client:
            remaining_classes = [c for c in all_classes if c not in assigned_classes]
            needed = classes_per_client - len(assigned_classes)
            assigned_classes.extend(np.random.choice(remaining_classes, needed, replace=False))
        
        dict_classes[client_id] = sorted(assigned_classes)
    
    # 创建一个映射：每个类被分配给哪些客户端
    class_to_clients = {}
    for client_id, classes in dict_classes.items():
        for class_id in classes:
            if class_id not in class_to_clients:
                class_to_clients[class_id] = []
            class_to_clients[class_id].append(client_id)
    
    # 初始化每个客户端的数据索引列表
    for client_id in range(args.client_num):
        dict_users[client_id] = []
    
    # 对每个类使用狄利克雷分布分配数据
    for class_id in range(num_classes):
        # 获取该类的所有数据索引
        idx_k = np.where(labels == class_id)[0]
        np.random.shuffle(idx_k)
        
        # 获取拥有该类的客户端
        clients_with_class = class_to_clients.get(class_id, [])
        
        if len(clients_with_class) == 0:
            continue
            
        # 使用狄利克雷分布为这些客户端分配数据
        proportions = np.random.dirichlet(np.repeat(args.alpha, len(clients_with_class)))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # 将数据分配给相应的客户端
        idx_splits = np.split(idx_k, proportions)
        for i, client_id in enumerate(clients_with_class):
            dict_users[client_id].extend(idx_splits[i].tolist())
    
    # 为每个客户端打乱数据顺序
    for client_id in range(args.client_num):
        np.random.shuffle(dict_users[client_id])

    net_cls_counts = {}

    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))
    print('Client class assignments:')
    for client_id, classes in dict_classes.items():
        print(f'Client {client_id}: classes {classes}')

    save_path = os.path.join(os.path.dirname(__file__), '../', 'split_file')
    file_name = f'{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
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
