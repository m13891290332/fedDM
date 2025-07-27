# This is an implementation of FedDM:Iterative Distribution Matching
# for Communication-Efficient Federated Learning (without Differential Privacy)
import copy
import json
import os
import sys
import logging
from datetime import datetime

import torch
import wandb
from torch.utils.data import Subset

from client.fedDMclient import Client
from client.fedprotoDMclient import ProtoDMClient
from client.trueprotoDMclient import ProtoDMClient as TrueProtoDMClient
from server.fedDMserver import Server
from server.fedprotoDMserver import ProtoDMServer
from server.trueprotoDMserver import ProtoDMServer as TrueProtoDMServer
from config import parser
from dataset.data.dataset import get_dataset, PerLabelDatasetNonIID
from models.fedDMmodels import ResNet18, ConvNet
from utils.fedDMutils import setup_seed

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0 , 1 , 2 , 3"
    args = parser.parse_args()
    
    # 设置日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{current_time}_{args.algorithm}_{args.dataset}_alpha{args.alpha}_{args.client_num}clients_{args.partition_method}partition_{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_cr{args.communication_rounds}_init{args.init_method}.log"
    log_dir = "/home/MaCS/fedDM/log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 重定向print输出到日志
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass
    
    # 将stdout和stderr重定向到日志
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    
    logging.info(f"Log file: {log_path}")
    logging.info(f"Arguments: {vars(args)}")
    
    # 根据划分方式选择对应的split文件
    if args.partition_method == 'part':
        split_file = f'/{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
    elif args.partition_method == 'only':
        split_file = f'/{args.dataset}_only_dirichlet_client_num={args.client_num}_alpha={args.alpha}.json'
    else:
        raise ValueError(f"Unknown partition method: {args.partition_method}")
    
    args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
    
    # 检查split文件是否存在，如果不存在则自动生成
    if not os.path.exists(args.split_file):
        logging.info(f"Split file not found: {args.split_file}")
        logging.info(f"Generating data partition using method: {args.partition_method}")
        
        # 动态导入并运行相应的划分脚本
        if args.partition_method == 'part':
            from dataset.data.part_dataset_partition import partition
        elif args.partition_method == 'only':
            from dataset.data.only_dataset_partition import partition
        
        # 创建划分参数
        partition_args = type('Args', (), {
            'dataset': args.dataset,
            'client_num': args.client_num,
            'alpha': args.alpha,
            'dataset_root': args.dataset_root,
            'seed': args.seed
        })()
        
        # 执行数据划分
        partition(partition_args)
        logging.info(f"Data partition completed. File saved to: {args.split_file}")

    # set seeds and parse args, init wandb
    mode = "disabled" if args.debug else "offline"
    wandb.init(
        project=f'{args.algorithm}_{args.dataset}_client{args.client_num}_{args.alpha}_{args.partition_method}',
        name=f'{args.model}_cr{args.communication_rounds}_join_ratio{args.join_ratio}',
        mode=mode,
    )
    wandb.config.update(args)
    setup_seed(args.seed)
    device = torch.device(args.device)

    # get dataset and init models
    dataset_info, train_set, test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    with open(args.split_file, 'r') as file:
        file_data = json.load(file)
    client_indices, client_classes = file_data['client_idx'], file_data['client_classes']

    # 检查数据分配情况
    logging.info("Checking data allocation...")
    empty_clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            empty_clients.append(i)
        logging.info(f"Client {i}: {len(indices)} samples, classes: {client_classes[i]}")
    
    if empty_clients:
        logging.warning(f"Found empty clients: {empty_clients}")
        logging.info("Filtering out empty clients to continue training...")
        
        # 过滤掉空客户端
        valid_clients = [i for i in range(args.client_num) if i not in empty_clients]
        client_indices = [client_indices[i] for i in valid_clients]
        client_classes = [client_classes[i] for i in valid_clients]
        
        # 更新客户端数量
        args.client_num = len(valid_clients)
        logging.info(f"Updated client_num to {args.client_num} (excluded {len(empty_clients)} empty clients)")

    train_sets = [Subset(train_set, indices) for indices in client_indices]

    if args.model == "ConvNet":
        global_model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif args.model == "ResNet":
        global_model = ResNet18(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    else:
        raise NotImplemented("only support ConvNet and ResNet")

    # init server and clients
    model_identification = f'{args.algorithm}_{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs'
    
    if args.algorithm == "fedDM":
        # 原版fedDM实现
        client_list = [Client(
            cid=i,  # 新的连续ID
            train_set=PerLabelDatasetNonIID(
                train_sets[i],
                client_classes[i],
                dataset_info['channel'],
                device,
            ),
            classes=client_classes[i],
            dataset_info=dataset_info,
            ipc=args.ipc,
            rho=args.rho,
            dc_iterations=args.dc_iterations,
            real_batch_size=args.dc_batch_size,
            image_lr=args.image_lr,
            device=device,
        ) for i in range(args.client_num)]

        server = Server(
            global_model=global_model,
            clients=client_list,
            communication_rounds=args.communication_rounds,
            join_ratio=args.join_ratio,
            batch_size=args.batch_size,
            model_epochs=args.model_epochs,
            eval_gap=args.eval_gap,
            test_set=test_set,
            test_loader=test_loader,
            device=device,
            model_identification=model_identification,
        )
        
    elif args.algorithm == "fedprotoDM":
        # fedprotoDM实现
        client_list = [ProtoDMClient(
            cid=i,  # 新的连续ID
            train_set=PerLabelDatasetNonIID(
                train_sets[i],
                client_classes[i],
                dataset_info['channel'],
                device,
            ),
            classes=client_classes[i],
            dataset_info=dataset_info,
            ipc=args.ipc,
            rho=args.rho,
            dc_iterations=args.dc_iterations,
            real_batch_size=args.dc_batch_size,
            device=device,
            init_method=args.init_method,
        ) for i in range(args.client_num)]

        server = ProtoDMServer(
            global_model=global_model,
            clients=client_list,
            communication_rounds=args.communication_rounds,
            join_ratio=args.join_ratio,
            batch_size=args.batch_size,
            model_epochs=args.model_epochs,
            dc_iterations=args.dc_iterations,
            image_lr=args.image_lr,
            rho=args.rho,
            init_method=args.init_method,
            eval_gap=args.eval_gap,
            test_set=test_set,
            test_loader=test_loader,
            device=device,
            model_identification=model_identification,
            dataset_info=dataset_info,
        )
    elif args.algorithm == "trueprotoDM":
        # trueprotoDM实现
        client_list = [TrueProtoDMClient(
            cid=i,  # 新的连续ID
            train_set=PerLabelDatasetNonIID(
                train_sets[i],
                client_classes[i],
                dataset_info['channel'],
                device,
            ),
            classes=client_classes[i],
            dataset_info=dataset_info,
            ipc=args.ipc,
            rho=args.rho,
            dc_iterations=args.dc_iterations,
            real_batch_size=args.dc_batch_size,
            device=device,
        ) for i in range(args.client_num)]

        server = TrueProtoDMServer(
            global_model=global_model,
            clients=client_list,
            communication_rounds=args.communication_rounds,
            join_ratio=args.join_ratio,
            batch_size=args.batch_size,
            model_epochs=args.model_epochs,
            dc_iterations=args.dc_iterations,
            image_lr=args.image_lr,
            rho=args.rho,
            eval_gap=args.eval_gap,
            test_set=test_set,
            test_loader=test_loader,
            device=device,
            model_identification=model_identification,
            dataset_info=dataset_info,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    print(f'Server and Clients have been created for {args.algorithm}.')

    # fit the model
    logging.info(f"Starting {args.algorithm} model training...")
    server.fit()
    logging.info(f"{args.algorithm} model training completed.")

if __name__ == "__main__":
    main()