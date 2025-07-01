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
from server.fedDMserver import Server
from config import parser
from dataset.data.dataset import get_dataset, PerLabelDatasetNonIID
from models.fedDMmodels import ResNet18, ConvNet
from utils.fedDMutils import setup_seed

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    args = parser.parse_args()
    
    # 设置日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{current_time}_{args.dataset}_alpha{args.alpha}_{args.client_num}clients_{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_cr{args.communication_rounds}.log"
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
    
    split_file = f'/{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
    args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)

    # set seeds and parse args, init wandb
    mode = "disabled" if args.debug else "offline"
    wandb.init(
        project=f'FedDM_{args.dataset}_client{args.client_num}_{args.alpha}',
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
    client_list = [Client(
        cid=i,
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

    model_identification = f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs'

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
    print('Server and Clients have been created.')

    # fit the model
    logging.info("Starting model training...")
    server.fit()
    logging.info("Model training completed.")

if __name__ == "__main__":
    main()