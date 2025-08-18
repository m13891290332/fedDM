import copy
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from client.fednumclient import FedNumClient
from utils.fedDMutils import random_pertube


class FedNumServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: list[FedNumClient],
        # --- model training params ---
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        # --- data condensation params ---
        ipc: int,
        rho: float,
        avg_num: int,
        batch_num: int,
        dc_iterations: int,
        image_lr: float,
        init_method: str,  # 添加初始化方法参数
        # --- test and evaluation information ---
        eval_gap: int,
        test_set: object,
        test_loader: DataLoader,
        device: torch.device,
        # --- save model and synthetic images ---
        model_identification: str,
        dataset_info: dict,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients

        self.communication_rounds = communication_rounds
        self.join_ratio = join_ratio
        self.batch_size = batch_size
        self.model_epochs = model_epochs

        self.ipc = ipc
        self.rho = rho
        self.avg_num = avg_num
        self.batch_num = batch_num
        self.dc_iterations = dc_iterations
        self.image_lr = image_lr
        self.init_method = init_method  # 保存初始化方法

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

        self.model_identification = model_identification
        self.dataset_info = dataset_info

        # 初始化全局合成数据，收集所有客户端的类别
        all_classes = set()
        for client in self.clients:
            all_classes.update(client.classes)
        self.all_classes = sorted(list(all_classes))
        
        # 初始化合成数据
        self.synthetic_images = self._initialize_synthetic_data()

    def _initialize_synthetic_data(self):
        """
        初始化合成数据，支持两种方式：随机初始化或真实样本初始化
        """
        total_synthetic_images = len(self.all_classes) * self.ipc
        
        synthetic_images = torch.randn(
            size=(
                total_synthetic_images,
                self.dataset_info['channel'],
                self.dataset_info['im_size'][0],
                self.dataset_info['im_size'][1],
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        
        # 如果选择真实样本初始化，用真实样本覆盖随机初始化的数据
        if self.init_method == 'real_sample':
            print('Initializing synthetic data with real samples...')
            self._initialize_with_real_samples(synthetic_images)
        else:
            print('Initializing synthetic data with random noise...')
        
        print(f'Initialized synthetic data with shape: {synthetic_images.shape}')
        print(f'Classes: {self.all_classes}')
        print(f'Initialization method: {self.init_method}')
        
        return synthetic_images

    def _initialize_with_real_samples(self, synthetic_images, seed_offset=0):
        """
        使用真实样本初始化合成数据，支持随机种子偏移以增加随机性
        """
        with torch.no_grad():
            for i, c in enumerate(self.all_classes):
                start_idx = i * self.ipc
                end_idx = (i + 1) * self.ipc
                
                # 从所有客户端收集该类别的真实样本
                real_samples = []
                samples_collected = 0
                
                # 为了增加随机性，打乱客户端顺序
                client_indices = list(range(len(self.clients)))
                random.seed(seed_offset * 100 + c)  # 使用不同的种子
                random.shuffle(client_indices)
                
                for client_idx in client_indices:
                    client = self.clients[client_idx]
                    if c in client.classes and samples_collected < self.ipc:
                        # 检查该客户端该类别的样本数量
                        indices = client.train_set.indices_class.get(c, [])
                        available_samples = len(indices)
                        
                        if available_samples > 0:
                            # 计算需要从该客户端获取的样本数
                            samples_needed = min(self.ipc - samples_collected, available_samples)
                            
                            if samples_needed > 0:
                                # 随机选择样本而不是总是取前面的样本
                                random.seed(seed_offset * 100 + c + client_idx)
                                selected_indices = random.sample(indices, min(samples_needed, len(indices)))
                                
                                # 使用 get_images 方法获取真实样本（仿照 FedDM 的方式）
                                if hasattr(client.train_set, 'get_images'):
                                    try:
                                        client_samples = client.train_set.get_images(c, samples_needed, avg=False)
                                        real_samples.append(client_samples)
                                        samples_collected += samples_needed
                                        print('client sample success')
                                    except:
                                        # 如果get_images失败，直接从数据集获取
                                        selected_indices = indices[:samples_needed]
                                        for idx in selected_indices:
                                            sample = client.train_set.images_all[idx].unsqueeze(0)
                                            real_samples.append(sample)
                                            samples_collected += 1
                                            if samples_collected >= self.ipc:
                                                break
                                        print('1.client sample failed')
                                else:
                                    # 直接从数据集获取样本
                                    selected_indices = indices[:samples_needed]
                                    for idx in selected_indices:
                                        sample = client.train_set.images_all[idx].unsqueeze(0)
                                        real_samples.append(sample)
                                        samples_collected += 1
                                        if samples_collected >= self.ipc:
                                            break
                                    print('2.client sample failed')
                    
                    if samples_collected >= self.ipc:
                        break
                
                # 如果收集到了真实样本，用它们初始化合成数据
                if real_samples:
                    if len(real_samples) == 1 and real_samples[0].size(0) >= self.ipc:
                        # 如果第一个样本包含足够多的图像
                        init_samples = real_samples[0][:self.ipc]
                    else:
                        # 拼接多个样本
                        init_samples = torch.cat(real_samples, dim=0)[:self.ipc]
                    
                    # 确保有足够的样本
                    if init_samples.size(0) < self.ipc:
                        # 如果样本不足，重复使用已有样本
                        repeats = (self.ipc + init_samples.size(0) - 1) // init_samples.size(0)
                        init_samples = init_samples.repeat(repeats, 1, 1, 1)[:self.ipc]
                    
                    # 复制到合成数据
                    synthetic_images.data[start_idx:end_idx] = init_samples.detach().to(self.device)
                    print(f'Class {c}: initialized with {init_samples.size(0)} real samples')
                else:
                    print(f'Class {c}: no real samples found, keeping random initialization')

    def fit(self):
        evaluate_acc = 0
        for rounds in range(self.communication_rounds):
            # evaluate the model and make checkpoint
            print(f' ====== round {rounds} ======')
            if rounds % self.eval_gap == 0:
                acc = self.evaluate()
                print(f'round {rounds} evaluation: test acc is {acc}')
                evaluate_acc = acc

            start_time = time.time()

            print('---------- client training ----------')

            # 收集客户端的平均特征和logits
            selected_clients = self.select_clients()
            print('selected clients:', [selected_client.cid for selected_client in selected_clients])
            
            client_data_list = []
            for client in selected_clients:
                client.recieve_model(self.global_model)
                client_data = client.train()  # 返回平均后的特征和logits
                client_data_list.append(client_data)

            print('---------- update synthetic data ----------')

            # 使用收集到的平均特征和logits更新合成数据
            self._update_synthetic_data(client_data_list)

            print('---------- update global model ----------')

            # 使用更新后的合成数据训练全局模型
            synthetic_data, synthetic_labels = self._get_synthetic_dataset()
            synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=4)
            
            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=0.01,
                weight_decay=0.0005,
                momentum=0.9,
            )
            loss_function = torch.nn.CrossEntropyLoss()

            total_loss = 0
            for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
                for x, target in synthetic_dataloader:
                    x, target = x.to(self.device), target.to(self.device)
                    target = target.long()
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    model_optimizer.zero_grad()
                    loss.backward()
                    model_optimizer.step()
                    total_loss += loss.item()

            round_time = time.time() - start_time
            print(f'epoch avg loss = {total_loss / self.model_epochs}, total time = {round_time}')
            wandb.log({
                "time": round_time,
                "epoch_avg_loss": total_loss / self.model_epochs,
                "evaluate_acc": evaluate_acc
            })

            save_root_path = os.path.join(os.path.dirname(__file__), '../results/')
            save_root_path = os.path.join(save_root_path, self.model_identification, 'net')
            os.makedirs(save_root_path, exist_ok=True)
            self.save_model(path=save_root_path, rounds=rounds, include_image=False)

    def _update_synthetic_data(self, client_data_list: List[Dict]):
        """
        使用客户端的平均特征和logits更新合成数据
        """
        # 如果使用真实样本初始化，每轮都重新初始化合成数据
        if self.init_method == 'real_sample':
            print('Re-initializing synthetic data with real samples for this round...')
            self._initialize_with_real_samples(self.synthetic_images)
        
        # 合并所有客户端的数据
        all_avg_features = {}
        all_avg_logits = {}
        
        for client_data in client_data_list:
            for c in client_data['avg_features']:
                if c not in all_avg_features:
                    all_avg_features[c] = []
                    all_avg_logits[c] = []
                all_avg_features[c].append(client_data['avg_features'][c])
                all_avg_logits[c].append(client_data['avg_logits'][c])

        # 将每个类别的数据合并
        for c in all_avg_features:
            all_avg_features[c] = torch.cat(all_avg_features[c], dim=0)
            all_avg_logits[c] = torch.cat(all_avg_logits[c], dim=0)

        print(f'Collected averaged features and logits for classes: {list(all_avg_features.keys())}')

        # 设置优化器
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)

        for dc_iteration in range(self.dc_iterations):
            # 使用扰动模型
            sample_model = random_pertube(self.global_model, self.rho)
            sample_model.eval()

            total_loss = torch.tensor(0.0).to(self.device)

            for i, c in enumerate(self.all_classes):
                if c not in all_avg_features:
                    continue
                
                start_idx = i * self.ipc
                end_idx = (i + 1) * self.ipc
                
                # 获取当前类别的合成图像
                synthetic_images_class = self.synthetic_images[start_idx:end_idx].reshape(
                    (self.ipc, self.dataset_info['channel'], 
                     self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))

                # 计算合成图像的特征和logits
                synthetic_features = sample_model.embed(synthetic_images_class)
                synthetic_logits = sample_model(synthetic_images_class)

                # 每次随机取同一类的batch_num/avg_num个平均特征和平均logits
                available_avg_samples = all_avg_features[c].size(0)
                sample_size = min(self.batch_num // self.avg_num, available_avg_samples)
                
                if sample_size > 0:
                    # 随机选择样本
                    indices = torch.randperm(available_avg_samples)[:sample_size]
                    target_avg_features = all_avg_features[c][indices].to(self.device)
                    target_avg_logits = all_avg_logits[c][indices].to(self.device)

                    # 计算合成数据的特征和logits均值
                    synthetic_feature_mean = torch.mean(synthetic_features, dim=0)
                    synthetic_logit_mean = torch.mean(synthetic_logits, dim=0)

                    # 计算损失：每个目标特征和logits都单独与合成数据的均值计算损失
                    feature_loss = torch.tensor(0.0).to(self.device)
                    logit_loss = torch.tensor(0.0).to(self.device)
                    
                    for j in range(sample_size):
                        target_feature = target_avg_features[j]
                        target_logit = target_avg_logits[j]
                        
                        feature_loss += torch.sum((synthetic_feature_mean - target_feature) ** 2)
                        logit_loss += torch.sum((synthetic_logit_mean - target_logit) ** 2)
                    
                    # 使用平均而不是累加，避免损失过大但保持足够的梯度信号
                    feature_loss = feature_loss / sample_size
                    logit_loss = logit_loss / sample_size
                    
                    total_loss += feature_loss + logit_loss

            # 更新合成数据
            optimizer_image.zero_grad()
            total_loss.backward()
            optimizer_image.step()

            if dc_iteration % 100 == 0:
                print(f'Synthetic data update iteration {dc_iteration}, total loss = {total_loss.item()}')

    def _get_synthetic_dataset(self):
        """
        获取合成数据集
        """
        synthetic_labels = []
        for c in self.all_classes:
            synthetic_labels.extend([c] * self.ipc)
        
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
        
        return self.synthetic_images.detach().cpu(), synthetic_labels

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        return correct / float(total)

    def make_checkpoint(self, rounds):
        checkpoint = {
            'current_round': rounds,
            'model': self.global_model.state_dict()
        }
        return checkpoint

    def save_model(self, path, rounds, include_image):
        torch.save(self.make_checkpoint(rounds), os.path.join(path, 'model.pt'))
        if include_image:
            raise NotImplementedError('not implement yet')
