import copy
import os
import random
import time
from typing import List, Dict

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.utils as vutils
import numpy as np

from utils.fedDMutils import random_pertube


class ProtoDMServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: List,  # List of ProtoDMClient
        # --- model training params ---
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        # --- data condensation params ---
        dc_iterations: int,
        image_lr: float,
        rho: float,
        # --- initialization method ---
        init_method: str,
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

        self.dc_iterations = dc_iterations
        self.image_lr = image_lr
        self.rho = rho
        self.init_method = init_method

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

        self.model_identification = model_identification
        self.dataset_info = dataset_info

    def fit(self):
        evaluate_acc = 0
        for rounds in range(self.communication_rounds):
            # 评估模型并制作检查点
            print(f' ====== round {rounds} ======')
            if rounds % self.eval_gap == 0:
                acc = self.evaluate()
                print(f'round {rounds} evaluation: test acc is {acc}')
                evaluate_acc = acc

            start_time = time.time()

            print('---------- client data preparation ----------')
            
            # 选择客户端并收集合成信息
            selected_clients = self.select_clients()
            print('selected clients:', [client.cid for client in selected_clients])
            
            client_synthesis_info = []
            for client in selected_clients:
                client.receive_model(self.global_model)
                synthesis_info = client.prepare_synthesis_data()
                # 只有当客户端实际有数据时才添加到列表中
                if synthesis_info['real_features']:  # 检查是否有任何类别的特征
                    client_synthesis_info.append(synthesis_info)
                else:
                    print(f'Client {client.cid} has no valid data for synthesis, skipping...')

            print('---------- server-side data synthesis ----------')
            
            # 检查是否有有效的合成信息
            if not client_synthesis_info:
                print('No clients have valid data for synthesis, skipping this round...')
                continue
            
            # 为每个客户端分别合成数据，然后聚合
            all_synthetic_data = []
            all_synthetic_labels = []
            
            for client_info in client_synthesis_info:
                print(f'Synthesizing data for client {client_info["cid"]}...')
                synthetic_data, synthetic_labels = self.synthesize_data_for_client(client_info, rounds)
                if synthetic_data.size(0) > 0:
                    all_synthetic_data.append(synthetic_data)
                    all_synthetic_labels.append(synthetic_labels)
                else:
                    print(f'No synthetic data generated for client {client_info["cid"]}')
            
            # 检查是否有有效的合成数据
            if not all_synthetic_data:
                print('No synthetic data generated from any client, skipping model training for this round...')
                continue
            
            # 聚合所有客户端的合成数据
            synthetic_data = torch.cat(all_synthetic_data, dim=0).cpu()
            synthetic_labels = torch.cat(all_synthetic_labels, dim=0)

            print('---------- update global model ----------')

            # 使用合成数据更新模型参数
            synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=4)
            
            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=0.01,
                weight_decay=0.0005,
                momentum=0.9,
            )
            model_optimizer.zero_grad()
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

    def synthesize_data_for_client(self, client_info: Dict, round_num: int):
        """
        为单个客户端合成数据
        """
        cid = client_info['cid']
        print(f"Starting data synthesis for client {cid}...")
        
        # 检查客户端是否有有效的特征数据
        if not client_info['real_features']:
            print(f"Client {cid} has no valid features for synthesis")
            # 返回空的合成数据
            empty_data = torch.zeros((0, self.dataset_info['channel'], 
                                    self.dataset_info['im_size'][0], 
                                    self.dataset_info['im_size'][1]))
            empty_labels = torch.zeros((0,), dtype=torch.long)
            return empty_data, empty_labels
        
        # 获取客户端的类别信息
        client_classes = [c for c in client_info['classes'] if c in client_info['real_features']]
        
        if not client_classes:
            print(f"Client {cid} has no classes available for synthesis")
            empty_data = torch.zeros((0, self.dataset_info['channel'], 
                                    self.dataset_info['im_size'][0], 
                                    self.dataset_info['im_size'][1]))
            empty_labels = torch.zeros((0,), dtype=torch.long)
            return empty_data, empty_labels
        
        print(f"Client {cid} available classes for synthesis: {client_classes}")
        
        # 计算每个类别的IPC
        ipc = client_info['ipc']
        total_synthetic_images = len(client_classes) * ipc
        
        # 初始化合成图像 - 使用标准正态分布N(0,1)
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
        
        # 根据初始化方法选择不同的初始化策略
        if self.init_method == "real_sample":
            # 使用客户端提供的样本图像初始化合成图像
            for i, c in enumerate(client_classes):
                start_idx = i * ipc
                end_idx = (i + 1) * ipc
                
                if c in client_info['sample_images']:
                    sample_images = client_info['sample_images'][c]
                    num_samples = min(ipc, sample_images.size(0))
                    # 确保使用 avg=False 的方式初始化（与 fedDMclient 保持一致）
                    synthetic_images.data[start_idx:start_idx+num_samples] = sample_images[:num_samples].to(self.device)
                    # 如果样本数量不足，用标准正态分布N(0,1)填充剩余部分
                    if num_samples < ipc:
                        remaining = ipc - num_samples
                        # 使用标准正态分布N(0,1)初始化剩余的合成图像
                        remaining_shape = (remaining, self.dataset_info['channel'], 
                                         self.dataset_info['im_size'][0], self.dataset_info['im_size'][1])
                        remaining_images = torch.randn(remaining_shape, device=self.device, dtype=torch.float)
                        synthetic_images.data[start_idx+num_samples:start_idx+num_samples+remaining] = remaining_images
            print(f"Client {cid}: Using real sample initialization")
        elif self.init_method == "random":
            # 保持标准正态分布N(0,1)随机初始化，不进行任何调整
            # synthetic_images已经通过torch.randn初始化为标准正态分布N(0,1)
            print(f"Client {cid}: Using standard normal distribution N(0,1) random initialization")
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

       
        # 保存初始化后的图像
        initial_labels = []
        for c in client_classes:
            initial_labels.extend([c] * ipc)
        initial_labels = torch.tensor(initial_labels, dtype=torch.long)
        
        print(f"Saving initial images for client {cid}, round {round_num}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            initial_labels, 
            cid, 
            round_num, 
            'initial', 
            client_classes
        )

        # 设置优化器
        optimizer_image = torch.optim.SGD([synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        
        # 数据合成迭代
        for dc_iteration in range(self.dc_iterations):
            # 使用扰动模型
            sample_model = random_pertube(self.global_model, self.rho)
            sample_model.eval()
            
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i, c in enumerate(client_classes):
                start_idx = i * ipc
                end_idx = (i + 1) * ipc
                
                # 获取当前类别的合成图像
                synthetic_images_class = synthetic_images[start_idx:end_idx].reshape(
                    (ipc, self.dataset_info['channel'], 
                     self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))
                
                # 计算合成图像的特征和logits
                synthetic_features = sample_model.embed(synthetic_images_class)
                synthetic_logits = sample_model(synthetic_images_class)
                
                # 与客户端的该类别真实数据进行匹配
                if c in client_info['real_features'] and c in client_info['real_logits']:
                    # 使用真实样本特征进行更精确的匹配
                    if 'samples' in client_info['real_features'][c] and client_info['real_features'][c]['samples'].size(0) > 0:
                        # 使用实际的真实样本特征，而不只是均值
                        real_feature_samples = client_info['real_features'][c]['samples'].to(self.device)
                        real_logits_samples = client_info['real_logits'][c]['samples'].to(self.device)
                        
                        # 模拟 fedDMclient 的随机采样行为
                        if real_feature_samples.size(0) > 1:
                            # 随机选择一个子集进行匹配，模拟 real_batch_size 的效果
                            batch_size = min(real_feature_samples.size(0), max(1, real_feature_samples.size(0) // 2))
                            indices = torch.randperm(real_feature_samples.size(0))[:batch_size]
                            sampled_features = real_feature_samples[indices]
                            sampled_logits = real_logits_samples[indices]
                        else:
                            sampled_features = real_feature_samples
                            sampled_logits = real_logits_samples
                        
                        # 计算真实样本的均值
                        real_feature_mean = torch.mean(sampled_features, dim=0).detach()
                        real_logits_mean = torch.mean(sampled_logits, dim=0).detach()
                        
                        # 特征匹配损失
                        synthetic_feature_mean = torch.mean(synthetic_features, dim=0)
                        total_loss += torch.sum((real_feature_mean - synthetic_feature_mean)**2)
                        
                        # Logits匹配损失
                        synthetic_logits_mean = torch.mean(synthetic_logits, dim=0)
                        total_loss += torch.sum((real_logits_mean - synthetic_logits_mean)**2)
                        
                        # 添加特征分布匹配损失（增强数据质量）
                        if sampled_features.size(0) > 1 and synthetic_features.size(0) > 1:
                            real_feature_std = torch.std(sampled_features, dim=0)
                            synthetic_feature_std = torch.std(synthetic_features, dim=0)
                            total_loss += 0.1 * torch.sum((real_feature_std - synthetic_feature_std)**2)
                    else:
                        # 回退到预计算的均值（如果没有样本数据）
                        target_feature_mean = client_info['real_features'][c]['mean'].to(self.device)
                        synthetic_feature_mean = torch.mean(synthetic_features, dim=0)
                        total_loss += torch.sum((target_feature_mean - synthetic_feature_mean)**2)
                        
                        target_logits_mean = client_info['real_logits'][c]['mean'].to(self.device)
                        synthetic_logits_mean = torch.mean(synthetic_logits, dim=0)
                        total_loss += torch.sum((target_logits_mean - synthetic_logits_mean)**2)
            
            # 更新合成图像
            optimizer_image.zero_grad()
            total_loss.backward()
            optimizer_image.step()
            
            if dc_iteration % 100 == 0:
                print(f'Client {cid} data synthesis iteration {dc_iteration}, total loss = {total_loss.item()}')
        
        # 准备返回数据
        synthetic_labels = []
        for c in client_classes:
            synthetic_labels.extend([c] * ipc)
        
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
        
        # 保存最终合成的图像
        print(f"Saving final synthetic images for client {cid}, round {round_num}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            synthetic_labels, 
            cid, 
            round_num, 
            'final', 
            client_classes
        )
        
        print(f"Data synthesis completed for client {cid}. Generated {len(synthetic_labels)} synthetic samples for {len(client_classes)} classes.")
        
        return synthetic_images.detach().cpu(), synthetic_labels

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

    def save_synthetic_images(self, images, labels, client_id, round_num, stage, client_classes):
        """
        保存合成图像到fakedata文件夹
        
        Args:
            images: 图像tensor, shape: (N, C, H, W)
            labels: 标签tensor, shape: (N,)
            client_id: 客户端ID
            round_num: 轮次
            stage: 阶段 ('initial' 或 'final')
            client_classes: 客户端的类别列表
        """
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), '../fakedata', 
                               f'round_{round_num:03d}_client_{client_id}_{stage}')
        os.makedirs(save_dir, exist_ok=True)
        
        # 将图像数据移动到CPU并确保数值范围正确
        images = images.cpu()
        labels = labels.cpu()
        
        # 标准化图像到 [0, 1] 范围
        images_norm = images.clone()
        for i in range(images_norm.size(0)):
            img = images_norm[i]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                images_norm[i] = (img - img_min) / (img_max - img_min)
        
        # 按类别保存图像
        ipc = images.size(0) // len(client_classes) if len(client_classes) > 0 else 0
        
        for i, class_label in enumerate(client_classes):
            start_idx = i * ipc
            end_idx = min((i + 1) * ipc, images.size(0))
            
            class_images = images_norm[start_idx:end_idx]
            class_labels = labels[start_idx:end_idx]
            
            if class_images.size(0) > 0:
                # 保存单个类别的所有图像到一个文件
                class_save_path = os.path.join(save_dir, f'class_{class_label}.png')
                
                # 计算网格布局
                num_images = class_images.size(0)
                nrow = min(8, num_images)  # 每行最多8张图片
                
                # 保存图像网格
                vutils.save_image(class_images, class_save_path, 
                                nrow=nrow, normalize=False, padding=2)
                
                print(f"Saved {num_images} {stage} images for class {class_label} to {class_save_path}")
        
        # 同时保存所有类别的图像到一个大的网格中
        all_images_path = os.path.join(save_dir, 'all_classes.png')
        nrow = min(8, images_norm.size(0))
        vutils.save_image(images_norm, all_images_path, 
                         nrow=nrow, normalize=False, padding=2)
        
        print(f"Saved all {stage} images to {all_images_path}")
        
        # 保存图像和标签的统计信息
        stats_path = os.path.join(save_dir, 'stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Round: {round_num}\n")
            f.write(f"Client ID: {client_id}\n")
            f.write(f"Stage: {stage}\n")
            f.write(f"Total images: {images.size(0)}\n")
            f.write(f"Image shape: {images.shape}\n")
            f.write(f"Classes: {client_classes}\n")
            f.write(f"Images per class: {ipc}\n")
            f.write(f"Label distribution: {torch.bincount(labels).tolist()}\n")
            f.write(f"Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]\n")
