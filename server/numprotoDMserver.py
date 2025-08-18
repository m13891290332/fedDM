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

from utils.fedDMutils import random_pertube
from utils.protoDMutils import proto_aggregation


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
        batch_num: int,
        # --- test and evaluation information ---
        eval_gap: int,
        test_set: object,
        test_loader: DataLoader,
        device: torch.device,
        # --- save model and synthetic images ---
        model_identification: str,
        dataset_info: dict,
        # --- initialization method ---
        init_method: str = "real_sample",
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
        self.batch_num = batch_num

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

        self.model_identification = model_identification
        self.dataset_info = dataset_info
        self.init_method = init_method
        
        # 添加全局proto管理，参考trueprotoDMmain中的global_protos
        self.global_protos = {}

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
                client.receive_model(self.global_model, self.global_protos)
                synthesis_info = client.prepare_synthesis_data()
                # 只有当客户端实际有proto数据时才添加到列表中
                if synthesis_info['protos']:  # 检查是否有任何类别的proto
                    client_synthesis_info.append(synthesis_info)
                else:
                    print(f'Client {client.cid} has no valid proto data for synthesis, skipping...')

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
            
            # 聚合客户端的proto，参考trueprotoDMmain中的proto_aggregation
            print('---------- aggregating client protos ----------')
            local_protos_list = {}
            for idx, client_info in enumerate(client_synthesis_info):
                local_protos_list[idx] = client_info['protos']
            
            # 使用proto_aggregation函数聚合全局proto
            self.global_protos = proto_aggregation(local_protos_list)
            print(f'Global protos updated for classes: {list(self.global_protos.keys())}')

            print('---------- update global model ----------')

            # 使用合成数据更新模型参数
            synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=4)
            
            self.global_model.train()
            
            # 动态调整学习率，防止后期过拟合
            base_lr = 0.01
            lr_decay_factor = max(0.95 ** rounds, 0.001)  # 学习率随轮次衰减，最小0.001
            current_lr = base_lr * lr_decay_factor
            
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=current_lr,
                weight_decay=0.0005,
                momentum=0.9,
            )
            
            # 添加学习率调度器，进一步防止过拟合
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optimizer, T_max=self.model_epochs, eta_min=current_lr * 0.1
            )
            
            loss_function = torch.nn.CrossEntropyLoss()

            total_loss = 0
            for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
                epoch_loss = 0
                batch_count = 0
                for x, target in synthetic_dataloader:
                    x, target = x.to(self.device), target.to(self.device)
                    target = target.long()
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    model_optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                    
                    model_optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                
                total_loss += epoch_loss / batch_count if batch_count > 0 else 0
                scheduler.step()  # 更新学习率
                
                # 早停机制：如果损失过低，提前停止避免过拟合
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
                if avg_epoch_loss < 0.001 and epoch > 10:
                    print(f'Early stopping at epoch {epoch} to prevent overfitting (loss={avg_epoch_loss:.6f})')
                    break

            round_time = time.time() - start_time
            avg_loss = total_loss / self.model_epochs if self.model_epochs > 0 else 0
            print(f'epoch avg loss = {avg_loss}, total time = {round_time}, current_lr = {current_lr}')
            wandb.log({
                "time": round_time,
                "epoch_avg_loss": avg_loss,
                "evaluate_acc": evaluate_acc,
                "learning_rate": current_lr
            })

            save_root_path = os.path.join(os.path.dirname(__file__), '../results/')
            save_root_path = os.path.join(save_root_path, self.model_identification, 'net')
            os.makedirs(save_root_path, exist_ok=True)
            self.save_model(path=save_root_path, rounds=rounds, include_image=False)

    def synthesize_data_for_client(self, client_info: Dict, rounds: int):
        """
        为单个客户端合成数据，基于多个proto原型进行合成
        根据客户端提供的多个proto数量，增加合成数据数量
        """
        cid = client_info['cid']
        batch_num = client_info.get('batch_num', self.batch_num)
        print(f"Starting proto-based data synthesis for client {cid}...")
        
        # 检查客户端是否有有效的proto数据
        if not client_info['protos']:
            print(f"Client {cid} has no valid protos for synthesis")
            empty_data = torch.zeros((0, self.dataset_info['channel'], 
                                    self.dataset_info['im_size'][0], 
                                    self.dataset_info['im_size'][1]))
            empty_labels = torch.zeros((0,), dtype=torch.long)
            return empty_data, empty_labels
        
        # 获取客户端的类别信息
        client_classes = list(client_info['protos'].keys())
        
        if not client_classes:
            print(f"Client {cid} has no classes available for synthesis")
            empty_data = torch.zeros((0, self.dataset_info['channel'], 
                                    self.dataset_info['im_size'][0], 
                                    self.dataset_info['im_size'][1]))
            empty_labels = torch.zeros((0,), dtype=torch.long)
            return empty_data, empty_labels
        
        print(f"Client {cid} available classes for synthesis: {client_classes}")
        
        # 计算每个类别的合成图像数量
        # 根据每个类别的proto数量来决定合成数据数量
        ipc = client_info['ipc']
        
        # 计算每个类别的proto数量和总合成图像数量
        class_proto_counts = {}
        total_synthetic_images = 0
        
        for c in client_classes:
            # 获取该类别的proto数据
            class_protos = client_info['protos'][c]
            
            # 检查proto数据格式
            if isinstance(class_protos, list):
                proto_count = len(class_protos)
            else:
                proto_count = 1  # 单个proto的情况
            
            class_proto_counts[c] = proto_count
            
            # 每个proto对应ipc个合成图像
            class_synthetic_count = proto_count * ipc
            total_synthetic_images += class_synthetic_count
            
            print(f"Client {cid} class {c}: {proto_count} protos -> {class_synthetic_count} synthetic images")
        
        print(f"Client {cid}: Total synthetic images to generate: {total_synthetic_images}")
        
        # 初始化合成图像
        if self.init_method == "random":
            synthetic_images = torch.randn(
                size=(
                    total_synthetic_images,
                    self.dataset_info['channel'],
                    self.dataset_info['im_size'][0],
                    self.dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                device=self.device,
            ) * 0.1
            synthetic_images = torch.clamp(synthetic_images, -2.0, 2.0)
            synthetic_images = synthetic_images.detach().requires_grad_(True)
        else:
            synthetic_images = torch.randn(
                size=(
                    total_synthetic_images,
                    self.dataset_info['channel'],
                    self.dataset_info['im_size'][0],
                    self.dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                device=self.device,
            )
            synthetic_images = synthetic_images.detach().requires_grad_(True)
        
        # 根据初始化方法选择不同的初始化策略
        current_idx = 0
        if self.init_method == "real_sample":
            for c in client_classes:
                proto_count = class_proto_counts[c]
                class_synthetic_count = proto_count * ipc
                
                if c in client_info['sample_images']:
                    sample_images = client_info['sample_images'][c]
                    
                    # 为每个proto对应的图像进行初始化
                    for proto_idx in range(proto_count):
                        start_idx = current_idx + proto_idx * ipc
                        end_idx = start_idx + ipc
                        
                        # 重复使用样本图像初始化
                        if sample_images.size(0) >= ipc:
                            with torch.no_grad():
                                synthetic_images.data[start_idx:end_idx] = sample_images[:ipc].to(self.device)
                        else:
                            # 重复样本图像
                            repeated_samples = sample_images.repeat((ipc // sample_images.size(0) + 1, 1, 1, 1))[:ipc]
                            with torch.no_grad():
                                synthetic_images.data[start_idx:end_idx] = repeated_samples.to(self.device)
                else:
                    # 使用随机初始化
                    with torch.no_grad():
                        synthetic_images.data[current_idx:current_idx+class_synthetic_count] = torch.randn_like(
                            synthetic_images.data[current_idx:current_idx+class_synthetic_count])
                
                current_idx += class_synthetic_count
            
            print(f"Client {cid}: Using real sample initialization with repetition for multiple protos")
        elif self.init_method == "random":
            print(f"Client {cid}: Using improved random initialization")
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        # 保存初始化后的图像
        initial_labels = []
        current_idx = 0
        for c in client_classes:
            proto_count = class_proto_counts[c]
            class_synthetic_count = proto_count * ipc
            initial_labels.extend([c] * class_synthetic_count)
            current_idx += class_synthetic_count
        
        initial_labels = torch.tensor(initial_labels, dtype=torch.long)
        
        print(f"Saving initial proto images for client {cid}, round {rounds}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            initial_labels, 
            cid, 
            rounds, 
            'initial', 
            client_classes
        )

        # 设置优化器
        base_image_lr = self.image_lr * 0.1 if self.init_method == "random" else self.image_lr
        optimizer_image = torch.optim.SGD([synthetic_images], lr=base_image_lr, momentum=0.5, weight_decay=0)
        
        # 数据合成迭代，基于多个proto距离损失
        best_loss = float('inf')
        patience = 200 if self.init_method == "random" else 100
        stagnant_count = 0
        learning_rate_decayed = False
        
        for dc_iteration in range(self.dc_iterations):
            # 使用扰动模型
            sample_model = random_pertube(self.global_model, self.rho)
            sample_model.eval()
            
            total_loss = torch.tensor(0.0).to(self.device)
            current_idx = 0
            
            for i, c in enumerate(client_classes):
                proto_count = class_proto_counts[c]
                class_synthetic_count = proto_count * ipc
                
                # 获取该类别的所有合成图像
                class_start_idx = current_idx
                class_end_idx = current_idx + class_synthetic_count
                class_synthetic_images = synthetic_images[class_start_idx:class_end_idx]
                
                # 计算合成图像的proto特征
                syn_protos, syn_log_probs = sample_model(class_synthetic_images, train=True)
                
                # 获取该类别的目标proto列表
                target_protos = client_info['protos'][c]
                if not isinstance(target_protos, list):
                    target_protos = [target_protos]
                
                # 为每个目标proto计算对应的损失
                proto_loss = torch.tensor(0.0).to(self.device)
                
                for proto_idx, target_proto in enumerate(target_protos):
                    # 获取对应这个proto的合成图像
                    proto_start_idx = proto_idx * ipc
                    proto_end_idx = (proto_idx + 1) * ipc
                    
                    if proto_end_idx <= syn_protos.size(0):
                        proto_syn_protos = syn_protos[proto_start_idx:proto_end_idx]
                        proto_syn_logprobs = syn_log_probs[proto_start_idx:proto_end_idx]
                        
                        target_proto = target_proto.to(self.device)
                        
                        # 计算该批次合成proto与目标proto的距离损失
                        proto_avg = proto_syn_protos.mean(dim=0)
                        proto_distance_loss = torch.mean((target_proto - proto_avg) ** 2)
                        proto_loss += proto_distance_loss
                        
                        # 添加分类损失
                        class_targets = torch.full((proto_syn_protos.size(0),), c, device=self.device, dtype=torch.long)
                        class_loss = torch.nn.CrossEntropyLoss()(proto_syn_logprobs, class_targets)
                        proto_loss += 0.1 * class_loss
                
                # 平均proto损失
                if len(target_protos) > 0:
                    proto_loss = proto_loss / len(target_protos)
                
                total_loss += proto_loss
                current_idx += class_synthetic_count
                
                if dc_iteration % 100 == 0 and i == 0:
                    print(f'Iteration {dc_iteration}, Class {c}: proto_loss={proto_loss.item():.4f} (from {len(target_protos)} protos)')
            
            # 更新合成图像
            optimizer_image.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([synthetic_images], max_norm=1.0)
            
            optimizer_image.step()
            
            # 值范围约束
            if self.init_method == "random":
                with torch.no_grad():
                    synthetic_images.data = torch.clamp(synthetic_images.data, -3.0, 3.0)
            
            # 早停机制
            current_loss = total_loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                stagnant_count = 0
            else:
                stagnant_count += 1
                
            # 自适应学习率调整
            if self.init_method == "random" and not learning_rate_decayed and stagnant_count > 50 and dc_iteration > 100:
                for param_group in optimizer_image.param_groups:
                    param_group['lr'] *= 0.5
                print(f"Client {cid}: Reduced learning rate to {param_group['lr']}")
                learning_rate_decayed = True
                stagnant_count = 0
                
            # 早停条件
            if stagnant_count >= patience and dc_iteration > 300:
                print(f"Client {cid}: Early stopping at iteration {dc_iteration}")
                break
            
            if dc_iteration % 100 == 0:
                print(f'Client {cid} iteration {dc_iteration}: total_loss={current_loss:.4f}, best_loss={best_loss:.4f}')
        
        # 保存最终合成的图像
        print(f"Saving final proto synthetic images for client {cid}, round {rounds}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            initial_labels, 
            cid, 
            rounds, 
            'final', 
            client_classes
        )
        
        total_proto_count = sum(class_proto_counts.values())
        print(f"Proto-based data synthesis completed for client {cid}. Generated {len(initial_labels)} synthetic samples from {total_proto_count} protos for {len(client_classes)} classes.")
        
        return synthetic_images.detach().cpu(), initial_labels
        
        # 初始化合成图像 - 改进随机初始化策略
        if self.init_method == "random":
            # 使用更小范围的随机初始化，避免数值过大
            synthetic_images = torch.randn(
                size=(
                    total_synthetic_images,
                    self.dataset_info['channel'],
                    self.dataset_info['im_size'][0],
                    self.dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                device=self.device,
            ) * 0.1  # 缩小初始化范围到0.1倍标准正态分布
            
            # 将初始化值限制在合理范围内
            synthetic_images = torch.clamp(synthetic_images, -2.0, 2.0)
            # 确保tensor是叶子节点且requires_grad=True
            synthetic_images = synthetic_images.detach().requires_grad_(True)
        else:
            # 对于real_sample初始化，保持原来的N(0,1)范围
            synthetic_images = torch.randn(
                size=(
                    total_synthetic_images,
                    self.dataset_info['channel'],
                    self.dataset_info['im_size'][0],
                    self.dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                device=self.device,
            )
            # 确保tensor是叶子节点且requires_grad=True
            synthetic_images = synthetic_images.detach().requires_grad_(True)
        
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
                    with torch.no_grad():
                        synthetic_images.data[start_idx:start_idx+num_samples] = sample_images[:num_samples].to(self.device)
                    # 如果样本数量不足，用标准正态分布N(0,1)填充剩余部分
                    if num_samples < ipc:
                        remaining = ipc - num_samples
                        # 使用标准正态分布N(0,1)初始化剩余的合成图像
                        remaining_shape = (remaining, self.dataset_info['channel'], 
                                         self.dataset_info['im_size'][0], self.dataset_info['im_size'][1])
                        remaining_images = torch.randn(remaining_shape, device=self.device, dtype=torch.float)
                        with torch.no_grad():
                            synthetic_images.data[start_idx+num_samples:start_idx+num_samples+remaining] = remaining_images
                else:
                    # 如果没有样本图像，使用标准正态分布N(0,1)初始化作为后备
                    with torch.no_grad():
                        synthetic_images.data[start_idx:end_idx] = torch.randn_like(synthetic_images.data[start_idx:end_idx])
            print(f"Client {cid}: Using real sample initialization with N(0,1) fallback")
                    
        elif self.init_method == "random":
            # 改进的随机初始化（已在上面处理）
            print(f"Client {cid}: Using improved random initialization (0.1*N(0,1), clamped to [-2,2])")
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        # 保存初始化后的图像
        initial_labels = []
        for c in client_classes:
            initial_labels.extend([c] * ipc)
        initial_labels = torch.tensor(initial_labels, dtype=torch.long)
        
        print(f"Saving initial proto images for client {cid}, round {rounds}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            initial_labels, 
            cid, 
            rounds, 
            'initial', 
            client_classes
        )

        # 设置优化器 - 使用更小的学习率适应改进的初始化
        base_image_lr = self.image_lr * 0.1 if self.init_method == "random" else self.image_lr
        optimizer_image = torch.optim.SGD([synthetic_images], lr=base_image_lr, momentum=0.5, weight_decay=0)
        
        # 数据合成迭代，基于proto距离损失
        best_loss = float('inf')
        patience = 200 if self.init_method == "random" else 100  # 随机初始化需要更多耐心
        stagnant_count = 0
        learning_rate_decayed = False
        
        for dc_iteration in range(self.dc_iterations):
            # 使用扰动模型，参考trueprotoDMupdate
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
                
                # 计算合成图像的proto特征，参考trueprotoDMupdate: log_probs, protos = model(images)
                synthetic_protos, log_probs = sample_model(synthetic_images_class, train=True)
                
                # 获取客户端该类别的目标proto
                target_proto = client_info['protos'][c].to(self.device)
                
                # 计算proto距离损失，参考trueprotoDMupdate中的loss2计算方式
                # 模拟trueprotoDMupdate中: loss2 = loss_mse(proto_new, protos)
                loss_mse = nn.MSELoss()
                
                # 为每个合成图像计算与目标proto的距离损失
                proto_loss = torch.tensor(0.0).to(self.device)
                for j in range(synthetic_protos.size(0)):
                    proto_loss += loss_mse(synthetic_protos[j], target_proto)
                
                # 主要损失：proto距离损失，增加权重使其更强
                main_proto_loss = 5.0 * proto_loss / synthetic_protos.size(0)  # 增加权重
                total_loss += main_proto_loss
                
                # 辅助损失1：特征分布约束（模拟serverDM的特征匹配）
                if synthetic_protos.size(0) > 1:
                    # 计算合成proto的均值，与目标proto对比
                    synthetic_proto_mean = torch.mean(synthetic_protos, dim=0)
                    feature_match_loss = torch.sum((target_proto - synthetic_proto_mean)**2)
                    total_loss += 0.5 * feature_match_loss  # 辅助权重
                    
                    # 添加多样性损失，防止mode collapse
                    diversity_loss = torch.tensor(0.0).to(self.device)
                    for j1 in range(synthetic_protos.size(0)):
                        for j2 in range(j1+1, synthetic_protos.size(0)):
                            diversity_loss += torch.exp(-torch.sum((synthetic_protos[j1] - synthetic_protos[j2])**2))
                    total_loss += 0.1 * diversity_loss / (synthetic_protos.size(0) * (synthetic_protos.size(0) - 1) / 2)
                
                # 辅助损失2：图像正则化，防止生成病态图像
                image_reg_loss = 0.001 * torch.sum(synthetic_images_class**2) / synthetic_images_class.numel()
                total_loss += image_reg_loss
                
                # 动态调整全局proto约束权重，早期轮次更重视个性化，后期增强全局一致性
                if hasattr(self, 'global_protos') and len(self.global_protos) > 0 and c in self.global_protos:
                    # 全局约束权重随通信轮次增加而增加，但不会过强
                    global_weight = min(0.1 + rounds * 0.02, 0.3)  # 从0.1增加到最多0.3
                    
                    global_target_proto = self.global_protos[c][0].to(self.device)
                    global_proto_loss = torch.tensor(0.0).to(self.device)
                    for j in range(synthetic_protos.size(0)):
                        global_proto_loss += loss_mse(synthetic_protos[j], global_target_proto)
                    
                    # 添加全局proto约束，动态权重
                    total_loss += global_weight * global_proto_loss / synthetic_protos.size(0)
            
            # 更新合成图像
            optimizer_image.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_([synthetic_images], max_norm=1.0)
            
            optimizer_image.step()
            
            # 对于随机初始化，添加值范围约束
            if self.init_method == "random":
                with torch.no_grad():
                    synthetic_images.data.clamp_(-3.0, 3.0)
            
            # 早停机制和损失监控
            current_loss = total_loss.item()
            if current_loss < best_loss - 1e-6:  # 有显著改善
                best_loss = current_loss
                stagnant_count = 0
            else:
                stagnant_count += 1
                
            # 自适应学习率调整（针对随机初始化）
            if self.init_method == "random" and not learning_rate_decayed and stagnant_count > 50 and dc_iteration > 100:
                # 减少学习率，尝试更精细的优化
                for param_group in optimizer_image.param_groups:
                    param_group['lr'] *= 0.5
                learning_rate_decayed = True
                stagnant_count = 0  # 重置计数
                print(f'Client {cid} reduced learning rate to {param_group["lr"]:.6f} for better convergence')
                
            # 早停条件
            if stagnant_count >= patience and dc_iteration > 300:
                print(f'Client {cid} early stopping at iteration {dc_iteration} (best_loss={best_loss:.6f})')
                break
            
            if dc_iteration % 100 == 0:
                current_lr = optimizer_image.param_groups[0]['lr']
                print(f'Client {cid} proto-based synthesis iteration {dc_iteration}, total loss = {current_loss:.6f}, best_loss = {best_loss:.6f}, lr = {current_lr:.6f}')
        
        # 准备返回数据
        synthetic_labels = []
        for c in client_classes:
            synthetic_labels.extend([c] * ipc)
        
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
        
        # 保存最终合成的图像
        print(f"Saving final proto synthetic images for client {cid}, round {rounds}...")
        self.save_synthetic_images(
            synthetic_images.detach().clone(), 
            synthetic_labels, 
            cid, 
            rounds, 
            'final', 
            client_classes
        )
        
        print(f"Proto-based data synthesis completed for client {cid}. Generated {len(synthetic_labels)} synthetic samples for {len(client_classes)} classes.")
        
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
                               f'protoDM_round_{round_num:03d}_client_{client_id}_{stage}')
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
                
                print(f"Saved {num_images} proto {stage} images for class {class_label} to {class_save_path}")
        
        # 同时保存所有类别的图像到一个大的网格中
        all_images_path = os.path.join(save_dir, 'all_classes.png')
        nrow = min(8, images_norm.size(0))
        vutils.save_image(images_norm, all_images_path, 
                         nrow=nrow, normalize=False, padding=2)
        
        print(f"Saved all proto {stage} images to {all_images_path}")
        
        # 保存图像和标签的统计信息
        stats_path = os.path.join(save_dir, 'proto_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"ProtoDM Algorithm\n")
            f.write(f"Round: {round_num}\n")
            f.write(f"Client ID: {client_id}\n")
            f.write(f"Stage: {stage}\n")
            f.write(f"Initialization Method: {self.init_method}\n")
            f.write(f"Total images: {images.size(0)}\n")
            f.write(f"Image shape: {images.shape}\n")
            f.write(f"Classes: {client_classes}\n")
            f.write(f"Images per class (IPC): {ipc}\n")
            f.write(f"Label distribution: {torch.bincount(labels).tolist()}\n")
            f.write(f"Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]\n")
            f.write(f"Image normalized range: [{images_norm.min().item():.4f}, {images_norm.max().item():.4f}]\n")
            if stage == 'initial':
                f.write(f"Note: Images initialized using {self.init_method} method\n")
                if self.init_method == "random":
                    f.write(f"      Standard normal distribution N(0,1) used for random initialization\n")
                else:
                    f.write(f"      Real samples used with N(0,1) fallback for missing samples\n")
            else:
                f.write(f"Note: Images optimized using proto-based distance loss\n")
