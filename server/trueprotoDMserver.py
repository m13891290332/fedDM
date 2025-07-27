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

from utils.fedDMutils import random_pertube
from utils.trueprotoDMutils import proto_aggregation


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

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

        self.model_identification = model_identification
        self.dataset_info = dataset_info
        
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
                synthetic_data, synthetic_labels = self.synthesize_data_for_client(client_info)
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

    def synthesize_data_for_client(self, client_info: Dict):
        """
        为单个客户端合成数据，基于proto原型进行合成
        参考trueprotoDMupdate算法中的proto距离损失
        """
        cid = client_info['cid']
        print(f"Starting proto-based data synthesis for client {cid}...")
        
        # 检查客户端是否有有效的proto数据
        if not client_info['protos']:
            print(f"Client {cid} has no valid protos for synthesis")
            # 返回空的合成数据
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
        
        # 计算每个类别的IPC
        ipc = client_info['ipc']
        total_synthetic_images = len(client_classes) * ipc
        
        # 初始化合成图像
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
        
        # 使用客户端提供的样本图像初始化合成图像
        for i, c in enumerate(client_classes):
            start_idx = i * ipc
            end_idx = (i + 1) * ipc
            
            if c in client_info['sample_images']:
                sample_images = client_info['sample_images'][c]
                num_samples = min(ipc, sample_images.size(0))
                synthetic_images.data[start_idx:start_idx+num_samples] = sample_images[:num_samples].to(self.device)
                # 如果样本数量不足，用随机扰动填充剩余部分
                if num_samples < ipc:
                    remaining = ipc - num_samples
                    base_sample = sample_images[0:1].to(self.device) if sample_images.size(0) > 0 else synthetic_images[start_idx:start_idx+1]
                    for j in range(remaining):
                        noise = torch.randn_like(base_sample) * 0.1
                        synthetic_images.data[start_idx+num_samples+j:start_idx+num_samples+j+1] = base_sample + noise

        # 设置优化器
        optimizer_image = torch.optim.SGD([synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        
        # 数据合成迭代，基于proto距离损失
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
                
                total_loss += proto_loss / synthetic_protos.size(0)  # 平均化损失
                
                # 如果有全局proto，也添加全局约束
                if hasattr(self, 'global_protos') and len(self.global_protos) > 0 and c in self.global_protos:
                    global_target_proto = self.global_protos[c][0].to(self.device)
                    global_proto_loss = torch.tensor(0.0).to(self.device)
                    for j in range(synthetic_protos.size(0)):
                        global_proto_loss += loss_mse(synthetic_protos[j], global_target_proto)
                    
                    # 添加全局proto约束，权重较小
                    total_loss += 0.5 * global_proto_loss / synthetic_protos.size(0)
            
            # 更新合成图像
            optimizer_image.zero_grad()
            total_loss.backward()
            optimizer_image.step()
            
            if dc_iteration % 100 == 0:
                print(f'Client {cid} proto-based synthesis iteration {dc_iteration}, total loss = {total_loss.item()}')
        
        # 准备返回数据
        synthetic_labels = []
        for c in client_classes:
            synthetic_labels.extend([c] * ipc)
        
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
        
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
