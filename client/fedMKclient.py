import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.utils as vutils
import os
import numpy as np
from log.logger import logger

class FedMKClient:
    def __init__(self, client_id, local_data, num_classes=10, device='cuda', 
                round_num=0, personal_hist_data=None, knowledge_pool=None, dataset_info=None):
        self.client_id = client_id
        self.local_data = local_data
        self.num_classes = num_classes
        self.device = device
        self.round_num = round_num
        self.personal_hist_data = personal_hist_data
        self.knowledge_pool = knowledge_pool  # 新增
        self.distilled_data = None
        self.distilled_labels = None
        self.tau = 5.0
        self.dataset_info = dataset_info  # 新增，用于存储数据集信息

    def _init_distilled_data(self):
        """支持条件初始化：第一轮从本地数据，后续从其他客户端元知识"""
        if self.round_num == 0:
            # 第一轮：从本地数据初始化
            num_per_class = 100
            total_samples = num_per_class * self.num_classes
            loader = DataLoader(self.local_data, batch_size=total_samples, shuffle=True)
            real_data, real_labels = next(iter(loader))
            self.distilled_data = real_data.to(self.device).requires_grad_(True)  # 确保是叶子节点
            self.distilled_data.data += torch.randn_like(self.distilled_data) * 0.01  # 减小噪声幅度
            self.distilled_labels = real_labels.to(self.device)
            logger.info(f"Client {self.client_id}: Initialized distilled data from local dataset")
        else:
            # 第二轮及以后：从其他客户端元知识中随机选择一个作为初始化
            if self.knowledge_pool is not None and len(self.knowledge_pool) > 0:
                other_clients = [k for k in self.knowledge_pool.keys() if k != self.client_id]
                if len(other_clients) > 0:
                    selected_client = np.random.choice(other_clients)
                    self.distilled_data = self.knowledge_pool[selected_client][0].clone().to(self.device).requires_grad_(True)  # 确保是叶子节点
                    self.distilled_labels = self.knowledge_pool[selected_client][1].clone().to(self.device)
                    self.distilled_data.data += torch.randn_like(self.distilled_data) * 0.001  # 更小的噪声
                    logger.info(f"Client {self.client_id}: Initialized distilled data from client {selected_client}")
                else:
                    self._init_from_local()
            else:
                self._init_from_local()

    def _init_from_local(self):
        """回退到本地数据初始化"""
        num_per_class = 100
        total_samples = num_per_class * self.num_classes
        loader = DataLoader(self.local_data, batch_size=total_samples, shuffle=True)
        real_data, real_labels = next(iter(loader))
        self.distilled_data = real_data.to(self.device).requires_grad_(True)  # 确保是叶子节点
        self.distilled_labels = real_labels.to(self.device)
        # logger.warning(f"Client {self.client_id}: Fallback to local initialization (no other clients available)")

    def dynamic_weight(self, loss):
        """动态权重计算"""
        return 1 / (1 + torch.exp(-self.tau * loss.detach()))
   
    def local_distillation(self, global_model, server_distilled_data=None):
        if self.round_num >= 2 and self.personal_hist_data is not None:
            # 第三轮及以后：强制使用个人历史数据
            self.distilled_data = self.personal_hist_data[0].clone().to(self.device).requires_grad_(True)  # 确保是叶子节点
            self.distilled_labels = self.personal_hist_data[1].clone().to(self.device)
            logger.info(f"Client {self.client_id}: Using personal distilled data (Round {self.round_num})")
        elif server_distilled_data is not None:
            # 第一轮和第二轮：使用服务端数据
            if self.client_id in server_distilled_data['distilled_data']:
                self.distilled_data = server_distilled_data['distilled_data'][self.client_id].to(self.device).requires_grad_(True)  # 确保是叶子节点
                self.distilled_labels = server_distilled_data['distilled_labels'][self.client_id].to(self.device)
                logger.info(f"Client {self.client_id}: Using server-distilled data")
            else:
                logger.warning(f"Client {self.client_id}: No server-distilled data available for this client. Falling back to local initialization.")
                self._init_distilled_data()
        else:
            # 回退到本地初始化
            self._init_distilled_data()

        # 保存原始模型参数
        original_state = {k: v.clone() for k, v in global_model.state_dict().items()}

        # === 内层优化 ===
        # 根据全局模型类型创建新实例
        model_class = type(global_model)
        if model_class.__name__ == 'ConvNet':
            inner_model = model_class(
                channel=self.dataset_info['channel'],
                num_classes=self.num_classes,
                net_width=128,
                net_depth=3,
                net_act='relu',
                net_norm='instancenorm',
                net_pooling='avgpooling',
                im_size=self.dataset_info['im_size']
            ).to(self.device)
        elif model_class.__name__ == 'ResNet18':
            inner_model = model_class(
                channel=self.dataset_info['channel'],
                num_classes=self.num_classes
            ).to(self.device)
        elif model_class.__name__ == 'LeNet':
            inner_model = model_class(
                num_classes=self.num_classes
            ).to(self.device)
        else:
            raise NotImplementedError(f"Unsupported model type: {model_class.__name__}")
            
        inner_model.load_state_dict(global_model.state_dict())
        inner_optimizer = Adam(inner_model.parameters(), lr=0.1)

        for _ in range(10):
            outputs = inner_model(self.distilled_data)
            loss_inner = F.cross_entropy(outputs, self.distilled_labels)
            inner_optimizer.zero_grad()
            loss_inner.backward()
            inner_optimizer.step()
        logger.info(f"Client {self.client_id} inner_Loss: {loss_inner:.4f}")

        # === 外层优化 ===
        # 重新创建 distilled_data 为叶子节点
        distilled_data_clone = self.distilled_data.detach().clone().requires_grad_(True)
        outer_optimizer = Adam([distilled_data_clone], lr=0.001)
        loader = DataLoader(self.local_data, batch_size=64, shuffle=True)

        for _ in range(10):
            outer_optimizer.zero_grad()
            total_loss = 0.0

            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = inner_model(data)
                losses = F.cross_entropy(outputs, labels, reduction='none')
                weights = self.dynamic_weight(losses)
                (torch.mean(weights * losses) + 1e-4 * torch.norm(distilled_data_clone)).backward()
                total_loss += torch.mean(weights * losses).item()
            outer_optimizer.step()
        logger.info(f"Client {self.client_id} Outer Loss: {total_loss:.4f}")

        # 恢复全局模型参数
        global_model.load_state_dict(original_state)
        return distilled_data_clone.detach(), self.distilled_labels.detach()