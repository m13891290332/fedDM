import copy
from typing import List

import torch
from tqdm import tqdm

from dataset.data.dataset import PerLabelDatasetNonIID
from utils.fedDMutils import sample_random_model, random_pertube


class ProtoDMClient:
    def __init__(
        self,
        cid: int,
        # --- dataset information ---
        train_set: PerLabelDatasetNonIID,
        classes: List[int],
        dataset_info: dict,
        # --- data condensation params ---
        ipc: int,
        rho: float,
        dc_iterations: int,
        real_batch_size: int,
        device: torch.device,
        # --- initialization method ---
        init_method: str = "real_sample",
    ):
        self.cid = cid

        self.train_set = train_set
        self.classes = classes
        self.dataset_info = dataset_info

        self.ipc = ipc
        self.rho = rho
        self.dc_iterations = dc_iterations
        self.real_batch_size = real_batch_size
        self.init_method = init_method

        self.device = device

    def prepare_synthesis_data(self):
        """
        准备数据合成所需的信息，包括每个类别的真实数据特征和统计信息
        返回给服务器用于数据合成的信息
        """
        synthesis_info = {
            'cid': self.cid,
            'classes': self.classes,
            'ipc': self.ipc,
            'real_features': {},
            'real_logits': {},
            'class_statistics': {},
            'sample_images': {}
        }

        # 使用扰动后的模型来提取特征
        sample_model = random_pertube(self.global_model, self.rho)
        sample_model.eval()

        with torch.no_grad():
            for c in self.classes:
                # 检查该类别是否有可用样本
                available_samples = len(self.train_set.indices_class[c])
                if available_samples == 0:
                    print(f'Client {self.cid} has no samples for class {c}, skipping...')
                    continue
                    
                # 获取该类别的所有真实图像，确保至少请求1个样本
                num_samples = max(1, min(self.real_batch_size * 10, available_samples))  # 增加样本数量
                real_images = self.train_set.get_images(c, num_samples)
                real_images = real_images.to(self.device)
                
                # 再次检查获取的图像是否有效
                if real_images.size(0) == 0:
                    print(f'Client {self.cid} got empty tensor for class {c}, skipping...')
                    continue
                
                # 提取特征和logits
                real_features = sample_model.embed(real_images)
                real_logits = sample_model(real_images)
                
                # 计算统计信息
                synthesis_info['real_features'][c] = {
                    'mean': torch.mean(real_features, dim=0).cpu(),
                    'std': torch.std(real_features, dim=0).cpu(),
                    'samples': real_features[:min(self.real_batch_size * 2, real_features.size(0))].cpu()  # 保存更多样本特征
                }
                
                synthesis_info['real_logits'][c] = {
                    'mean': torch.mean(real_logits, dim=0).cpu(),
                    'std': torch.std(real_logits, dim=0).cpu(),
                    'samples': real_logits[:min(self.real_batch_size * 2, real_logits.size(0))].cpu()  # 保存更多样本logits
                }
                
                # 根据初始化方法决定是否保存原始图像
                if self.init_method == "real_sample":
                    # 保存一些原始图像作为初始化参考（模拟 avg=False 行为）
                    synthesis_info['sample_images'][c] = self.train_set.get_images(c, self.ipc, avg=False).cpu()
                elif self.init_method == "random":
                    # 随机初始化模式下不需要发送样本图像，但仍需要一个占位符
                    synthesis_info['sample_images'][c] = torch.empty(0, self.dataset_info['channel'], 
                                                                   self.dataset_info['im_size'][0], 
                                                                   self.dataset_info['im_size'][1])
                
                # 类别统计信息（对随机初始化有用）
                synthesis_info['class_statistics'][c] = {
                    'num_samples': len(self.train_set.indices_class[c]),
                    'data_mean': torch.mean(real_images, dim=[0, 2, 3]).cpu(),
                    'data_std': torch.std(real_images, dim=[0, 2, 3]).cpu()
                }

        print(f'Client {self.cid} prepared synthesis info for classes {self.classes}')
        return synthesis_info

    def receive_model(self, global_model):
        """接收全局模型"""
        self.global_model = copy.deepcopy(global_model)
        self.global_model.eval()

    def compute_gradients(self):
        """
        计算梯度信息，用于服务器端的数据合成
        这个方法可以用来提供额外的梯度信息给服务器
        """
        gradient_info = {
            'cid': self.cid,
            'classes': self.classes,
            'gradients': {}
        }

        sample_model = random_pertube(self.global_model, self.rho)
        sample_model.train()

        for c in self.classes:
            real_images = self.train_set.get_images(c, self.real_batch_size)
            real_images = real_images.to(self.device)
            targets = torch.full((real_images.size(0),), c, device=self.device, dtype=torch.long)
            
            # 计算损失和梯度
            sample_model.zero_grad()
            outputs = sample_model(real_images)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            # 收集梯度信息
            class_gradients = []
            for param in sample_model.parameters():
                if param.grad is not None:
                    class_gradients.append(param.grad.clone().cpu())
            
            gradient_info['gradients'][c] = class_gradients

        return gradient_info
