import copy
from typing import List

import torch
from tqdm import tqdm

from dataset.data.dataset import PerLabelDatasetNonIID
from utils.fedDMutils import sample_random_model, random_pertube
from utils.trueprotoDMutils import agg_func


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
    ):
        self.cid = cid

        self.train_set = train_set
        self.classes = classes
        self.dataset_info = dataset_info

        self.ipc = ipc
        self.rho = rho
        self.dc_iterations = dc_iterations
        self.real_batch_size = real_batch_size

        self.device = device
        
        # 初始化全局proto为空字典
        self.global_protos = {}

    def prepare_synthesis_data(self):
        """
        准备数据合成所需的信息，基于trueprotoDMupdate算法获取proto（原型）
        返回给服务器用于数据合成的proto信息
        """
        synthesis_info = {
            'cid': self.cid,
            'classes': self.classes,
            'ipc': self.ipc,
            'protos': {},  # 存储每个类别的proto原型
            'sample_images': {}
        }

        # 使用扰动后的模型来提取proto特征，参考trueprotoDMupdate的做法
        sample_model = random_pertube(self.global_model, self.rho)
        sample_model.eval()

        with torch.no_grad():
            # 按类别收集proto，模拟trueprotoDMupdate中的agg_protos_label过程
            agg_protos_label = {}
            
            for c in self.classes:
                # 检查该类别是否有可用样本
                available_samples = len(self.train_set.indices_class[c])
                if available_samples == 0:
                    print(f'Client {self.cid} has no samples for class {c}, skipping...')
                    continue
                    
                # 获取该类别的真实图像，参考trueprotoDMupdate中batch处理的方式
                num_samples = max(1, min(self.real_batch_size * 5, available_samples))
                real_images = self.train_set.get_images(c, num_samples)
                real_images = real_images.to(self.device)
                
                # 再次检查获取的图像是否有效
                if real_images.size(0) == 0:
                    print(f'Client {self.cid} got empty tensor for class {c}, skipping...')
                    continue
                
                # 提取proto特征，参考trueprotoDMupdate: log_probs, protos = model(images)
                # 使用train=True模式获取特征表示和预测结果
                protos, log_probs = sample_model(real_images, train=True)
                
                # 如果有全局proto，应用proto距离约束，参考trueprotoDMupdate中的loss2计算
                if len(self.global_protos) > 0 and c in self.global_protos:
                    # 参考trueprotoDMupdate中的proto距离损失处理
                    proto_new = copy.deepcopy(protos.data)
                    for i in range(protos.size(0)):
                        proto_new[i, :] = self.global_protos[c][0].data
                    
                    # 计算proto距离，但这里我们只记录，不反向传播（因为是no_grad）
                    proto_distance = torch.mean((proto_new - protos) ** 2)
                    print(f'Client {self.cid} class {c} proto distance to global: {proto_distance.item():.4f}')
                
                # 按照trueprotoDMupdate的方式收集每个样本的proto
                for i in range(protos.size(0)):
                    if c in agg_protos_label:
                        agg_protos_label[c].append(protos[i,:])
                    else:
                        agg_protos_label[c] = [protos[i,:]]
                
                # 保存一些原始图像作为初始化参考
                synthesis_info['sample_images'][c] = self.train_set.get_images(c, self.ipc, avg=False).cpu()

            # 使用agg_func聚合proto，参考trueprotoDMupdate中的处理方式
            aggregated_protos = agg_func(agg_protos_label)
            synthesis_info['protos'] = aggregated_protos

        print(f'Client {self.cid} prepared proto synthesis info for classes {list(aggregated_protos.keys())}')
        return synthesis_info

    def receive_model(self, global_model, global_protos=None):
        """接收全局模型和全局proto"""
        self.global_model = copy.deepcopy(global_model)
        self.global_model.eval()
        
        if global_protos is not None:
            self.global_protos = global_protos
            print(f'Client {self.cid} received global protos for classes: {list(global_protos.keys())}')

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
