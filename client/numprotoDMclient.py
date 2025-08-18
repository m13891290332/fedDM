import copy
from typing import List

import torch
from tqdm import tqdm

from dataset.data.dataset import PerLabelDatasetNonIID
from utils.fedDMutils import sample_random_model, random_pertube
from utils.protoDMutils import agg_func


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
        batch_num: int,
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
        self.batch_num = batch_num

        self.device = device
        
        # 初始化全局proto为空字典
        self.global_protos = {}

    def prepare_synthesis_data(self):
        """
        准备数据合成所需的信息，按照batch_num个样本为每个类别收集多个proto
        每个proto由batch_num个样本计算得出，剩余样本不足batch_num时单独组成一个proto
        """
        synthesis_info = {
            'cid': self.cid,
            'classes': self.classes,
            'ipc': self.ipc,
            'batch_num': self.batch_num,
            'protos': {},  # 存储每个类别的多个proto
            'sample_images': {}
        }

        # 使用扰动后的模型来提取proto特征
        sample_model = random_pertube(self.global_model, self.rho)
        sample_model.eval()

        with torch.no_grad():
            # 按类别收集多个proto
            agg_protos_label = {}
            
            for c in self.classes:
                # 检查该类别是否有可用样本
                available_samples = len(self.train_set.indices_class[c])
                if available_samples == 0:
                    print(f'Client {self.cid} has no samples for class {c}, skipping...')
                    continue
                
                print(f'Client {self.cid} collecting protos for class {c} (available samples: {available_samples}, batch_num: {self.batch_num})')
                
                # 获取该类别的所有真实图像
                real_images = self.train_set.get_images(c, available_samples)
                real_images = real_images.to(self.device)
                
                if real_images.size(0) == 0:
                    print(f'Client {self.cid} got empty tensor for class {c}, skipping...')
                    continue
                
                # 按照batch_num个样本为一组，计算多个proto
                class_protos = []
                num_full_batches = real_images.size(0) // self.batch_num
                
                # 处理完整的batch_num大小的批次
                for batch_idx in range(num_full_batches):
                    start_idx = batch_idx * self.batch_num
                    end_idx = start_idx + self.batch_num
                    batch_images = real_images[start_idx:end_idx]
                    
                    # 提取该批次的proto特征
                    protos, log_probs = sample_model(batch_images, train=True)
                    
                    # 计算该批次的平均proto
                    batch_avg_proto = protos.mean(dim=0)
                    class_protos.append(batch_avg_proto)
                    
                    print(f'Client {self.cid} class {c} batch {batch_idx}: computed proto from {self.batch_num} samples')
                
                # 处理剩余样本（不足batch_num个）
                remaining_samples = real_images.size(0) % self.batch_num
                if remaining_samples > 0:
                    start_idx = num_full_batches * self.batch_num
                    remaining_images = real_images[start_idx:]
                    
                    # 提取剩余样本的proto特征
                    remaining_protos, remaining_log_probs = sample_model(remaining_images, train=True)
                    
                    # 计算剩余样本的平均proto
                    remaining_avg_proto = remaining_protos.mean(dim=0)
                    class_protos.append(remaining_avg_proto)
                    
                    print(f'Client {self.cid} class {c} remaining batch: computed proto from {remaining_samples} samples')
                
                if class_protos:
                    # 存储该类别的多个proto
                    agg_protos_label[c] = class_protos
                    
                    print(f'Client {self.cid} class {c}: collected {len(class_protos)} protos total')
                    
                    # 如果有全局proto，计算距离用于监控
                    if len(self.global_protos) > 0 and c in self.global_protos:
                        # 计算所有local proto与全局proto的平均距离
                        total_distance = 0.0
                        for local_proto in class_protos:
                            proto_distance = torch.mean((self.global_protos[c][0] - local_proto) ** 2)
                            total_distance += proto_distance.item()
                        avg_distance = total_distance / len(class_protos)
                        print(f'Client {self.cid} class {c} average proto distance to global: {avg_distance:.4f}')
                
                # 保存一些原始图像作为初始化参考
                if available_samples >= self.ipc:
                    synthesis_info['sample_images'][c] = self.train_set.get_images(c, self.ipc, avg=False).cpu()
                else:
                    # 如果样本不足，使用所有可用样本并重复到足够数量
                    available_imgs = self.train_set.get_images(c, available_samples, avg=False).cpu()
                    repeated_imgs = available_imgs.repeat((self.ipc // available_samples + 1, 1, 1, 1))[:self.ipc]
                    synthesis_info['sample_images'][c] = repeated_imgs

            # 使用agg_func聚合proto（现在处理的是多个proto的列表）
            aggregated_protos = agg_func(agg_protos_label)
            synthesis_info['protos'] = aggregated_protos

        total_protos = sum(len(protos) if isinstance(protos, list) else 1 for protos in aggregated_protos.values())
        print(f'Client {self.cid} prepared synthesis info for classes {list(aggregated_protos.keys())} with total {total_protos} protos')
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
