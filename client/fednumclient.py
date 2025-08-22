import copy
from typing import List, Dict

import torch
from tqdm import tqdm

from dataset.data.dataset import PerLabelDatasetNonIID
from utils.fedDMutils import random_pertube


class FedNumClient:
    """
    FedNum算法的客户端实现
    
    FedNum是一种基于数据特征和逻辑输出统计信息的联邦学习算法，它通过以下步骤工作：
    1. 从服务器接收全局模型
    2. 使用全局模型提取本地数据的特征和逻辑输出(logits)
    3. 将数据按照avg_num进行分组，计算每组的平均特征和平均logits
    4. 将平均后的统计信息发送给服务器，而不是传输原始数据或模型参数
    
    这种方法的优势：
    - 保护数据隐私：只传输统计信息，不传输原始数据
    - 减少通信开销：传输的是压缩后的特征统计信息
    - 支持非独立同分布(Non-IID)数据：每个客户端可以有不同的数据分布
    """
    def __init__(
        self,
        cid: int,
        # --- dataset information ---
        train_set: PerLabelDatasetNonIID,
        classes: List[int],
        dataset_info: dict,
        # --- data condensation params ---
        ipc: int,  # images per class: 每个类别的合成图像数量
        rho: float,  # 模型扰动参数，用于增强鲁棒性
        avg_num: int,  # 平均组大小：将多少个样本分为一组进行平均
        device: torch.device,
    ):
        """
        初始化FedNum客户端
        
        参数说明：
        - cid: 客户端ID，用于标识不同的客户端
        - train_set: 客户端的训练数据集，按类别分组的Non-IID数据集
        - classes: 客户端拥有的数据类别列表
        - dataset_info: 数据集的元信息（通道数、图像尺寸等）
        - ipc: 每个类别的合成图像数量，用于服务器端的数据合成
        - rho: 模型随机扰动的强度，用于提高算法的鲁棒性
        - avg_num: 特征平均的组大小，控制统计信息的粒度
        - device: 计算设备（CPU或GPU）
        """
        self.cid = cid

        self.train_set = train_set
        self.classes = classes
        self.dataset_info = dataset_info

        self.ipc = ipc
        self.rho = rho
        self.avg_num = avg_num

        self.device = device

    def prepare_averaged_features_and_logits(self):
        """
        FedNum算法的核心：提取并平均化本地数据的特征和logits
        
        算法步骤：
        1. 使用全局模型对本地数据进行前向传播
        2. 提取中间层特征（embed）和最终输出（logits）
        3. 按照avg_num参数将样本分组
        4. 计算每组的平均特征和平均logits
        5. 返回所有类别的平均统计信息
        
        这种方法的关键思想：
        - 通过平均化减少数据的敏感性，保护隐私
        - 压缩数据表示，减少通信开销
        - 保留重要的统计特性，用于服务器端的合成数据生成
        
        返回值：
        - 包含每个类别的平均特征和平均logits的字典
        """
        # 步骤1: 设置模型为评估模式，确保不更新模型参数
        self.global_model.eval()

        # 初始化存储每个类别平均特征和logits的字典
        class_avg_features = {}
        class_avg_logits = {}

        # 步骤2: 在不计算梯度的情况下处理数据（节省内存，提高效率）
        with torch.no_grad():
            # 步骤3: 遍历客户端拥有的每个数据类别
            for c in self.classes:
                # 检查该类别是否有可用样本
                available_samples = len(self.train_set.indices_class[c])
                if available_samples == 0:
                    print(f'Client {self.cid} has no samples for class {c}, skipping...')
                    continue

                # 步骤4: 收集该类别的所有真实图像
                # 注意：这里获取所有样本，不进行随机采样，确保统计信息的完整性
                all_real_images = []
                indices = self.train_set.indices_class[c]
                for idx in indices:
                    # 为每个图像添加batch维度
                    all_real_images.append(self.train_set.images_all[idx].unsqueeze(0))
                
                if len(all_real_images) == 0:
                    continue
                
                # 将所有图像拼接成一个batch
                all_real_images = torch.cat(all_real_images, dim=0)
                all_real_images = all_real_images.to(self.device)

                # 步骤5: 使用全局模型提取特征和logits
                # embed(): 提取中间层特征，通常是全连接层之前的特征
                # forward(): 提取最终的logits输出
                all_features = self.global_model.embed(all_real_images)
                all_logits = self.global_model(all_real_images)

                # 步骤6: 按avg_num分组并计算平均值
                # 这是FedNum算法的关键步骤：通过分组平均来压缩数据表示
                num_samples = all_features.size(0)
                num_groups = (num_samples + self.avg_num - 1) // self.avg_num  # 向上取整

                avg_features_list = []
                avg_logits_list = []

                # 遍历每个分组
                for group_idx in range(num_groups):
                    start_idx = group_idx * self.avg_num
                    end_idx = min((group_idx + 1) * self.avg_num, num_samples)
                    
                    # 获取当前组的特征和logits
                    group_features = all_features[start_idx:end_idx]
                    group_logits = all_logits[start_idx:end_idx]
                    
                    # 计算组内平均：这是隐私保护的关键步骤
                    # 平均化操作模糊了单个样本的信息，同时保留了统计特性
                    avg_feature = torch.mean(group_features, dim=0)
                    avg_logit = torch.mean(group_logits, dim=0)
                    
                    # 将结果移到CPU以节省GPU内存
                    avg_features_list.append(avg_feature.cpu())
                    avg_logits_list.append(avg_logit.cpu())

                # 步骤7: 存储每个类别的所有平均组
                class_avg_features[c] = torch.stack(avg_features_list)
                class_avg_logits[c] = torch.stack(avg_logits_list)

                print(f'Client {self.cid} class {c}: {num_samples} samples -> {len(avg_features_list)} averaged groups')

        # 步骤8: 返回完整的统计信息
        return {
            'cid': self.cid,
            'classes': self.classes,
            'ipc': self.ipc,
            'avg_features': class_avg_features,  # 每个类别的平均特征列表
            'avg_logits': class_avg_logits        # 每个类别的平均logits列表
        }

    def train(self):
        """
        FedNum客户端的训练方法
        
        与传统联邦学习不同，FedNum不在客户端进行模型训练，
        而是提取和处理数据的统计信息。
        
        工作流程：
        1. 接收来自服务器的全局模型
        2. 使用全局模型提取本地数据的特征和logits
        3. 按组平均化这些统计信息
        4. 返回平均后的统计信息给服务器
        
        这种方法的优势：
        - 不需要在客户端进行耗时的模型训练
        - 通信的是压缩后的统计信息，而不是大型模型参数
        - 保护了原始数据的隐私
        
        返回值：
        - 包含客户端所有类别平均特征和logits的字典
        """
        return self.prepare_averaged_features_and_logits()

    def recieve_model(self, global_model):
        """
        从服务器接收全局模型
        
        参数：
        - global_model: 服务器端的全局模型
        
        操作：
        1. 深拷贝全局模型，确保不会意外修改服务器的模型
        2. 设置为评估模式，因为客户端只用于特征提取，不进行训练
        
        注意：这里使用深拷贝而不是浅拷贝，确保模型参数的完全独立
        """
        self.global_model = copy.deepcopy(global_model)
        self.global_model.eval()  # 设置为评估模式，禁用dropout和batch normalization的训练行为
