import copy
from typing import List, Dict

import torch
from tqdm import tqdm

from dataset.data.dataset import PerLabelDatasetNonIID
from utils.fedDMutils import random_pertube


class FedNumClient:
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
        avg_num: int,
        device: torch.device,
    ):
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
        获取所有真实数据的特征和logits，并按avg_num进行平均
        返回每个类别的平均特征和平均logits
        """
        # 使用扰动后的模型来提取特征
        sample_model = random_pertube(self.global_model, self.rho)
        sample_model.eval()

        class_avg_features = {}
        class_avg_logits = {}

        with torch.no_grad():
            for c in self.classes:
                # 检查该类别是否有可用样本
                available_samples = len(self.train_set.indices_class[c])
                if available_samples == 0:
                    print(f'Client {self.cid} has no samples for class {c}, skipping...')
                    continue

                # 获取该类别的所有真实图像（不是随机取样本）
                all_real_images = []
                indices = self.train_set.indices_class[c]
                for idx in indices:
                    all_real_images.append(self.train_set.images_all[idx].unsqueeze(0))
                
                if len(all_real_images) == 0:
                    continue
                
                all_real_images = torch.cat(all_real_images, dim=0)
                all_real_images = all_real_images.to(self.device)

                # 提取所有样本的特征和logits
                all_features = sample_model.embed(all_real_images)
                all_logits = sample_model(all_real_images)

                # 按avg_num分组并计算平均值
                num_samples = all_features.size(0)
                num_groups = (num_samples + self.avg_num - 1) // self.avg_num  # 向上取整

                avg_features_list = []
                avg_logits_list = []

                for group_idx in range(num_groups):
                    start_idx = group_idx * self.avg_num
                    end_idx = min((group_idx + 1) * self.avg_num, num_samples)
                    
                    # 计算该组的平均特征和平均logits
                    group_features = all_features[start_idx:end_idx]
                    group_logits = all_logits[start_idx:end_idx]
                    
                    avg_feature = torch.mean(group_features, dim=0)
                    avg_logit = torch.mean(group_logits, dim=0)
                    
                    avg_features_list.append(avg_feature.cpu())
                    avg_logits_list.append(avg_logit.cpu())

                # 将每个类别的平均特征和平均logits存储
                class_avg_features[c] = torch.stack(avg_features_list)
                class_avg_logits[c] = torch.stack(avg_logits_list)

                print(f'Client {self.cid} class {c}: {num_samples} samples -> {len(avg_features_list)} averaged groups')

        return {
            'cid': self.cid,
            'classes': self.classes,
            'ipc': self.ipc,
            'avg_features': class_avg_features,
            'avg_logits': class_avg_logits
        }

    def train(self):
        """
        FedNum客户端训练方法
        返回平均后的特征和logits数据
        """
        return self.prepare_averaged_features_and_logits()

    def recieve_model(self, global_model):
        self.global_model = copy.deepcopy(global_model)
        self.global_model.eval()
