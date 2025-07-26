import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import defaultdict
import numpy as np
import torchvision.utils as vutils
import os
from log.logger import logger

class FedMKServer:
    def __init__(self, global_model, num_classes=10, device='cuda'):
        self.global_model = global_model
        self.device = device
        self.num_classes = num_classes
        self.global_knowledge_pool = {}  # 全局知识池
        self.current_round = 0
        logger.info("Server initialized")
        
    def aggregate(self, clients_knowledge):
        """聚合客户端上传的蒸馏数据，并更新全局知识池"""
        self.global_knowledge_pool.update(clients_knowledge)  # 更新知识池
        logger.info("Global knowledge pool updated")
        return self.global_knowledge_pool

    def get_knowledge_pool(self):
        """获取全局知识池"""
        return self.global_knowledge_pool

    def update_global_model(self):
        """使用全局知识池中的蒸馏数据更新全局模型"""
        if not self.global_knowledge_pool:
            logger.warning("Global knowledge pool is empty. No data to update the global model.")
            return

        self.global_model.train()
        optimizer = Adam(self.global_model.parameters(), lr=0.001)
        
        logger.info("Updating global model using data from the global knowledge pool...")
        for epoch in range(3):
            total_loss = 0.0
            for client_id, (data, labels) in self.global_knowledge_pool.items():
                data = data.to(self.device).requires_grad_()
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.global_model(data)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Server Epoch {epoch+1}: Loss = {total_loss:.4f}")

    def broadcast_distilled_data(self):
        """每轮广播全局知识池中的蒸馏数据"""
        if not self.global_knowledge_pool:
            logger.info("Global knowledge pool is empty. No data to broadcast.")
            return None

        return {
            'distilled_data': {client_id: data.clone().detach() for client_id, (data, labels) in self.global_knowledge_pool.items()},
            'distilled_labels': {client_id: labels.clone().detach() for client_id, (data, labels) in self.global_knowledge_pool.items()}
        }

    def broadcast_model(self):
        """广播全局模型给所有客户端"""
        logger.info("Broadcasting global model to clients")
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}