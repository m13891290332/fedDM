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
            logger.info("Global knowledge pool is empty. No data to update the global model.")
            return

        # 确保模型在正确的设备上
        self.global_model = self.global_model.to(self.device)
        self.global_model.train()
        optimizer = Adam(self.global_model.parameters(), lr=0.001)
        
        logger.info("Updating global model using data from the global knowledge pool...")
        for epoch in range(3):
            total_loss = 0.0
            for client_id, (data, labels) in self.global_knowledge_pool.items():
                # 确保数据和标签在正确的设备上
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
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}
    def fit(self, communication_rounds=10, test_loader=None):
            """主训练循环"""
            logger.info("FedMKServer training started.")
            
            # 确保有客户端列表
            if not hasattr(self, 'clients') or not self.clients:
                logger.warning("No clients available for training")
                return

            # 主训练循环
            for round in range(communication_rounds):
                # 只用logger.info打印一次
                logger.info(f"\n=== Communication Round {round + 1}/{communication_rounds} ===")
                logger.info("Broadcasting global model to clients")
                
                # 广播全局模型给所有客户端
                global_model_state = self.broadcast_model()
                
                # 准备服务端蒸馏数据
                server_distilled_data = self.broadcast_distilled_data()

                # 客户端本地蒸馏
                clients_knowledge = {}
                for client in self.clients:
                    data, labels = client.local_distillation(
                        self.global_model,
                        server_distilled_data=server_distilled_data
                    )
                    clients_knowledge[client.client_id] = (data, labels)

                # 更新全局知识池
                self.aggregate(clients_knowledge)
                self.current_round = round
                
                # 服务端更新全局模型
                self.update_global_model()

                # 评估全局模型（如果提供了测试集）
                if test_loader is not None:
                    self.global_model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            outputs = self.global_model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    logger.info(f"Global Model Test Accuracy: {accuracy:.2f}%")

                logger.info(f"Round {round + 1} completed.")

            logger.info("\nTraining completed!")

    def evaluate(model, test_loader, device):
        """评估模型在测试集上的准确率"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy