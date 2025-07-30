import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

class SimpleCNN(nn.Module):
    """基础CNN模型（适用于CIFAR-10）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # CIFAR-10经过两次池化后尺寸为8x8
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_param(self):
        """获取所有可训练参数（用于双层次优化）"""
        return list(self.parameters())

    def forward_with_param(self, x, params):
        """手动前向传播（支持外部传入参数）"""
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b = params
        x = F.conv2d(x, conv1_w, conv1_b, stride=1, padding=1)
        x = self.pool(F.relu(x))
        x = F.conv2d(x, conv2_w, conv2_b, stride=1, padding=1)
        x = self.pool(F.relu(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.linear(x, fc1_w, fc1_b)
        x = F.relu(x)
        x = F.linear(x, fc2_w, fc2_b)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))       # Layer 1
        x = self.pool(F.relu(self.conv2(x)))       # Layer 2
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))                    # Layer 3
        x = F.relu(self.fc2(x))                    # Layer 4
        x = self.fc3(x)
        return x
    def get_param(self):
        """获取所有参数（用于双层次优化）"""
        params = []
        for name, param in self.named_parameters():
            params.append(param)
        return params
    def get_intermediate_features(self, x, layer_idx):
        """获取指定层的特征（用于特征匹配）"""
        feat = self.pool(F.relu(self.conv1(x)))
        if layer_idx == 1:
            return feat
        feat = self.pool(F.relu(self.conv2(feat)))
        if layer_idx == 2:
            return feat
        feat = feat.view(-1, 16*5*5)
        feat = F.relu(self.fc1(feat))
        if layer_idx == 3:
            return feat
        raise ValueError("Invalid layer index")

    def forward_with_param(self, x, params):
        """手动前向传播"""
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
        
        x = F.conv2d(x, conv1_w, conv1_b)
        x = self.pool(F.relu(x))
        x = F.conv2d(x, conv2_w, conv2_b)
        x = self.pool(F.relu(x))
        x = x.view(-1, 16*5*5)
        x = F.linear(x, fc1_w, fc1_b)
        x = F.relu(x)
        x = F.linear(x, fc2_w, fc2_b)
        x = F.relu(x)
        x = F.linear(x, fc3_w, fc3_b)
        return x


# 兼容原有代码的快捷函数

def ResNet18(num_classes=10):
    raise NotImplementedError("本文件已替换为LeNet，如需ResNet请使用原model.py")

def ResNet34(num_classes=10):
    raise NotImplementedError("本文件已替换为LeNet，如需ResNet请使用原model.py")