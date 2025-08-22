import copy
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from client.fednumclient import FedNumClient
from utils.fedDMutils import random_pertube


class FedNumServer:
    """
    FedNum算法的服务器端实现
    
    FedNum服务器的主要职责：
    1. 维护全局模型和全局合成数据集
    2. 协调客户端参与训练过程
    3. 收集客户端的统计信息（平均特征和logits）
    4. 使用收集到的统计信息更新合成数据
    5. 使用合成数据训练全局模型
    
    算法流程：
    - 初始化阶段：创建合成数据集，可选择随机初始化或真实样本初始化
    - 训练循环：
      1. 选择参与的客户端
      2. 发送全局模型给客户端
      3. 收集客户端返回的统计信息
      4. 使用统计信息通过梯度匹配更新合成数据（使用全局模型保持一致性）
      5. 使用更新后的合成数据训练全局模型
      6. 评估模型性能
    
    这种方法结合了数据蒸馏和联邦学习的优点：
    - 数据隐私保护：只使用统计信息，不传输原始数据
    - 通信效率：传输压缩的特征统计信息而不是模型参数
    - 数据质量：通过梯度匹配生成高质量的合成数据
    - 模型一致性：使用全局模型确保客户端和服务器端的特征提取完全一致
    """
    def __init__(
        self,
        global_model: nn.Module,
        clients: list[FedNumClient],
        # --- model training params ---
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        # --- data condensation params ---
        ipc: int,
        rho: float,
        avg_num: int,
        batch_num: int,
        dc_iterations: int,
        image_lr: float,
        init_method: str,  # 添加初始化方法参数
        # --- test and evaluation information ---
        eval_gap: int,
        test_set: object,
        test_loader: DataLoader,
        device: torch.device,
        # --- save model and synthetic images ---
        model_identification: str,
        dataset_info: dict,
        # --- loss configuration ---
        logit_loss_type: str = 'wasserstein',  # logits损失类型: 'wasserstein', 'kl', 'l2'
        loss_computation: str = 'avg-loss',  # 损失计算方式: 'avg-loss', 'loss-avg'
    ):
        # FedNum算法的核心组件初始化
        # 1. 全局模型：用于协调客户端和生成合成数据的共享模型
        self.global_model = global_model.to(device)
        self.clients = clients

        # 2. 联邦学习训练参数
        self.communication_rounds = communication_rounds  # 联邦学习的通信轮次
        self.join_ratio = join_ratio  # 每轮参与训练的客户端比例
        self.batch_size = batch_size  # 全局模型训练的批量大小
        self.model_epochs = model_epochs  # 每轮全局模型训练的轮次

        # 3. 数据凝聚(Data Condensation)参数
        self.ipc = ipc  # Images Per Class: 每个类别的合成图像数量
        self.rho = rho  # 模型扰动强度：保留参数但当前版本不使用（原用于随机扰动）
        self.avg_num = avg_num  # 客户端特征平均组大小
        self.batch_num = batch_num  # 每次更新合成数据时使用的批量大小
        self.dc_iterations = dc_iterations  # 数据凝聚的迭代次数
        self.image_lr = image_lr  # 合成图像的学习率
        self.init_method = init_method  # 合成数据初始化方法('random', 'real_sample', 'dm')
        self.logit_loss_type = logit_loss_type  # logits损失类型('wasserstein', 'kl', 'l2')
        self.loss_computation = loss_computation  # 损失计算方式('avg-loss' 或 'loss-avg')

        # 4. 模型评估和测试相关参数
        self.eval_gap = eval_gap  # 模型评估的间隔轮次
        self.test_set = test_set  # 测试数据集
        self.test_loader = test_loader  # 测试数据加载器
        self.device = device  # 计算设备

        # 5. 模型保存和标识参数
        self.model_identification = model_identification  # 模型标识符，用于保存路径
        self.dataset_info = dataset_info  # 数据集元信息（图像尺寸、通道数等）

        # 6. 初始化全局合成数据集
        # 收集所有客户端的类别信息，构建全局类别集合
        all_classes = set()
        for client in self.clients:
            all_classes.update(client.classes)
        self.all_classes = sorted(list(all_classes))
        
        # DM初始化相关设置
        self.dm_data_path = os.path.join(os.path.dirname(__file__), '../results/', 
                                        self.model_identification, 'dm_synthetic_data.pt')
        self.dm_data_generated = False  # 标记是否已生成DM合成数据
        
        # 初始化合成数据：这是FedNum算法的核心数据结构
        self.synthetic_images = self._initialize_synthetic_data()

    def _initialize_synthetic_data(self):
        """
        初始化合成数据集
        
        FedNum算法的关键组件：创建用于训练全局模型的合成数据。
        支持两种初始化策略，以适应不同的应用场景。
        
        合成数据的组织结构：
        - 总数量 = len(all_classes) * ipc
        - 按类别顺序排列：[class_0_images, class_1_images, ...]
        - 每个类别包含ipc个合成图像
        
        返回值：
        - 初始化后的合成图像张量，形状为 [total_images, C, H, W]
        """
        # 计算合成数据的总数量
        total_synthetic_images = len(self.all_classes) * self.ipc
        
        # 步骤1: 创建基础的随机初始化张量
        # 使用标准正态分布初始化，这是深度学习中常用的初始化方法
        synthetic_images = torch.randn(
            size=(
                total_synthetic_images,                    # 总图像数量
                self.dataset_info['channel'],              # 通道数（如RGB=3）
                self.dataset_info['im_size'][0],          # 图像高度
                self.dataset_info['im_size'][1],          # 图像宽度
            ),
            dtype=torch.float,                            # 使用float32精度
            requires_grad=True,                           # 启用梯度计算，用于后续优化
            device=self.device,                           # 直接在目标设备上创建
        )
        
        # 步骤2: 根据初始化方法进行进一步处理
        if self.init_method == 'real_sample':
            print('Initializing synthetic data with real samples...')
            # 使用真实样本初始化：用客户端的真实数据覆盖随机初始化
            # 这种方法的优势：
            # - 提供更好的初始化起点
            # - 加速收敛过程
            # - 保持数据的真实分布特性
            self._initialize_with_real_samples(synthetic_images)
        elif self.init_method == 'dm':
            print('Initializing synthetic data with DM method...')
            # DM初始化：第一轮使用FedDM算法生成，后续轮次从保存的数据中随机选择
            # 这种方法的优势：
            # - 结合了FedDM的高质量合成数据
            # - 后续轮次快速初始化，提高效率
            # - 保持数据凝聚的优势
            self._initialize_with_dm_data(synthetic_images)
        else:
            print('Initializing synthetic data with random noise...')
            # 纯随机初始化：保持原有的随机张量
            # 这种方法的优势：
            # - 不依赖客户端真实数据
            # - 更好的隐私保护
            # - 避免对特定数据分布的偏向
        
        # 输出初始化信息，便于调试和监控
        print(f'Initialized synthetic data with shape: {synthetic_images.shape}')
        print(f'Classes: {self.all_classes}')
        print(f'Images per class: {self.ipc}')
        print(f'Initialization method: {self.init_method}')
        
        return synthetic_images

    def _initialize_with_real_samples(self, synthetic_images, seed_offset=0):
        """
        使用客户端的真实样本初始化合成数据
        
        这种初始化方法的核心思想是用真实数据作为合成数据的初始值，
        从而提供更好的优化起点，加速算法收敛。
        
        算法流程：
        1. 遍历每个类别
        2. 从拥有该类别数据的客户端收集真实样本
        3. 随机选择样本以增加多样性
        4. 用选中的样本替换对应类别的合成数据初始值
        
        隐私考虑：
        - 虽然使用了真实样本，但仅用于初始化
        - 后续的优化过程会逐渐改变这些初始值
        - 最终的合成数据与原始样本有显著差异
        
        参数：
        - synthetic_images: 待初始化的合成图像张量
        - seed_offset: 随机种子偏移量，用于增加随机性
        """
        with torch.no_grad():  # 禁用梯度计算，节省内存和提高效率
            # 遍历每个全局类别
            for i, c in enumerate(self.all_classes):
                # 计算当前类别在合成数据中的存储位置
                start_idx = i * self.ipc  # 起始索引
                end_idx = (i + 1) * self.ipc  # 结束索引
                
                # 初始化样本收集容器
                real_samples = []  # 存储收集到的真实样本
                samples_collected = 0  # 已收集的样本数量
                
                # 步骤1: 随机化客户端访问顺序
                # 为了增加随机性和公平性，避免总是从同一客户端获取样本
                client_indices = list(range(len(self.clients)))
                random.seed(seed_offset * 100 + c)  # 使用类别相关的种子
                random.shuffle(client_indices)
                
                # 步骤2: 从各客户端收集该类别的真实样本
                for client_idx in client_indices:
                    client = self.clients[client_idx]
                    
                    # 检查客户端是否有该类别的数据，且还需要更多样本
                    if c in client.classes and samples_collected < self.ipc:
                        # 获取该客户端该类别的样本索引
                        indices = client.train_set.indices_class.get(c, [])
                        available_samples = len(indices)
                        
                        if available_samples > 0:
                            # 计算从该客户端需要获取的样本数
                            samples_needed = min(self.ipc - samples_collected, available_samples)
                            
                            if samples_needed > 0:
                                # 步骤3: 随机选择样本
                                # 使用不同的种子确保随机性
                                random.seed(seed_offset * 100 + c + client_idx)
                                selected_indices = random.sample(indices, min(samples_needed, len(indices)))
                                
                                # 步骤4: 获取真实样本数据
                                # 优先尝试使用数据集的get_images方法（如果存在）
                                if hasattr(client.train_set, 'get_images'):
                                    try:
                                        # 使用专门的方法获取样本
                                        client_samples = client.train_set.get_images(c, samples_needed, avg=False)
                                        real_samples.append(client_samples)
                                        samples_collected += samples_needed
                                        print(f'Successfully collected {samples_needed} samples from client {client_idx} for class {c}')
                                    except Exception as e:
                                        # 如果专门方法失败，回退到直接索引方式
                                        print(f'get_images failed for client {client_idx}, using direct indexing: {e}')
                                        self._collect_samples_direct(client, selected_indices, real_samples, samples_collected, self.ipc)
                                else:
                                    # 直接从数据集索引获取样本
                                    self._collect_samples_direct(client, selected_indices, real_samples, samples_collected, self.ipc)
                    
                    # 如果已收集足够样本，停止遍历客户端
                    if samples_collected >= self.ipc:
                        break
                
                # 步骤5: 用收集到的真实样本初始化合成数据
                if real_samples:
                    # 处理收集到的样本数据
                    if len(real_samples) == 1 and real_samples[0].size(0) >= self.ipc:
                        # 如果单个样本包含足够多的图像
                        init_samples = real_samples[0][:self.ipc]
                    else:
                        # 拼接多个样本
                        init_samples = torch.cat(real_samples, dim=0)[:self.ipc]
                    
                    # 确保有足够的样本数量
                    if init_samples.size(0) < self.ipc:
                        # 如果样本不足，通过重复现有样本来填充
                        repeats = (self.ipc + init_samples.size(0) - 1) // init_samples.size(0)
                        init_samples = init_samples.repeat(repeats, 1, 1, 1)[:self.ipc]
                    
                    # 将真实样本数据复制到合成数据的对应位置
                    synthetic_images.data[start_idx:end_idx] = init_samples.detach().to(self.device)
                    print(f'Class {c}: initialized with {init_samples.size(0)} real samples')
                else:
                    # 如果没有收集到真实样本，保持随机初始化
                    print(f'Class {c}: no real samples found, keeping random initialization')

    def _collect_samples_direct(self, client, selected_indices, real_samples, samples_collected, ipc):
        """
        直接从客户端数据集索引收集样本的辅助方法
        
        参数：
        - client: 客户端对象
        - selected_indices: 选中的样本索引列表
        - real_samples: 样本收集容器
        - samples_collected: 当前已收集样本数
        - ipc: 每个类别需要的图像数量
        """
        for idx in selected_indices:
            if samples_collected >= ipc:
                break
            # 获取单个样本并添加批次维度
            sample = client.train_set.images_all[idx].unsqueeze(0)
            real_samples.append(sample)
            samples_collected += 1

    def _initialize_with_dm_data(self, synthetic_images):
        """
        使用DM方法初始化合成数据
        
        DM初始化策略：
        1. 第一轮：使用FedDM算法生成高质量的合成数据集并保存
        2. 后续轮次：从保存的DM合成数据集中随机选择
        
        这种方法结合了FedDM的高质量数据生成能力和FedNum的高效统计学习方法。
        
        参数：
        - synthetic_images: 待初始化的合成图像张量
        """
        if os.path.exists(self.dm_data_path) and self.dm_data_generated:
            # 从保存的DM合成数据中随机选择
            print('Loading DM synthetic data from saved file...')
            self._load_and_sample_dm_data(synthetic_images)
        else:
            # 第一轮：使用FedDM算法生成合成数据
            print('First round: generating DM synthetic data using FedDM algorithm...')
            self._generate_dm_data(synthetic_images)
    
    def _generate_dm_data(self, synthetic_images):
        """
        使用FedDM算法生成合成数据集
        
        这个方法在第一轮训练时被调用，使用FedDM算法从客户端生成
        高质量的合成数据集，并保存以供后续轮次使用。
        
        参数：
        - synthetic_images: 待初始化的合成图像张量
        """
        print('Starting FedDM data generation process...')
        
        # 临时创建FedDM客户端来生成合成数据
        from client.fedDMclient import Client as FedDMClient
        
        # 收集所有客户端生成的合成数据
        all_synthetic_data = []
        all_synthetic_labels = []
        
        # 选择参与DM数据生成的客户端
        selected_clients = self.select_clients()
        print(f'Selected {len(selected_clients)} clients for DM data generation')
        
        for client in selected_clients:
            # 创建对应的FedDM客户端
            fedDM_client = FedDMClient(
                cid=client.cid,
                train_set=client.train_set,
                classes=client.classes,
                dataset_info=self.dataset_info,
                ipc=self.ipc,
                rho=self.rho,
                dc_iterations=self.dc_iterations,
                real_batch_size=min(32, len(client.train_set.indices_class.get(client.classes[0], []))),
                image_lr=self.image_lr,
                device=self.device,
                init_method='real_sample'  # 对DM客户端使用真实样本初始化
            )
            
            # 发送全局模型并训练
            fedDM_client.recieve_model(self.global_model)
            client_synthetic_data, client_synthetic_labels = fedDM_client.train()
            
            # 收集合成数据
            all_synthetic_data.append(client_synthetic_data)
            all_synthetic_labels.append(client_synthetic_labels)
            
            print(f'Client {client.cid} generated {client_synthetic_data.shape[0]} synthetic images')
        
        # 合并所有客户端的合成数据
        if all_synthetic_data:
            dm_synthetic_data = torch.cat(all_synthetic_data, dim=0)
            dm_synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
            
            # 保存DM合成数据集
            self._save_dm_data(dm_synthetic_data, dm_synthetic_labels)
            
            # 从生成的数据中采样初始化当前的合成数据
            self._sample_from_dm_data(dm_synthetic_data, dm_synthetic_labels, synthetic_images)
            
            self.dm_data_generated = True
            print(f'DM data generation completed. Generated {dm_synthetic_data.shape[0]} synthetic images.')
        else:
            print('Warning: No synthetic data generated, falling back to random initialization')
    
    def _save_dm_data(self, dm_synthetic_data, dm_synthetic_labels):
        """
        保存DM生成的合成数据集
        
        参数：
        - dm_synthetic_data: DM算法生成的合成图像数据
        - dm_synthetic_labels: 对应的标签
        """
        # 确保保存目录存在
        save_dir = os.path.dirname(self.dm_data_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据
        torch.save({
            'synthetic_data': dm_synthetic_data.cpu(),
            'synthetic_labels': dm_synthetic_labels.cpu(),
            'all_classes': self.all_classes,
            'ipc': self.ipc,
            'dataset_info': self.dataset_info
        }, self.dm_data_path)
        
        print(f'DM synthetic data saved to: {self.dm_data_path}')
    
    def _load_and_sample_dm_data(self, synthetic_images):
        """
        从保存的DM合成数据中加载并采样
        
        参数：
        - synthetic_images: 待初始化的合成图像张量
        """
        try:
            dm_data = torch.load(self.dm_data_path, map_location='cpu')
            dm_synthetic_data = dm_data['synthetic_data']
            dm_synthetic_labels = dm_data['synthetic_labels']
            
            print(f'Loaded DM synthetic data: {dm_synthetic_data.shape[0]} images')
            
            # 从加载的数据中采样
            self._sample_from_dm_data(dm_synthetic_data, dm_synthetic_labels, synthetic_images)
            
        except Exception as e:
            print(f'Error loading DM synthetic data: {e}')
            print('Falling back to random initialization')
    
    def _sample_from_dm_data(self, dm_synthetic_data, dm_synthetic_labels, synthetic_images):
        """
        从DM合成数据中为每个类别采样指定数量的图像
        
        参数：
        - dm_synthetic_data: DM算法生成的合成图像数据
        - dm_synthetic_labels: 对应的标签
        - synthetic_images: 待初始化的合成图像张量
        """
        with torch.no_grad():
            for i, c in enumerate(self.all_classes):
                start_idx = i * self.ipc
                end_idx = (i + 1) * self.ipc
                
                # 找到属于当前类别的所有DM合成图像
                class_mask = (dm_synthetic_labels == c)
                class_data = dm_synthetic_data[class_mask]
                
                if len(class_data) > 0:
                    if len(class_data) >= self.ipc:
                        # 如果有足够的样本，随机选择
                        indices = torch.randperm(len(class_data))[:self.ipc]
                        selected_data = class_data[indices]
                    else:
                        # 如果样本不足，重复采样
                        repeats = (self.ipc + len(class_data) - 1) // len(class_data)
                        repeated_data = class_data.repeat(repeats, 1, 1, 1)
                        selected_data = repeated_data[:self.ipc]
                    
                    # 初始化合成数据
                    synthetic_images.data[start_idx:end_idx] = selected_data.to(self.device)
                    print(f'Class {c}: initialized with {selected_data.shape[0]} DM synthetic images')
                else:
                    print(f'Class {c}: no DM synthetic data found, keeping random initialization')

    def fit(self):
        """
        FedNum算法的主训练循环
        
        这是算法的核心执行流程，包含以下主要阶段：
        1. 模型评估阶段：定期评估全局模型性能
        2. 客户端协调阶段：选择客户端并收集统计信息
        3. 合成数据更新阶段：使用客户端统计信息优化合成数据
        4. 全局模型训练阶段：使用合成数据训练全局模型
        5. 模型保存阶段：保存训练检查点
        """
        evaluate_acc = 0
        
        # 主要的联邦学习训练循环
        for rounds in range(self.communication_rounds):
            print(f' ====== round {rounds} ======')
            
            # === 阶段1: 模型评估 ===
            # 定期评估全局模型在测试集上的性能
            if rounds % self.eval_gap == 0:
                acc = self.evaluate()
                print(f'round {rounds} evaluation: test acc is {acc}')
                evaluate_acc = acc

            start_time = time.time()

            # === 阶段2: 客户端协调和数据收集 ===
            print('---------- client training ----------')

            # 步骤2.1: 选择参与本轮训练的客户端
            # 根据join_ratio参数随机选择客户端，支持部分客户端参与
            selected_clients = self.select_clients()
            print('selected clients:', [selected_client.cid for selected_client in selected_clients])
            
            # 步骤2.2: 从选中的客户端收集统计信息
            client_data_list = []
            for client in selected_clients:
                # 发送全局模型给客户端
                client.recieve_model(self.global_model)
                
                # 客户端处理本地数据，返回平均后的特征和logits统计信息
                # 这是FedNum算法的核心：传输统计信息而不是原始数据或模型参数
                client_data = client.train()  # 返回包含平均特征和logits的字典
                client_data_list.append(client_data)

            # === 阶段3: 合成数据更新 ===
            print('---------- update synthetic data ----------')

            # 使用收集到的客户端统计信息通过梯度匹配更新合成数据
            # 这是算法的创新点：通过优化合成数据使其产生类似的特征和logits分布
            self._update_synthetic_data(client_data_list)

            # === 阶段4: 全局模型训练 ===
            print('---------- update global model ----------')

            # 步骤4.1: 准备合成数据集用于训练
            synthetic_data, synthetic_labels = self._get_synthetic_dataset()
            synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=4)
            
            # 步骤4.2: 配置全局模型训练
            self.global_model.train()  # 切换到训练模式
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=0.01,
                weight_decay=0.0005,
                momentum=0.9,
            )
            loss_function = torch.nn.CrossEntropyLoss()

            # 步骤4.3: 使用合成数据训练全局模型
            total_loss = 0
            for epoch in tqdm(range(self.model_epochs), desc='global model training', leave=True):
                for x, target in synthetic_dataloader:
                    # 数据移至设备
                    x, target = x.to(self.device), target.to(self.device)
                    target = target.long()
                    
                    # 前向传播
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    
                    # 反向传播和参数更新
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

    def _compute_logit_loss(self, synthetic_logit, target_logit, loss_type):
        """
        计算不同类型的logits损失函数
        
        这是FedNum算法中的关键组件，用于度量合成数据的输出分布
        与真实数据输出分布之间的差异。支持三种不同的损失类型。
        
        参数：
        - synthetic_logit: 合成数据通过模型得到的logits输出
        - target_logit: 目标logits（来自客户端真实数据的统计）
        - loss_type: 损失类型 ('wasserstein', 'kl', 'l2')
        
        返回值：
        - 计算得到的损失值（标量张量）
        """
        
        if loss_type == 'wasserstein':
            # Wasserstein距离（Earth Mover's Distance）
            # 这是一种度量两个概率分布之间"传输成本"的方法
            # 相比KL散度，Wasserstein距离在分布支撑不重叠时仍然有意义
            
            # 将logits转换为概率分布
            synthetic_prob = F.softmax(synthetic_logit, dim=0)
            target_prob = F.softmax(target_logit, dim=0)
            
            # 对于离散分布的Wasserstein-1距离计算：
            # 1. 对概率值排序
            synthetic_sorted, synthetic_indices = torch.sort(synthetic_prob)
            target_sorted, target_indices = torch.sort(target_prob)
            
            # 2. 计算累积分布函数(CDF)
            synthetic_cumsum = torch.cumsum(synthetic_sorted, dim=0)
            target_cumsum = torch.cumsum(target_sorted, dim=0)
            
            # 3. Wasserstein-1距离 = 累积分布函数差值的积分
            wasserstein_loss = torch.sum(torch.abs(synthetic_cumsum - target_cumsum))
            return wasserstein_loss
            
        elif loss_type == 'kl':
            # Kullback-Leibler散度
            # 度量一个概率分布相对于另一个分布的"信息增益"
            # KL(P||Q) = ∑ P(x) * log(P(x)/Q(x))
            
            # 转换为概率分布
            synthetic_prob = F.softmax(synthetic_logit, dim=0)
            target_prob = F.softmax(target_logit, dim=0)
            
            # 使用PyTorch的稳定实现计算KL散度
            # F.kl_div期望第一个参数是log概率，第二个参数是概率
            kl_loss = F.kl_div(
                F.log_softmax(synthetic_logit, dim=0),  # log P
                target_prob,                            # Q
                reduction='sum'
            )
            return kl_loss
            
        elif loss_type == 'l2':
            # L2损失（均方误差）
            # 简单直观，计算效率高
            # ||synthetic_logit - target_logit||²
            l2_loss = torch.sum((synthetic_logit - target_logit) ** 2)
            return l2_loss
            
        else:
            raise ValueError(f"Unsupported logit loss type: {loss_type}. "
                           f"Supported types: 'wasserstein', 'kl', 'l2'")

    def _update_synthetic_data(self, client_data_list: List[Dict]):
        """
        使用客户端统计信息更新合成数据的核心算法
        
        这是FedNum算法的关键创新：通过梯度匹配优化合成数据，
        使得合成数据在当前全局模型下产生与客户端真实数据相似的特征和logits分布。
        
        算法步骤：
        1. 数据预处理：合并所有客户端的统计信息
        2. 梯度匹配优化：通过多次迭代优化合成数据
        3. 特征匹配：使合成数据的特征分布接近真实数据
        4. Logits匹配：使合成数据的输出分布接近真实数据
        
        注意：使用全局模型而不是扰动模型进行特征提取，确保与客户端使用的模型完全一致
        
        参数：
        - client_data_list: 包含所有客户端统计信息的列表
        """
        
        # 步骤1: 数据预处理 - 重新初始化（可选）
        # 如果使用真实样本初始化，每轮都重新初始化合成数据以增加多样性
        if self.init_method == 'real_sample':
            print('Re-initializing synthetic data with real samples for this round...')
            self._initialize_with_real_samples(self.synthetic_images)
        elif self.init_method == 'dm':
            print('Re-initializing synthetic data with DM method for this round...')
            self._initialize_with_dm_data(self.synthetic_images)
        
        # 步骤2: 数据整合 - 合并所有客户端的统计信息
        all_avg_features = {}  # 存储所有客户端的平均特征
        all_avg_logits = {}    # 存储所有客户端的平均logits
        
        # 遍历每个客户端的数据
        for client_data in client_data_list:
            for c in client_data['avg_features']:
                if c not in all_avg_features:
                    all_avg_features[c] = []
                    all_avg_logits[c] = []
                # 将同一类别的数据从不同客户端合并
                all_avg_features[c].append(client_data['avg_features'][c])
                all_avg_logits[c].append(client_data['avg_logits'][c])

        # 步骤3: 数据拼接 - 将每个类别的统计信息拼接成张量
        for c in all_avg_features:
            all_avg_features[c] = torch.cat(all_avg_features[c], dim=0)
            all_avg_logits[c] = torch.cat(all_avg_logits[c], dim=0)

        print(f'Collected averaged features and logits for classes: {list(all_avg_features.keys())}')

        # 步骤4: 合成数据优化设置
        # 为合成图像设置优化器，使用SGD进行梯度匹配优化
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)

        # 步骤5: 迭代优化合成数据
        # 这是FedNum算法的核心：通过梯度匹配优化合成数据质量
        for dc_iteration in range(self.dc_iterations):
            
            # 步骤5.1: 使用全局模型进行特征提取
            # 直接使用全局模型而不是扰动模型，确保与客户端使用的模型一致
            self.global_model.eval()  # 设置为评估模式

            total_loss = torch.tensor(0.0).to(self.device)

            # 步骤5.2: 遍历每个类别进行优化
            for i, c in enumerate(self.all_classes):
                if c not in all_avg_features:
                    continue  # 跳过没有数据的类别
                
                # 计算当前类别合成图像在存储中的索引范围
                start_idx = i * self.ipc
                end_idx = (i + 1) * self.ipc
                
                # 步骤5.3: 获取当前类别的合成图像
                # 将扁平化的合成图像重塑为正确的图像格式
                synthetic_images_class = self.synthetic_images[start_idx:end_idx].reshape(
                    (self.ipc, self.dataset_info['channel'], 
                     self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))

                # 步骤5.4: 计算合成数据的特征和logits
                # 使用全局模型提取合成图像的特征和输出，与客户端保持一致
                synthetic_features = self.global_model.embed(synthetic_images_class)
                synthetic_logits = self.global_model(synthetic_images_class)

                # 步骤5.5: 选择目标统计信息
                # 从收集的客户端统计信息中随机采样一部分进行匹配
                # 这样可以平衡计算效率和优化效果
                available_avg_samples = all_avg_features[c].size(0)
                sample_size = min(self.batch_num // self.avg_num, available_avg_samples)
                
                if sample_size > 0:
                    # 随机选择样本，增加训练的随机性
                    indices = torch.randperm(available_avg_samples)[:sample_size]
                    target_avg_features = all_avg_features[c][indices].to(self.device)
                    target_avg_logits = all_avg_logits[c][indices].to(self.device)

                    # 步骤5.6: 计算合成数据的统计摘要
                    # 计算当前类别合成数据的平均特征和平均logits
                    synthetic_feature_mean = torch.mean(synthetic_features, dim=0)
                    synthetic_logit_mean = torch.mean(synthetic_logits, dim=0)

                    # 步骤5.7: 计算匹配损失
                    # 根据配置的损失计算方式选择不同的计算策略
                    feature_loss = torch.tensor(0.0).to(self.device)
                    logit_loss = torch.tensor(0.0).to(self.device)
                    
                    if self.loss_computation == 'loss-avg':
                        # 策略1: loss-avg
                        # 分别计算每个目标统计信息与合成数据的损失，然后取平均
                        # 这种方式考虑了所有目标样本的个体差异
                        for j in range(sample_size):
                            target_feature = target_avg_features[j]
                            target_logit = target_avg_logits[j]
                            
                            # 特征匹配损失：使用L2距离度量特征相似性
                            feature_loss += torch.sum((synthetic_feature_mean - target_feature) ** 2)
                            
                            # Logits匹配损失：使用指定的损失类型度量输出分布相似性
                            logit_loss += self._compute_logit_loss(
                                synthetic_logit_mean, 
                                target_logit, 
                                self.logit_loss_type
                            )
                        
                        # 计算平均损失，避免因样本数量影响梯度规模
                        feature_loss = feature_loss / sample_size
                        logit_loss = logit_loss / sample_size
                        
                    elif self.loss_computation == 'avg-loss':
                        # 策略2: avg-loss
                        # 先对目标统计信息取平均，然后计算一次损失
                        # 这种方式关注整体的平均效果，计算更高效
                        target_feature_mean = torch.mean(target_avg_features, dim=0)
                        target_logit_mean = torch.mean(target_avg_logits, dim=0)
                        
                        # 特征匹配损失：直接比较平均特征
                        feature_loss = torch.sum((synthetic_feature_mean - target_feature_mean) ** 2)
                        
                        # Logits匹配损失：直接比较平均输出
                        logit_loss = self._compute_logit_loss(
                            synthetic_logit_mean, 
                            target_logit_mean, 
                            self.logit_loss_type
                        )
                    
                    else:
                        raise ValueError(f"Unsupported loss computation method: {self.loss_computation}")
                    
                    # 累加所有类别的损失
                    total_loss += feature_loss + logit_loss

            # 步骤5.8: 更新合成数据
            # 使用梯度下降优化合成图像，使其产生期望的特征和输出分布
            optimizer_image.zero_grad()  # 清零梯度
            total_loss.backward()        # 反向传播计算梯度
            optimizer_image.step()       # 更新合成图像参数

            # 定期输出训练进度
            if dc_iteration % 100 == 0:
                print(f'Synthetic data update iteration {dc_iteration}, total loss = {total_loss.item()}')

    def _get_synthetic_dataset(self):
        """
        获取完整的合成数据集，用于训练全局模型
        
        返回值：
        - synthetic_images: 合成图像数据 [total_images, channels, height, width]
        - synthetic_labels: 对应的标签 [total_images]
        
        数据组织方式：
        - 按类别顺序排列，每个类别包含ipc个合成图像
        - 标签按照类别重复ipc次
        """
        # 构建标签序列
        synthetic_labels = []
        for c in self.all_classes:
            # 为每个类别创建ipc个标签
            synthetic_labels.extend([c] * self.ipc)
        
        # 转换为张量格式
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
        
        # 返回合成数据和标签（从GPU转移到CPU以节省显存）
        return self.synthetic_images.detach().cpu(), synthetic_labels

    def select_clients(self):
        """
        客户端选择策略
        
        根据join_ratio参数决定每轮参与训练的客户端：
        - 如果join_ratio = 1.0，所有客户端都参与
        - 否则随机选择指定比例的客户端参与
        
        这种策略的优势：
        - 减少每轮的计算和通信开销
        - 增加训练的随机性，提高模型泛化能力
        - 模拟现实中客户端可能离线的情况
        
        返回值：
        - 被选中参与训练的客户端列表
        """
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        """
        评估全局模型在测试集上的性能
        
        评估流程：
        1. 设置模型为评估模式
        2. 在测试集上进行前向传播
        3. 计算预测准确率
        
        返回值：
        - 模型在测试集上的准确率
        
        注意：
        - 使用torch.no_grad()节省内存
        - 在GPU上进行推理以提高速度
        """
        self.global_model.eval()  # 设置为评估模式
        
        with torch.no_grad():
            correct, total = 0, 0
            
            # 遍历测试数据
            for x, target in self.test_loader:
                # 将数据移至计算设备
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                
                # 模型推理
                pred = self.global_model(x)
                
                # 获取预测标签
                _, pred_label = torch.max(pred.data, 1)
                
                # 统计正确预测数量
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        # 计算并返回准确率
        return correct / float(total)

    def make_checkpoint(self, rounds):
        """
        创建模型检查点
        
        保存当前训练状态，包括：
        - 当前训练轮次
        - 全局模型的参数状态
        
        参数：
        - rounds: 当前训练轮次
        
        返回值：
        - 包含检查点信息的字典
        """
        checkpoint = {
            'current_round': rounds,
            'model': self.global_model.state_dict()
        }
        return checkpoint

    def save_model(self, path, rounds, include_image):
        """
        保存模型到指定路径
        
        参数：
        - path: 保存路径
        - rounds: 当前轮次
        - include_image: 是否保存合成图像（暂未实现）
        
        功能：
        - 保存模型检查点到文件
        - 支持训练恢复和模型部署
        """
        torch.save(self.make_checkpoint(rounds), os.path.join(path, 'model.pt'))
        
        # TODO: 实现合成图像保存功能
        if include_image:
            raise NotImplementedError('Synthetic image saving not implemented yet')
