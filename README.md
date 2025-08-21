# FedDM - 联邦学习项目

## 文件信息介绍

- `client/` - 保存客户端文件
- `dataset/` - 保存数据集下载、划分、划分信息
- `datasets/` - 保存下载的数据集、划分的数据集
- `log/` - 保存日志文件，命名格式为：
  ```
  {时间戳}_{数据集}_alpha{alpha值}_{客户端数量}clients_{模型}_{ipc}ipc_{dc迭代次数}dc_{模型训练轮数}epochs_cr{通信轮数}.log
  ```
- `models/` - 保存模型结构文件
- `results/` - 保存模型参数文件
- `server/` - 保存服务器文件
- `utils/` - 保存随机化等设置文件
- `wandb/` - 因为电脑没法远程连接wandb，所以选择离线运行而产生的wandb相关文件
- `config.py` - 参数文件
- `main.py` - 主函数

## 运行方式

### 1. 环境准备
主目录创建以下文件夹：
- `log/`
- `results/`
- `datasets/torchvision/`

### 2. 数据集划分
先运行数据集划分（如果修改了数据集、客户端数量和alpha参数，必须重新运行一次）：
```bash
python dataset/data/dataset_partition.py --dataset CIFAR10 --client_num 50 --alpha 0.1 --dataset_root datasets/torchvision
```

### 3. 主程序运行（推荐）
```bash
python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 10 --dc_iterations 1000 --model_epochs 50 --partition_method only --algorithm fednum --communication_rounds 20 --device cuda:2 --init_method real_sample
```

### 5. 测试FedMK算法
```bash
python main.py --model LeNet --dataset CIFAR10 --client_num 10 --alpha 1 --model_epochs 10 --partition_method only --algorithm fedMK
```

## 后台运行管理

### 后台运行
```bash
nohup python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 50 --dc_iterations 1000 --model_epochs 50 --partition_method only --algorithm fednum --communication_rounds 20 --device cuda:2 --init_method real_sample > numreal50.log 2>&1 &
nohup python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 50 --dc_iterations 1000 --model_epochs 50 --partition_method only --algorithm fednum --communication_rounds 20 --device cuda:3 --init_method random > numrandom50.log 2>&1 &
```

### 查看进程
```bash
ps -ef | grep MaCS
```

### 终止进程
```bash
kill [进程ID]
```

### 查看日志
```bash
tail -n 200 -f app.log
```
