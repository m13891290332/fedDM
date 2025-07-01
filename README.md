client文件夹保存客户端文件
dataset文件夹保存数据集下载、划分、划分信息
datasets文件夹保存下载的数据集、划分的数据集
log文件夹保存日志文件，命名格式为：{时间戳}_{数据集}_alpha{alpha值}_{客户端数量}clients_{模型}_{ipc}ipc_{dc迭代次数}dc_{模型训练轮数}epochs_cr{通信轮数}.log
models文件夹保存模型结构文件
results文件夹保存模型参数文件
server文件夹保存服务器文件
utils文件夹保存随机化等设置文件
wandb文件夹是因为我的电脑没法远程连接wandb，所以选择离线运行而产生的wandb相关文件，这个我也不太清楚
config.py文件是参数文件
main.py是主函数




先运行数据集划分（如果修改了数据集、客户端数量和alpha参数，必须重新运行一次）
python dataset/data/dataset_partition.py --dataset CIFAR10 --client_num 10 --alpha 0.5 --dataset_root datasets/torchvision
main.py方式运行（推荐）（设置使用哪个gpu的环境变量在main.py中修改）
python main.py --model ConvNet --dataset CIFAR10 --client_num 10 --alpha 0.5 --ipc 50 --dc_iterations 1000 --model_epochs 500




run.sh方式运行（未测试）
chmod +x run.sh  # 给脚本添加执行权限
./run.sh  # 运行脚本