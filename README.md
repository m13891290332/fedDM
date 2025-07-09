文件信息介绍：{
client文件夹保存客户端文件
dataset文件夹保存数据集下载、划分、划分信息
datasets文件夹保存下载的数据集、划分的数据集
log文件夹保存日志文件，命名格式为：{时间戳}_{数据集}_alpha{alpha值}_{客户端数量}clients_{模型}_{ipc}ipc_{dc迭代次数}dc_{模型训练轮数}epochs_cr{通信轮数}.log
models文件夹保存模型结构文件
results文件夹保存模型参数文件
server文件夹保存服务器文件
utils文件夹保存随机化等设置文件
wandb文件夹是因为我的电脑没法远程连接wandb，所以选择离线运行而产生的wandb相关文件
config.py文件是参数文件
main.py是主函数
}




运行方式：{
主目录创建log、results、datasets/torchvision文件夹
先运行数据集划分（如果修改了数据集、客户端数量和alpha参数，必须重新运行一次）
python dataset/data/dataset_partition.py --dataset CIFAR10 --client_num 50 --alpha 0.1 --dataset_root datasets/torchvision

main.py方式运行（推荐）（设置使用哪个gpu的环境变量在main.py中修改）
python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 10 --dc_iterations 1000 --model_epochs 500  --partition_method part --algorithm fedprotoDM

测试
python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 10 --dc_iterations 10 --model_epochs 50  --partition_method part --algorithm fedprotoDM --communication_rounds 20
}



后台运行、查看、终止的常见命令：{
nohup python main.py --model ConvNet --dataset CIFAR10 --client_num 50 --alpha 0.1 --ipc 10 --dc_iterations 1000 --model_epochs 500  --partition_method only > app.log 2>&1 &
ps -ef | grep MaCS
kill
tail -n 200 -f app.log
}




算法比较：{
功能               FedDM              FedprotoDM
数据合成位置        客户端	             服务器端
客户端发送         合成图像	             统计信息
隐私保护	         中等	              更好
计算资源利用      分散在客户端	       集中在服务器
通信开销	     较大（图像数据）	  较小（统计信息）
扩展性	         受客户端限制	        服务器主导
}