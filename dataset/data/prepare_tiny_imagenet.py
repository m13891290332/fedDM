import os
import urllib.request
import zipfile
import shutil

def download_and_prepare_tiny_imagenet(root_dir):
    """
    下载并准备 TinyImageNet 数据集
    
    参数:
        root_dir: 数据集存放的根目录
    """
    # 创建目录
    os.makedirs(root_dir, exist_ok=True)
    dataset_dir = os.path.join(root_dir, 'tiny-imagenet-200')
    
    # 下载数据集
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(root_dir, 'tiny-imagenet-200.zip')
    
    if not os.path.exists(dataset_dir):
        print('开始下载 TinyImageNet 数据集...')
        urllib.request.urlretrieve(url, zip_path)
        
        print('解压数据集...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        
        # 删除zip文件
        os.remove(zip_path)
        print('数据集准备完成！')
    else:
        print('TinyImageNet 数据集已存在')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/home/ChenXY/fedDM/datasets/torchvision',
                       help='数据集存放的根目录')
    args = parser.parse_args()
    
    download_and_prepare_tiny_imagenet(args.dataset_root)
