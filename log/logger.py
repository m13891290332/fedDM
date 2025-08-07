import logging
import sys
from datetime import datetime
import os

class FedMKLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"fedmk_{timestamp}.log")
        
        # 配置logging
        self.logger = logging.getLogger("FedMK")
        self.logger.setLevel(logging.INFO)
        
        # 文件handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(message)s'
        ))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# 全局logger实例
logger = FedMKLogger()