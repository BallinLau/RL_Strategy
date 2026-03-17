"""
数据缓存模块

用于缓存预处理后的数据，提高训练效率
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import hashlib
import pickle
import os


class DataCache:
    """数据缓存类"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化数据缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            params: 参数字典
            
        Returns:
            cache_key: 缓存键
        """
        # 将参数转换为字符串
        param_str = str(sorted(params.items()))
        # 生成MD5哈希
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def save(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """
        保存数据到缓存
        
        Args:
            data: 要缓存的数据
            params: 参数字典
            
        Returns:
            cache_path: 缓存文件路径
        """
        cache_key = self.get_cache_key(params)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        return cache_path
    
    def load(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            params: 参数字典
            
        Returns:
            data: 缓存的数据，如果不存在则返回None
        """
        cache_key = self.get_cache_key(params)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def clear(self):
        """清空缓存"""
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        return {
            'cache_dir': self.cache_dir,
            'num_files': len(cache_files),
            'total_size': sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files
            ) / 1024 / 1024,  # MB
            'files': cache_files
        }
    
    def __str__(self) -> str:
        info = self.get_cache_info()
        return (f"DataCache(dir={self.cache_dir}, "
                f"files={info['num_files']}, "
                f"size={info['total_size']:.2f}MB)")
    
    def __repr__(self) -> str:
        return self.__str__()