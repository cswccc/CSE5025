"""
数据生成模块
用于生成充电桩覆盖收益最大化问题的测试数据
"""

import numpy as np
import random
from typing import Tuple, Dict, List
import json


class DataGenerator:
    """数据生成器类"""
    
    def __init__(self, seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子
        """
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_coverage_matrix(self, n: int, m: int, coverage_rate: float = 0.3) -> np.ndarray:
        """
        生成覆盖关系矩阵 a_ij
        
        Args:
            n: 楼栋数量
            m: 区域数量
            coverage_rate: 覆盖率，表示平均每个区域能覆盖多少比例的楼栋
            
        Returns:
            a_ij: n x m 的覆盖关系矩阵
        """
        # 使用伯努利分布生成覆盖关系
        a_ij = np.random.binomial(1, coverage_rate, size=(n, m))
        
        # 确保每个楼栋至少被一个区域覆盖
        for i in range(n):
            if np.sum(a_ij[i, :]) == 0:
                j = random.randint(0, m - 1)
                a_ij[i, j] = 1
        
        # 确保每个区域至少覆盖一个楼栋
        for j in range(m):
            if np.sum(a_ij[:, j]) == 0:
                i = random.randint(0, n - 1)
                a_ij[i, j] = 1
        
        return a_ij
    
    def generate_demands(self, n: int, min_demand: int = 10, max_demand: int = 100) -> np.ndarray:
        """
        生成每栋楼的潜在用户需求 D_i
        
        Args:
            n: 楼栋数量
            min_demand: 最小需求
            max_demand: 最大需求
            
        Returns:
            D: 长度为n的需求数组
        """
        return np.random.randint(min_demand, max_demand + 1, size=n)
    
    def generate_unit_profits(self, n: int, min_profit: float = 1.0, max_profit: float = 10.0) -> np.ndarray:
        """
        生成每栋楼的单位收益 p_i
        
        Args:
            n: 楼栋数量
            min_profit: 最小单位收益
            max_profit: 最大单位收益
            
        Returns:
            p: 长度为n的单位收益数组
        """
        return np.random.uniform(min_profit, max_profit, size=n)
    
    def generate_costs(self, m: int, min_cost: float = 50.0, max_cost: float = 500.0) -> np.ndarray:
        """
        生成每个区域的建设成本 c_j
        
        Args:
            m: 区域数量
            min_cost: 最小成本
            max_cost: 最大成本
            
        Returns:
            c: 长度为m的成本数组
        """
        return np.random.uniform(min_cost, max_cost, size=m)
    
    def generate_capacity_limits(self, m: int, min_capacity: int = 20, max_capacity: int = 200) -> np.ndarray:
        """
        生成每个区域的容量上限 U_j
        
        Args:
            m: 区域数量
            min_capacity: 最小容量上限
            max_capacity: 最大容量上限
            
        Returns:
            U: 长度为m的容量上限数组
        """
        return np.random.randint(min_capacity, max_capacity + 1, size=m)
    
    def generate_instance(
        self, 
        n: int = 20, 
        m: int = 10,
        coverage_rate: float = 0.3,
        min_demand: int = 10,
        max_demand: int = 100,
        min_profit: float = 1.0,
        max_profit: float = 10.0,
        min_cost: float = 50.0,
        max_cost: float = 500.0,
        min_capacity: int = 20,
        max_capacity: int = 200
    ) -> Dict:
        """
        生成完整的问题实例
        
        Args:
            n: 楼栋数量
            m: 区域数量
            coverage_rate: 覆盖率
            min_demand: 最小需求
            max_demand: 最大需求
            min_profit: 最小单位收益
            max_profit: 最大单位收益
            min_cost: 最小建设成本
            max_cost: 最大建设成本
            min_capacity: 最小容量上限
            max_capacity: 最大容量上限
            
        Returns:
            包含所有问题参数的字典
        """
        instance = {
            'n': n,  # 楼栋数量
            'm': m,  # 区域数量
            'D': self.generate_demands(n, min_demand, max_demand).tolist(),  # 需求
            'p': self.generate_unit_profits(n, min_profit, max_profit).tolist(),  # 单位收益
            'c': self.generate_costs(m, min_cost, max_cost).tolist(),  # 建设成本
            'U': self.generate_capacity_limits(m, min_capacity, max_capacity).tolist(),  # 容量上限
            'a': self.generate_coverage_matrix(n, m, coverage_rate).tolist()  # 覆盖关系矩阵
        }
        
        return instance
    
    def save_instance(self, instance: Dict, filepath: str):
        """保存问题实例到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
    
    def load_instance(self, filepath: str) -> Dict:
        """从JSON文件加载问题实例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # 测试数据生成
    generator = DataGenerator(seed=42)
    
    # 生成一个小规模实例用于测试
    instance = generator.generate_instance(
        n=10, 
        m=5,
        coverage_rate=0.4
    )
    
    print("生成的问题实例:")
    print(f"楼栋数量: {instance['n']}")
    print(f"区域数量: {instance['m']}")
    print(f"需求: {instance['D']}")
    print(f"单位收益: {[f'{p:.2f}' for p in instance['p']]}")
    print(f"建设成本: {[f'{c:.2f}' for c in instance['c']]}")
    print(f"容量上限: {instance['U']}")
    print(f"覆盖矩阵形状: {len(instance['a'])}x{len(instance['a'][0])}")
    
    # 保存示例
    generator.save_instance(instance, "test_instance.json")
