"""
基础求解器类
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class BaseSolver(ABC):
    """求解器基类"""
    
    def __init__(self, instance: Dict):
        """
        初始化求解器
        
        Args:
            instance: 问题实例字典
        """
        self.instance = instance
        self.n = instance['n']  # 楼栋数量
        self.m = instance['m']  # 区域数量
        self.D = np.array(instance['D'])  # 需求
        self.p = np.array(instance['p'])  # 单位收益
        self.c = np.array(instance['c'])  # 建设成本
        self.U = np.array(instance['U'])  # 容量上限
        self.a = np.array(instance['a'])  # 覆盖关系矩阵
        
        self.best_solution = None
        self.best_objective = float('-inf')
        self.solve_time = 0.0
    
    @abstractmethod
    def solve(self) -> Tuple[Dict, float]:
        """
        求解问题
        
        Returns:
            (solution, objective_value): 解字典和目标函数值
        """
        pass
    
    def calculate_objective(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算目标函数值
        
        Args:
            z: 建设决策 (m,)
            x: 充电桩数量 (m,)
            y: 用户分配 (n, m)
            
        Returns:
            净收益值
        """
        # 总收益
        total_revenue = np.sum(self.p[:, np.newaxis] * y)
        
        # 总成本
        total_cost = np.sum(self.c * z)
        
        # 净收益
        net_profit = total_revenue - total_cost
        
        return net_profit
    
    def is_feasible(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """
        检查解的可行性
        
        Args:
            z: 建设决策 (m,)
            x: 充电桩数量 (m,)
            y: 用户分配 (n, m)
            
        Returns:
            (is_feasible, error_message)
        """
        # 检查类型
        if not np.all((z == 0) | (z == 1)):
            return False, "z必须为0-1变量"
        
        if np.any(x < 0):
            return False, "x必须非负"
        
        if np.any(y < 0):
            return False, "y必须非负"
        
        # 检查覆盖关系约束: y_ij <= D_i * a_ij * z_j
        for i in range(self.n):
            for j in range(self.m):
                if y[i, j] > self.D[i] * self.a[i, j] * z[j] + 1e-6:
                    return False, f"违反覆盖关系约束: y[{i},{j}] > D[{i}] * a[{i},{j}] * z[{j}]"
        
        # 检查需求约束: sum_j y_ij <= D_i
        for i in range(self.n):
            if np.sum(y[i, :]) > self.D[i] + 1e-6:
                return False, f"违反需求约束: 楼栋{i}的用户分配超过需求"
        
        # 检查容量约束: sum_i y_ij <= x_j
        for j in range(self.m):
            if np.sum(y[:, j]) > x[j] + 1e-6:
                return False, f"违反容量约束: 区域{j}的服务量超过容量"
        
        # 检查容量上限约束: 0 <= x_j <= U_j * z_j
        for j in range(self.m):
            if x[j] < 0:
                return False, f"违反非负约束: x[{j}] < 0"
            if x[j] > self.U[j] * z[j] + 1e-6:
                return False, f"违反容量上限约束: x[{j}] > U[{j}] * z[{j}]"
        
        return True, "可行"
