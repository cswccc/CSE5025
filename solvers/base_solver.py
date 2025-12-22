"""
基础求解器类

本模块定义了所有求解器的基类，提供了公共的方法和接口。
所有具体的求解器都继承自BaseSolver，并实现solve()方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class BaseSolver(ABC):
    """
    求解器基类
    
    这是一个抽象基类，定义了求解器的基本接口和公共方法。
    所有具体的求解算法都应该继承此类并实现solve()方法。
    
    属性说明:
        instance: 问题实例字典，包含所有输入参数
        n: 楼栋数量
        m: 区域数量  
        D: 每栋楼的潜在用户需求数组 (n,)
        p: 每栋楼的单位收益数组 (n,)
        c: 每个区域的建设成本数组 (m,)
        U: 每个区域的容量上限数组 (m,)
        a: 覆盖关系矩阵 (n, m)，a[i,j]=1表示区域j可以覆盖楼栋i
        best_solution: 找到的最佳解（字典格式，包含z, x, y）
        best_objective: 最佳解的目标函数值
        solve_time: 求解耗时（秒）
    """
    
    def __init__(self, instance: Dict):
        """
        初始化求解器
        
        Args:
            instance: 问题实例字典，必须包含以下键:
                - 'n': 楼栋数量 (int)
                - 'm': 区域数量 (int)
                - 'D': 需求列表 (list[float])
                - 'p': 单位收益列表 (list[float])
                - 'c': 建设成本列表 (list[float])
                - 'U': 容量上限列表 (list[int])
                - 'a': 覆盖关系矩阵 (list[list[int]])
        """
        self.instance = instance
        self.n = instance['n']  # 楼栋数量
        self.m = instance['m']  # 区域数量
        self.D = np.array(instance['D'])  # 需求数组 (n,)
        self.p = np.array(instance['p'])  # 单位收益数组 (n,)
        self.c = np.array(instance['c'])  # 建设成本数组 (m,)
        self.U = np.array(instance['U'])  # 容量上限数组 (m,)
        self.a = np.array(instance['a'])  # 覆盖关系矩阵 (n, m)
        
        # 初始化最佳解相关属性
        self.best_solution = None  # 最佳解字典 {'z': [...], 'x': [...], 'y': [[...]]}
        self.best_objective = float('-inf')  # 最佳目标函数值
        self.solve_time = 0.0  # 求解耗时
    
    @abstractmethod
    def solve(self) -> Tuple[Dict, float]:
        """
        求解问题的抽象方法
        
        所有子类必须实现此方法，用于执行具体的求解算法。
        
        Returns:
            Tuple[Dict, float]: 
                - solution: 解字典，包含:
                    - 'z': 建设决策列表 (list[int])，z[j]=1表示在区域j建设
                    - 'x': 充电桩数量列表 (list[float])，x[j]表示区域j的充电桩数
                    - 'y': 用户分配矩阵 (list[list[float]])，y[i][j]表示楼栋i分配到区域j的用户数
                - objective_value: 目标函数值（净收益），float类型
        """
        pass
    
    def calculate_objective(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算目标函数值（净收益）
        
        目标函数 = 总收益 - 总成本
        总收益 = Σ_i Σ_j (p_i * y_ij)  所有用户带来的收益总和
        总成本 = Σ_j (c_j * z_j)        所有区域的建设成本总和
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]∈{0,1}表示是否在区域j建设
            x: 充电桩数量数组，形状为(m,)，x[j]≥0表示区域j的充电桩数量
            y: 用户分配矩阵，形状为(n, m)，y[i,j]≥0表示楼栋i分配到区域j的用户人数
            
        Returns:
            float: 净收益值 = 总收益 - 总成本
            
        注意: x参数在本函数中实际上没有被使用（充电桩数量不影响目标函数），
             但保留此参数以保持接口一致性。
        """
        # 计算总收益: 对每个楼栋i和区域j，收益 = p_i * y_ij，然后求和
        # self.p[:, np.newaxis]将p从(n,)扩展为(n,1)，以便与y(n,m)进行广播运算
        total_revenue = np.sum(self.p[:, np.newaxis] * y)
        
        # 计算总成本: 对所有建设的区域（z[j]=1），累加建设成本c_j
        total_cost = np.sum(self.c * z)
        
        # 净收益 = 总收益 - 总成本
        net_profit = total_revenue - total_cost
        
        return net_profit
    
    def is_feasible(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """
        检查解的可行性
        
        验证解是否满足所有约束条件：
        1. 变量类型约束
        2. 覆盖关系约束: y_ij ≤ D_i * a_ij * z_j
        3. 需求约束: Σ_j y_ij ≤ D_i
        4. 容量约束: Σ_i y_ij ≤ x_j
        5. 容量上限约束: 0 ≤ x_j ≤ U_j * z_j
        
        Args:
            z: 建设决策数组，形状为(m,)，应为0-1整数
            x: 充电桩数量数组，形状为(m,)，应为非负
            y: 用户分配矩阵，形状为(n, m)，应为非负
            
        Returns:
            Tuple[bool, str]: 
                - is_feasible: 布尔值，True表示可行，False表示不可行
                - error_message: 字符串，如果不可行则说明违反的约束，否则为"可行"
        """
        # 约束1: 检查z是否为0-1变量
        # z中的所有元素必须为0或1
        if not np.all((z == 0) | (z == 1)):
            return False, "z必须为0-1变量"
        
        # 约束2: 检查x是否非负
        if np.any(x < 0):
            return False, "x必须非负"
        
        # 约束3: 检查y是否非负
        if np.any(y < 0):
            return False, "y必须非负"
        
        # 约束4: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        # 含义：只有区域j被建设（z_j=1）且可以覆盖楼栋i（a_ij=1）时，才能分配用户
        # 并且分配的用户数不能超过楼栋i的需求D_i
        for i in range(self.n):
            for j in range(self.m):
                # 计算允许的最大分配量：如果a_ij=1且z_j=1，则为D_i；否则为0
                max_allocation = self.D[i] * self.a[i, j] * z[j]
                # 使用1e-6容差来处理浮点数精度问题
                if y[i, j] > max_allocation + 1e-6:
                    return False, f"违反覆盖关系约束: y[{i},{j}] > D[{i}] * a[{i},{j}] * z[{j}]"
        
        # 约束5: 需求约束 Σ_j y_ij ≤ D_i
        # 含义：每栋楼被分配的用户总数不能超过其潜在需求
        for i in range(self.n):
            total_assigned = np.sum(y[i, :])  # 楼栋i被分配到所有区域的总用户数
            if total_assigned > self.D[i] + 1e-6:
                return False, f"违反需求约束: 楼栋{i}的用户分配超过需求"
        
        # 约束6: 容量约束 Σ_i y_ij ≤ x_j
        # 含义：区域j服务的用户总数不能超过其设置的充电桩数量
        for j in range(self.m):
            total_served = np.sum(y[:, j])  # 区域j服务的所有楼栋的总用户数
            if total_served > x[j] + 1e-6:
                return False, f"违反容量约束: 区域{j}的服务量超过容量"
        
        # 约束7: 容量上限约束 0 ≤ x_j ≤ U_j * z_j
        # 含义：充电桩数量必须非负，且只有在区域被建设时才能设置充电桩
        # 如果z_j=0，则x_j必须为0；如果z_j=1，则x_j不能超过U_j
        for j in range(self.m):
            if x[j] < 0:
                return False, f"违反非负约束: x[{j}] < 0"
            max_capacity = self.U[j] * z[j]  # 如果z_j=0则为0，如果z_j=1则为U_j
            if x[j] > max_capacity + 1e-6:
                return False, f"违反容量上限约束: x[{j}] > U[{j}] * z[{j}]"
        
        # 所有约束都满足
        return True, "可行"
