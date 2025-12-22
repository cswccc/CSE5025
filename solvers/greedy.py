"""
贪心算法求解器

本模块实现了贪心算法来求解充电桩覆盖收益最大化问题。

算法思想：
    贪心算法是一种启发式方法，每一步都做出当前看起来最优的选择。
    本实现采用以下策略：
    1. 计算每个区域的"性价比"（潜在收益/建设成本）
    2. 按性价比从高到低排序区域
    3. 依次尝试添加区域，如果目标函数提升则保留
    4. 对已选中的区域，使用贪心方法分配用户（按单位收益从高到低）

优点：
    - 运行速度快
    - 实现简单
    - 对于大多数问题能得到较好的解

缺点：
    - 不能保证全局最优
    - 可能陷入局部最优解
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver


class GreedySolver(BaseSolver):
    """
    贪心算法求解器
    
    使用贪心策略逐步选择建设区域，并对每个区域贪心分配用户。
    适用于快速获取可行解或作为其他算法的初始解。
    """
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用贪心算法求解
        
        算法流程：
        1. 计算每个区域的潜在收益和性价比
        2. 按性价比从高到低排序区域
        3. 依次尝试添加区域，如果目标函数提升则保留
        4. 对最终选中的区域集合，使用贪心方法分配用户
        
        Returns:
            Tuple[Dict, float]: 
                - solution: 解字典，包含z, x, y
                - objective_value: 目标函数值（净收益）
        """
        start_time = time.time()
        
        # 初始化决策变量
        z = np.zeros(self.m, dtype=int)  # 建设决策，初始全为0
        x = np.zeros(self.m, dtype=int)  # 充电桩数量，初始全为0（整数）
        y = np.zeros((self.n, self.m), dtype=int)  # 用户分配，初始全为0（整数）
        
        # 步骤1: 计算每个区域的潜在收益
        # 潜在收益 = 平均单位收益 × min(覆盖楼栋的总需求, 区域容量上限)
        region_profits = np.zeros(self.m)
        for j in range(self.m):
            # 找出区域j能覆盖的所有楼栋
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            # 计算这些楼栋的总需求，但不能超过区域容量上限
            max_demand = min(sum(self.D[covered_buildings]), self.U[j])
            # 计算覆盖楼栋的平均单位收益
            avg_profit = np.mean(self.p[covered_buildings]) if len(covered_buildings) > 0 else 0
            # 潜在收益 = 平均收益 × 最大服务量
            region_profits[j] = avg_profit * min(sum(self.D[covered_buildings]), self.U[j])
        
        # 步骤2: 计算性价比（潜在收益/建设成本）
        # 性价比越高，说明该区域越"划算"
        cost_benefit_ratio = np.zeros(self.m)
        for j in range(self.m):
            if self.c[j] > 0:
                cost_benefit_ratio[j] = region_profits[j] / self.c[j]
            else:
                # 如果成本为0，设为无穷大（优先选择）
                cost_benefit_ratio[j] = float('inf')
        
        # 步骤3: 按性价比从高到低排序区域
        # argsort(-ratio)表示降序排列后的索引
        sorted_regions = np.argsort(-cost_benefit_ratio)
        
        # 步骤4: 贪心选择区域
        # 依次尝试添加每个区域，如果目标函数提升则保留
        for j in sorted_regions:
            # 尝试添加区域j
            test_z = z.copy()
            test_z[j] = 1
            
            # 在包含区域j的情况下，重新分配用户
            test_x, test_y, test_obj = self._solve_given_z(test_z, x.copy(), y.copy())
            
            # 比较添加区域j前后的目标函数值
            current_obj = self.calculate_objective(z, x, y)
            if test_obj > current_obj:
                # 如果目标函数提升，则保留区域j
                z[j] = 1
                x = test_x.copy()
                y = test_y.copy()
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        self.best_objective = self.calculate_objective(z, x, y)
        
        return self.best_solution, self.best_objective
    
    def _solve_given_z(self, z: np.ndarray, x_init: np.ndarray, y_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        在给定建设决策z的情况下，贪心分配用户
        
        这是一个子问题：已知哪些区域被建设，如何最优地分配用户。
        使用贪心策略：优先将用户分配到单位收益最高的区域。
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]=1表示区域j被建设
            x_init: 初始充电桩数量数组，形状为(m,)
            y_init: 初始用户分配矩阵，形状为(n, m)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - x: 更新后的充电桩数量数组
                - y: 更新后的用户分配矩阵
                - objective: 目标函数值
        """
        x = x_init.copy()
        y = y_init.copy()
        
        # 计算剩余需求和剩余容量
        # remaining_demand[i]: 楼栋i还未被分配的用户数
        remaining_demand = self.D - np.sum(y, axis=1)
        # remaining_capacity[j]: 区域j还能服务的用户数
        remaining_capacity = np.zeros(self.m, dtype=int)
        for j in range(self.m):
            if z[j] == 1:  # 只考虑已建设的区域
                remaining_capacity[j] = self.U[j] - np.sum(y[:, j])
        
        # 构建收益矩阵：profit_matrix[i,j]表示楼栋i分配到区域j的单位收益
        # 只有在满足条件时才设置收益值：区域j已建设、可以覆盖楼栋i、还有剩余需求和容量
        profit_matrix = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if (self.a[i, j] == 1 and z[j] == 1 and 
                    remaining_demand[i] > 0 and remaining_capacity[j] > 0):
                    profit_matrix[i, j] = self.p[i]  # 单位收益即为p_i
        
        # 贪心分配：反复选择收益最高的分配，直到无法继续分配
        while True:
            # 找到收益最高的(楼栋,区域)对
            max_profit = -1
            best_i, best_j = -1, -1
            
            for i in range(self.n):
                for j in range(self.m):
                    if profit_matrix[i, j] > max_profit:
                        max_profit = profit_matrix[i, j]
                        best_i, best_j = i, j
            
            # 如果没有可分配的，退出循环
            if max_profit <= 0:
                break
            
            # 执行分配：分配尽可能多的用户（受剩余需求和容量限制）
            amount = min(remaining_demand[best_i], remaining_capacity[best_j])
            if amount > 0:
                y[best_i, best_j] += amount  # 增加分配量
                remaining_demand[best_i] -= amount  # 更新剩余需求
                remaining_capacity[best_j] -= amount  # 更新剩余容量
                profit_matrix[best_i, best_j] = 0  # 标记为已处理
            else:
                # 如果无法分配，也标记为已处理
                profit_matrix[best_i, best_j] = 0
        
        # 设置充电桩数量：每个已建设的区域，充电桩数至少等于实际服务的用户数
        for j in range(self.m):
            if z[j] == 1:
                # 取当前值和实际需要值的最大值（考虑之前可能已有分配）
                x[j] = max(x[j], np.sum(y[:, j]))
        
        # 计算目标函数值
        objective = self.calculate_objective(z, x, y)
        
        return x, y, objective
