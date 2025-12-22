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
            
            # 使用新的最优分配方法：x直接装满，然后最优分配y
            test_x, test_y, test_obj = self.optimal_assign_given_z(test_z)
            
            # 比较添加区域j前后的目标函数值
            current_obj = self.calculate_objective(z, x, y)
            if test_obj > current_obj:
                # 如果目标函数提升，则保留区域j
                z[j] = 1
                x = test_x.copy()
                y = test_y.copy()
        
        # 步骤5: 对最终选中的区域集合，进行最优分配
        x, y, _ = self.optimal_assign_given_z(z)
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        self.best_objective = self.calculate_objective(z, x, y)
        
        return self.best_solution, self.best_objective
