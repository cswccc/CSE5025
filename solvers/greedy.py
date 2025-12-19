"""
贪心算法求解器
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver


class GreedySolver(BaseSolver):
    """贪心算法求解器"""
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用贪心算法求解
        
        贪心策略：
        1. 计算每个区域的"性价比"（潜在收益/建设成本）
        2. 按性价比从高到低选择区域
        3. 对选中的区域，贪心分配用户
        
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        z = np.zeros(self.m, dtype=int)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        # 计算每个区域的潜在收益（假设完全利用）
        region_profits = np.zeros(self.m)
        for j in range(self.m):
            # 计算区域j能覆盖的所有楼栋的总潜在收益
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            max_demand = min(sum(self.D[covered_buildings]), self.U[j])
            # 粗略估算：假设分配的是高收益用户
            # 简单估算：平均收益 * 容量上限
            avg_profit = np.mean(self.p[covered_buildings]) if len(covered_buildings) > 0 else 0
            region_profits[j] = avg_profit * min(sum(self.D[covered_buildings]), self.U[j])
        
        # 计算性价比（潜在收益/成本）
        cost_benefit_ratio = np.zeros(self.m)
        for j in range(self.m):
            if self.c[j] > 0:
                cost_benefit_ratio[j] = region_profits[j] / self.c[j]
            else:
                cost_benefit_ratio[j] = float('inf')
        
        # 按性价比排序
        sorted_regions = np.argsort(-cost_benefit_ratio)
        
        # 贪心选择区域
        for j in sorted_regions:
            # 尝试添加区域j，检查是否能提升目标函数
            test_z = z.copy()
            test_z[j] = 1
            
            test_x, test_y, test_obj = self._solve_given_z(test_z, x.copy(), y.copy())
            
            # 如果目标函数提升，则选择该区域
            current_obj = self.calculate_objective(z, x, y)
            if test_obj > current_obj:
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
        
        Args:
            z: 建设决策
            x_init: 初始x值
            y_init: 初始y值
            
        Returns:
            (x, y, objective)
        """
        x = x_init.copy()
        y = y_init.copy()
        
        # 计算剩余需求和剩余容量
        remaining_demand = self.D - np.sum(y, axis=1)
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j] - np.sum(y[:, j])
        
        # 计算每个(楼栋i, 区域j)对的收益
        profit_matrix = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1 and remaining_demand[i] > 0 and remaining_capacity[j] > 0:
                    profit_matrix[i, j] = self.p[i]
        
        # 贪心分配：按收益从高到低
        while True:
            # 找到收益最高的分配
            max_profit = -1
            best_i, best_j = -1, -1
            
            for i in range(self.n):
                for j in range(self.m):
                    if profit_matrix[i, j] > max_profit:
                        max_profit = profit_matrix[i, j]
                        best_i, best_j = i, j
            
            if max_profit <= 0:
                break
            
            # 执行分配
            amount = min(remaining_demand[best_i], remaining_capacity[best_j])
            if amount > 0:
                y[best_i, best_j] += amount
                remaining_demand[best_i] -= amount
                remaining_capacity[best_j] -= amount
                profit_matrix[best_i, best_j] = 0  # 不能再分配
            else:
                profit_matrix[best_i, best_j] = 0
        
        # 设置充电桩数量
        for j in range(self.m):
            if z[j] == 1:
                x[j] = max(x[j], np.sum(y[:, j]))
        
        objective = self.calculate_objective(z, x, y)
        
        return x, y, objective
