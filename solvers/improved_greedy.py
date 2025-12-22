"""
改进的贪心算法求解器

针对问题特性优化：
1. 单位收益统一：所有楼栋用户价值相同，分配时优先考虑容量利用率
2. 成本与容量近似成正比：计算真实成本效益比
3. 使用多种贪心策略并选择最佳
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver


class ImprovedGreedySolver(BaseSolver):
    """
    改进的贪心算法求解器
    
    利用问题的特殊结构（统一收益、成本与容量成正比）进行优化。
    使用多种贪心策略，选择最优结果。
    """
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用改进的贪心算法求解
        
        策略1：按容量利用率选择（适合统一收益的情况）
        策略2：按成本效益比选择
        策略3：按总收益/成本比选择
        
        Returns:
            Tuple[Dict, float]: 最佳解和目标函数值
        """
        start_time = time.time()
        
        best_solution = None
        best_objective = float('-inf')
        
        # 策略1：基于容量利用率的贪心（利用统一收益特性）
        sol1, obj1 = self._greedy_by_capacity_utilization()
        if obj1 > best_objective:
            best_objective = obj1
            best_solution = sol1
        
        # 策略2：基于成本效益比的贪心
        sol2, obj2 = self._greedy_by_cost_benefit()
        if obj2 > best_objective:
            best_objective = obj2
            best_solution = sol2
        
        # 策略3：逐步构建（增量添加最佳区域）
        sol3, obj3 = self._greedy_incremental()
        if obj3 > best_objective:
            best_objective = obj3
            best_solution = sol3
        
        self.solve_time = time.time() - start_time
        self.best_solution = best_solution
        self.best_objective = best_objective
        
        return self.best_solution, self.best_objective
    
    def _greedy_by_capacity_utilization(self) -> Tuple[Dict, float]:
        """
        策略1：基于容量利用率的贪心
        
        由于单位收益统一，优先选择能够高效利用容量的区域。
        计算每个区域的"容量效率" = 可覆盖需求 / 容量
        """
        z = np.zeros(self.m, dtype=int)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        # 计算每个区域的容量效率
        # 效率 = min(可覆盖的总需求, 容量) / 容量
        efficiency = np.zeros(self.m)
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            if len(covered_buildings) > 0:
                total_demand = sum(self.D[covered_buildings])
                utilizable = min(total_demand, self.U[j])
                efficiency[j] = utilizable / self.U[j] if self.U[j] > 0 else 0
            else:
                efficiency[j] = 0
        
        # 计算每个区域的成本效益比（考虑容量效率）
        # 效益 = 单位收益 * 可利用容量 / 成本
        benefit_ratio = np.zeros(self.m)
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            if len(covered_buildings) > 0 and self.c[j] > 0:
                total_demand = sum(self.D[covered_buildings])
                utilizable = min(total_demand, self.U[j])
                # 由于单位收益统一，可以用容量利用率来估算效益
                benefit = self.p[0] * utilizable  # 统一收益
                benefit_ratio[j] = benefit / self.c[j]
            else:
                benefit_ratio[j] = 0
        
        # 按效益比排序
        sorted_regions = np.argsort(-benefit_ratio)
        
        # 贪心添加区域
        for j in sorted_regions:
            test_z = z.copy()
            test_z[j] = 1
            test_x, test_y, test_obj = self._solve_given_z_optimal(test_z, x.copy(), y.copy())
            current_obj = self.calculate_objective(z, x, y)
            if test_obj > current_obj:
                z[j] = 1
                x = test_x.copy()
                y = test_y.copy()
        
        solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        objective = self.calculate_objective(z, x, y)
        return solution, objective
    
    def _greedy_by_cost_benefit(self) -> Tuple[Dict, float]:
        """
        策略2：基于成本效益比的贪心
        
        计算每个区域的真实成本效益比，考虑容量限制。
        """
        z = np.zeros(self.m, dtype=int)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        # 计算每个区域的潜在净收益
        net_profits = np.zeros(self.m)
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            if len(covered_buildings) > 0:
                total_demand = sum(self.D[covered_buildings])
                utilizable = min(total_demand, self.U[j])
                # 潜在收益（由于统一收益，直接用容量*单位收益）
                potential_revenue = self.p[0] * utilizable
                # 净收益 = 收益 - 成本
                net_profits[j] = potential_revenue - self.c[j]
            else:
                net_profits[j] = -self.c[j]  # 负数，不划算
        
        # 按净收益排序（只考虑正净收益的区域）
        sorted_regions = np.argsort(-net_profits)
        
        for j in sorted_regions:
            if net_profits[j] <= 0:
                break  # 后续区域都是负收益，不需要考虑
            test_z = z.copy()
            test_z[j] = 1
            test_x, test_y, test_obj = self._solve_given_z_optimal(test_z, x.copy(), y.copy())
            current_obj = self.calculate_objective(z, x, y)
            if test_obj > current_obj:
                z[j] = 1
                x = test_x.copy()
                y = test_y.copy()
        
        solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        objective = self.calculate_objective(z, x, y)
        return solution, objective
    
    def _greedy_incremental(self) -> Tuple[Dict, float]:
        """
        策略3：增量贪心
        
        每一步选择能够带来最大边际收益的区域。
        """
        z = np.zeros(self.m, dtype=int)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        remaining_regions = list(range(self.m))
        
        while len(remaining_regions) > 0:
            best_j = -1
            best_margin = -float('inf')
            best_x_new = None
            best_y_new = None
            
            current_obj = self.calculate_objective(z, x, y)
            
            # 尝试每个剩余区域
            for j in remaining_regions:
                test_z = z.copy()
                test_z[j] = 1
                test_x, test_y, test_obj = self._solve_given_z_optimal(test_z, x.copy(), y.copy())
                margin = test_obj - current_obj  # 边际收益
                
                if margin > best_margin:
                    best_margin = margin
                    best_j = j
                    best_x_new = test_x.copy()
                    best_y_new = test_y.copy()
            
            # 如果边际收益为正，添加该区域
            if best_margin > 0 and best_j >= 0:
                z[best_j] = 1
                x = best_x_new
                y = best_y_new
                remaining_regions.remove(best_j)
            else:
                break  # 没有正边际收益的区域了
        
        solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        objective = self.calculate_objective(z, x, y)
        return solution, objective
    
    def _solve_given_z_optimal(self, z: np.ndarray, x_init: np.ndarray = None, y_init: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        在给定建设决策z的情况下，最优分配用户
        
        由于单位收益统一，这是一个最大流/最小费用流类型的子问题。
        可以使用贪心方法快速求解（按需求分配，优先满足需求大的楼栋）。
        
        Args:
            z: 建设决策
            x_init: 初始x值
            y_init: 初始y值
            
        Returns:
            (x, y, objective)
        """
        if x_init is None:
            x_init = np.zeros(self.m, dtype=float)
        if y_init is None:
            y_init = np.zeros((self.n, self.m), dtype=float)
            
        x = x_init.copy()
        y = y_init.copy()
        
        # 计算剩余需求和剩余容量
        remaining_demand = self.D - np.sum(y, axis=1)
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j] - np.sum(y[:, j])
        
        # 由于单位收益统一，可以简单地按需求大小排序分配
        # 优先满足需求大的楼栋
        
        # 构建所有可行的(楼栋, 区域)对，按需求降序排序
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                if (self.a[i, j] == 1 and z[j] == 1 and 
                    remaining_demand[i] > 0 and remaining_capacity[j] > 0):
                    allocations.append((i, j, self.D[i]))  # 使用总需求作为优先级
        
        # 按需求降序排序
        allocations.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心分配
        for i, j, _ in allocations:
            if remaining_demand[i] <= 0 or remaining_capacity[j] <= 0:
                continue
            amount = min(remaining_demand[i], remaining_capacity[j])
            y[i, j] += amount
            remaining_demand[i] -= amount
            remaining_capacity[j] -= amount
        
        # 设置充电桩数量
        for j in range(self.m):
            if z[j] == 1:
                x[j] = max(x[j], np.sum(y[:, j]))
        
        objective = self.calculate_objective(z, x, y)
        return x, y, objective
