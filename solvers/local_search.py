"""
局部搜索求解器

使用局部搜索方法改进解：
1. 从贪心解开始
2. 使用多种邻域操作（添加、删除、交换）
3. 选择最佳改进方向
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver


class LocalSearchSolver(BaseSolver):
    """
    局部搜索求解器
    
    从初始解开始，通过局部搜索改进解。
    适用于在已有解的基础上进行优化。
    """
    
    def __init__(self, instance: Dict, initial_solution: Dict = None, max_iterations: int = 1000):
        """
        初始化局部搜索求解器
        
        Args:
            instance: 问题实例
            initial_solution: 初始解（如果提供），否则使用贪心解
            max_iterations: 最大迭代次数
        """
        super().__init__(instance)
        self.initial_solution = initial_solution
        self.max_iterations = max_iterations
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用局部搜索求解
        
        Returns:
            Tuple[Dict, float]: 改进后的解和目标函数值
        """
        start_time = time.time()
        
        # 获取初始解
        if self.initial_solution is not None:
            z = np.array(self.initial_solution['z'])
            x = np.array(self.initial_solution['x'])
            y = np.array(self.initial_solution['y'])
        else:
            # 使用简单贪心获得初始解
            z, x, y = self._get_initial_solution()
        
        current_obj = self.calculate_objective(z, x, y)
        best_z, best_x, best_y = z.copy(), x.copy(), y.copy()
        best_obj = current_obj
        
        iteration = 0
        no_improve_count = 0
        
        while iteration < self.max_iterations and no_improve_count < 50:
            improved = False
            
            # 尝试所有可能的单步改进
            # 1. 尝试添加一个区域
            for j in range(self.m):
                if z[j] == 0:
                    test_z = z.copy()
                    test_z[j] = 1
                    test_x, test_y, test_obj = self._solve_given_z(test_z)
                    if test_obj > best_obj:
                        best_z, best_x, best_y = test_z.copy(), test_x.copy(), test_y.copy()
                        best_obj = test_obj
                        improved = True
            
            # 2. 尝试删除一个区域
            for j in range(self.m):
                if z[j] == 1:
                    test_z = z.copy()
                    test_z[j] = 0
                    test_x, test_y, test_obj = self._solve_given_z(test_z)
                    if test_obj > best_obj:
                        best_z, best_x, best_y = test_z.copy(), test_x.copy(), test_y.copy()
                        best_obj = test_obj
                        improved = True
            
            # 3. 尝试交换两个区域（删除一个，添加另一个）
            for j1 in range(self.m):
                if z[j1] == 1:
                    for j2 in range(self.m):
                        if z[j2] == 0 and j1 != j2:
                            test_z = z.copy()
                            test_z[j1] = 0
                            test_z[j2] = 1
                            test_x, test_y, test_obj = self._solve_given_z(test_z)
                            if test_obj > best_obj:
                                best_z, best_x, best_y = test_z.copy(), test_x.copy(), test_y.copy()
                                best_obj = test_obj
                                improved = True
            
            if improved:
                z, x, y = best_z.copy(), best_x.copy(), best_y.copy()
                current_obj = best_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            iteration += 1
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': best_z.tolist(),
            'x': best_x.tolist(),
            'y': best_y.tolist()
        }
        self.best_objective = best_obj
        
        return self.best_solution, self.best_objective
    
    def _get_initial_solution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取初始解（简单贪心）"""
        z = np.zeros(self.m, dtype=int)
        
        # 计算每个区域的效益
        benefits = np.zeros(self.m)
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            if len(covered_buildings) > 0:
                total_demand = sum(self.D[covered_buildings])
                utilizable = min(total_demand, self.U[j])
                potential_revenue = self.p[0] * utilizable
                benefits[j] = potential_revenue - self.c[j]
        
        # 按效益排序，选择正效益的区域
        sorted_regions = np.argsort(-benefits)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        for j in sorted_regions:
            if benefits[j] > 0:
                test_z = z.copy()
                test_z[j] = 1
                test_x, test_y, test_obj = self._solve_given_z(test_z)
                current_obj = self.calculate_objective(z, x, y)
                if test_obj > current_obj:
                    z[j] = 1
                    x, y = test_x.copy(), test_y.copy()
        
        return z, x, y
    
    def _solve_given_z(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """在给定z的情况下求解最优的x和y"""
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 由于单位收益统一，按需求大小排序分配
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    allocations.append((i, j, self.D[i]))
        
        allocations.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, _ in allocations:
            if remaining_demand[i] <= 0 or remaining_capacity[j] <= 0:
                continue
            amount = min(remaining_demand[i], remaining_capacity[j])
            y[i, j] = amount
            remaining_demand[i] -= amount
            remaining_capacity[j] -= amount
        
        for j in range(self.m):
            if z[j] == 1:
                x[j] = np.sum(y[:, j])
        
        objective = self.calculate_objective(z, x, y)
        return x, y, objective
