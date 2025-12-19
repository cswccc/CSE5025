"""
蚁群算法求解器
"""

import numpy as np
import time
import random
from typing import Dict, Tuple, List
from .base_solver import BaseSolver


class AntColonySolver(BaseSolver):
    """蚁群算法求解器"""
    
    def __init__(self, instance: Dict,
                 num_ants: int = 20,
                 max_iterations: int = 100,
                 alpha: float = 1.0,  # 信息素重要程度
                 beta: float = 2.0,   # 启发式信息重要程度
                 rho: float = 0.1,    # 信息素挥发系数
                 q: float = 100.0):   # 信息素强度
        """
        初始化蚁群算法求解器
        
        Args:
            instance: 问题实例
            num_ants: 蚂蚁数量
            max_iterations: 最大迭代次数
            alpha: 信息素重要程度
            beta: 启发式信息重要程度
            rho: 信息素挥发系数
            q: 信息素强度
        """
        super().__init__(instance)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        # 初始化信息素矩阵（区域之间的信息素）
        # 这里我们简化为每个区域的信息素水平
        self.pheromone = np.ones(self.m) * 0.1
        
        # 计算启发式信息（区域j的潜在收益/成本）
        self.heuristic = np.zeros(self.m)
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            if len(covered_buildings) > 0:
                avg_profit = np.mean(self.p[covered_buildings])
                max_demand = min(sum(self.D[covered_buildings]), self.U[j])
                potential_profit = avg_profit * max_demand
                if self.c[j] > 0:
                    self.heuristic[j] = potential_profit / self.c[j]
                else:
                    self.heuristic[j] = 1000.0
            else:
                self.heuristic[j] = 0.0
        
        # 归一化启发式信息
        if np.max(self.heuristic) > 0:
            self.heuristic = self.heuristic / np.max(self.heuristic)
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用蚁群算法求解
        
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        best_solution = None
        best_objective = float('-inf')
        
        for iteration in range(self.max_iterations):
            # 所有蚂蚁构建解
            solutions = []
            objectives = []
            
            for ant in range(self.num_ants):
                z = self._construct_solution()
                x, y, obj = self._decode_individual(z)
                solutions.append(z)
                objectives.append(obj)
                
                if obj > best_objective:
                    best_objective = obj
                    best_solution = z.copy()
            
            # 更新信息素
            self._update_pheromone(solutions, objectives, best_solution, best_objective)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最佳目标值 = {best_objective:.2f}")
        
        # 解码最佳解
        z_best = best_solution
        x_best, y_best, obj_best = self._decode_individual(z_best)
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z_best.tolist(),
            'x': x_best.tolist(),
            'y': y_best.tolist()
        }
        self.best_objective = obj_best
        
        return self.best_solution, self.best_objective
    
    def _construct_solution(self) -> np.ndarray:
        """蚂蚁构建解"""
        z = np.zeros(self.m, dtype=int)
        unselected = list(range(self.m))
        
        while len(unselected) > 0:
            # 计算选择概率
            probabilities = []
            for j in unselected:
                prob = (self.pheromone[j] ** self.alpha) * (self.heuristic[j] ** self.beta)
                probabilities.append(prob)
            
            probabilities = np.array(probabilities)
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            else:
                probabilities = np.ones(len(unselected)) / len(unselected)
            
            # 根据概率选择
            selected_idx = np.random.choice(len(unselected), p=probabilities)
            selected_j = unselected[selected_idx]
            
            # 尝试添加该区域
            test_z = z.copy()
            test_z[selected_j] = 1
            x_test, y_test, obj_test = self._decode_individual(test_z)
            
            # 如果目标函数提升，则选择
            current_obj = self._decode_individual(z)[2]
            if obj_test > current_obj:
                z[selected_j] = 1
            
            unselected.remove(selected_j)
        
        return z
    
    def _decode_individual(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """解码个体"""
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 贪心分配
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    allocations.append((i, j, self.p[i]))
        
        allocations.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, profit in allocations:
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
    
    def _update_pheromone(self, solutions: List[np.ndarray], objectives: List[float],
                         best_solution: np.ndarray, best_objective: float):
        """更新信息素"""
        # 信息素挥发
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # 根据解的优劣更新信息素
        for z, obj in zip(solutions, objectives):
            if obj > 0:
                for j in range(self.m):
                    if z[j] == 1:
                        self.pheromone[j] += self.q * obj / best_objective if best_objective > 0 else 0
        
        # 最佳解额外增强
        if best_objective > 0:
            for j in range(self.m):
                if best_solution[j] == 1:
                    self.pheromone[j] += self.q * 2.0
        
        # 信息素上下界
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
