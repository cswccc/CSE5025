"""
暴力枚举法求解器
适用于小规模问题
"""

import numpy as np
import time
from itertools import product
from typing import Dict, Tuple
from .base_solver import BaseSolver


class BruteForceSolver(BaseSolver):
    """暴力枚举法求解器"""
    
    def solve(self, time_limit: float = 300.0) -> Tuple[Dict, float]:
        """
        使用暴力枚举法求解
        
        Args:
            time_limit: 时间限制（秒）
            
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        best_z = None
        best_x = None
        best_y = None
        best_obj = float('-inf')
        
        # 枚举所有可能的建设决策 z
        # 对于小规模问题，我们可以枚举所有2^m种建设方案
        if self.m > 15:
            print(f"警告: 区域数量 {self.m} 太大，暴力枚举不可行（需要枚举 2^{self.m} 种方案）")
            print("仅枚举前1000种方案...")
            z_combinations = [np.zeros(self.m, dtype=int)]
            count = 0
            for z in product([0, 1], repeat=self.m):
                if count >= 1000:
                    break
                z_combinations.append(np.array(z))
                count += 1
        else:
            z_combinations = [np.array(z) for z in product([0, 1], repeat=self.m)]
        
        total_combinations = len(z_combinations)
        print(f"总共需要枚举 {total_combinations} 种建设方案...")
        
        for idx, z in enumerate(z_combinations):
            if time.time() - start_time > time_limit:
                print(f"达到时间限制，已检查 {idx}/{total_combinations} 种方案")
                break
            
            if idx % 100 == 0:
                print(f"进度: {idx}/{total_combinations}, 当前最优: {best_obj:.2f}")
            
            # 对于每种建设方案，求解最优的x和y
            x, y, obj = self._solve_given_z(z)
            
            if obj > best_obj:
                best_obj = obj
                best_z = z.copy()
                best_x = x.copy()
                best_y = y.copy()
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': best_z.tolist(),
            'x': best_x.tolist(),
            'y': best_y.tolist()
        }
        self.best_objective = best_obj
        
        return self.best_solution, self.best_objective
    
    def _solve_given_z(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        在给定建设决策z的情况下，求解最优的x和y
        
        这是一个线性规划子问题，应该使用线性规划求解而不是贪心法
        
        Returns:
            (x, y, objective)
        """
        # 尝试使用线性规划求解（如果可用）
        try:
            import pulp
            return self._solve_given_z_lp(z)
        except ImportError:
            # 如果没有PuLP，使用贪心方法作为备选
            return self._solve_given_z_greedy(z)
    
    def _solve_given_z_lp(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用线性规划求解给定z的最优x和y
        """
        import pulp
        
        # 创建线性规划问题
        prob = pulp.LpProblem("SubProblem", pulp.LpMaximize)
        
        # 决策变量
        x = [pulp.LpVariable(f'x_{j}', lowBound=0, upBound=self.U[j] * z[j], cat='Continuous')
             for j in range(self.m)]
        y = [[pulp.LpVariable(f'y_{i}_{j}', lowBound=0, cat='Continuous')
              for j in range(self.m)] for i in range(self.n)]
        
        # 目标函数: max sum_i sum_j p_i * y_ij (z_j和c_j是常数)
        prob += pulp.lpSum([self.p[i] * y[i][j] 
                           for i in range(self.n) for j in range(self.m)])
        
        # 约束1: y_ij <= D_i * a_ij * z_j
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    prob += y[i][j] <= self.D[i]
                else:
                    prob += y[i][j] == 0
        
        # 约束2: sum_j y_ij <= D_i
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: sum_i y_ij <= x_j
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 约束4: x_j <= U_j * z_j (已在变量定义中处理)
        
        # 求解
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        # 提取解
        x_sol = np.array([pulp.value(x[j]) if pulp.value(x[j]) is not None else 0.0
                         for j in range(self.m)])
        y_sol = np.array([[pulp.value(y[i][j]) if pulp.value(y[i][j]) is not None else 0.0
                          for j in range(self.m)] for i in range(self.n)])
        
        # 计算目标函数值（包括成本）
        objective = self.calculate_objective(z, x_sol, y_sol)
        
        return x_sol, y_sol, objective
    
    def _solve_given_z_greedy(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用贪心方法求解给定z的x和y（备选方法）
        """
        # 对于给定的z，我们需要决定x和y
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        # 计算每个(楼栋i, 区域j)对的单位收益
        profit_matrix = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    profit_matrix[i, j] = self.p[i]
        
        # 贪心分配：按收益从高到低分配
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                if profit_matrix[i, j] > 0:
                    allocations.append((i, j, profit_matrix[i, j]))
        
        allocations.sort(key=lambda x: x[2], reverse=True)
        
        # 跟踪剩余需求和剩余容量
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 贪心分配
        for i, j, profit in allocations:
            if remaining_demand[i] <= 0 or remaining_capacity[j] <= 0:
                continue
            
            amount = min(remaining_demand[i], remaining_capacity[j])
            y[i, j] = amount
            remaining_demand[i] -= amount
            remaining_capacity[j] -= amount
        
        # 设置充电桩数量
        for j in range(self.m):
            if z[j] == 1:
                x[j] = np.sum(y[:, j])
        
        # 计算目标函数值
        objective = self.calculate_objective(z, x, y)
        
        return x, y, objective
