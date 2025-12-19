"""
混合整数线性规划(MILP)求解器
使用PuLP作为求解器
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("警告: PuLP未安装，MILP求解器不可用。请运行: pip install pulp")


class MILPSolver(BaseSolver):
    """MILP求解器"""
    
    def __init__(self, instance: Dict, solver_name: str = 'PULP_CBC_CMD'):
        """
        初始化MILP求解器
        
        Args:
            instance: 问题实例
            solver_name: 求解器名称 ('PULP_CBC_CMD', 'PULP_CBC_CMD', 等)
        """
        super().__init__(instance)
        if not PULP_AVAILABLE:
            raise ImportError("PuLP未安装，请运行: pip install pulp")
        self.solver_name = solver_name
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用MILP求解
        
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        # 创建问题
        prob = pulp.LpProblem("ChargingStationOptimization", pulp.LpMaximize)
        
        # 决策变量
        # z_j: 是否在区域j建设 (0-1)
        z = [pulp.LpVariable(f'z_{j}', cat='Binary') for j in range(self.m)]
        
        # x_j: 区域j的充电桩数量
        x = [pulp.LpVariable(f'x_{j}', lowBound=0, upBound=self.U[j], cat='Continuous') 
             for j in range(self.m)]
        
        # y_ij: 楼栋i分配到区域j的用户数
        y = [[pulp.LpVariable(f'y_{i}_{j}', lowBound=0, cat='Continuous') 
              for j in range(self.m)] for i in range(self.n)]
        
        # 目标函数: max sum_i sum_j p_i * y_ij - sum_j c_j * z_j
        prob += (pulp.lpSum([self.p[i] * y[i][j] 
                            for i in range(self.n) for j in range(self.m)]) -
                pulp.lpSum([self.c[j] * z[j] for j in range(self.m)]))
        
        # 约束1: y_ij <= D_i * a_ij * z_j
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1:
                    # 使用大M方法: y_ij <= D_i * z_j
                    prob += y[i][j] <= self.D[i] * z[j]
                else:
                    prob += y[i][j] == 0
        
        # 约束2: sum_j y_ij <= D_i
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: sum_i y_ij <= x_j
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 约束4: 0 <= x_j <= U_j * z_j
        # 使用大M方法: x_j <= U_j * z_j
        for j in range(self.m):
            prob += x[j] <= self.U[j] * z[j]
        
        # 求解
        try:
            if self.solver_name == 'PULP_CBC_CMD':
                solver = pulp.PULP_CBC_CMD(msg=0)
            else:
                solver = pulp.getSolver(self.solver_name)
            
            prob.solve(solver)
            
            # 提取解
            z_sol = np.array([pulp.value(z[j]) for j in range(self.m)])
            x_sol = np.array([pulp.value(x[j]) for j in range(self.m)])
            y_sol = np.array([[pulp.value(y[i][j]) for j in range(self.m)] 
                             for i in range(self.n)])
            
            # 处理None值
            z_sol = np.nan_to_num(z_sol, nan=0)
            x_sol = np.nan_to_num(x_sol, nan=0)
            y_sol = np.nan_to_num(y_sol, nan=0)
            
            objective_value = pulp.value(prob.objective)
            if objective_value is None:
                objective_value = self.calculate_objective(z_sol, x_sol, y_sol)
            
        except Exception as e:
            print(f"MILP求解失败: {e}")
            # 返回零解
            z_sol = np.zeros(self.m)
            x_sol = np.zeros(self.m)
            y_sol = np.zeros((self.n, self.m))
            objective_value = 0.0
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z_sol.tolist(),
            'x': x_sol.tolist(),
            'y': y_sol.tolist()
        }
        self.best_objective = objective_value
        
        return self.best_solution, self.best_objective
