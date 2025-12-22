"""
混合整数线性规划(MILP)求解器

本模块实现了混合整数线性规划方法来求解充电桩覆盖收益最大化问题。

算法思想：
    将问题建模为标准的混合整数线性规划模型，使用PuLP库调用CBC等求解器求解。
    MILP可以保证找到全局最优解（如果求解器收敛）。
    
    模型特点：
    - z_j: 二进制变量（0-1变量），表示是否在区域j建设
    - x_j: 整数变量，表示区域j的充电桩数量
    - y_ij: 整数变量，表示楼栋i分配到区域j的用户数
    
优点：
    - 保证找到全局最优解
    - 可以给出最优性证明
    - 适用于中小规模问题
    
缺点：
    - 对于大规模问题可能求解时间很长
    - 需要安装PuLP库和底层求解器（如CBC）

依赖：
    - PuLP: Python混合整数规划库
    - CBC: 开源混合整数规划求解器（通过PuLP调用）
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
    """
    MILP求解器
    
    将问题建模为混合整数线性规划，使用PuLP库调用底层求解器（如CBC）求解。
    这是精确求解方法，可以找到全局最优解。
    """
    
    def __init__(self, instance: Dict, solver_name: str = 'PULP_CBC_CMD'):
        """
        初始化MILP求解器
        
        Args:
            instance: 问题实例字典，包含所有输入参数
            solver_name: 求解器名称，可选值：
                - 'PULP_CBC_CMD': CBC求解器（默认，开源）
                - 其他PuLP支持的求解器名称
                
        Raises:
            ImportError: 如果PuLP未安装
        """
        super().__init__(instance)
        if not PULP_AVAILABLE:
            raise ImportError("PuLP未安装，请运行: pip install pulp")
        self.solver_name = solver_name
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用MILP求解
        
        算法流程：
        1. 创建MILP问题模型
        2. 定义决策变量（z: 二进制，x和y: 整数）
        3. 设置目标函数和约束条件
        4. 调用求解器求解
        5. 提取并返回解
        
        Returns:
            Tuple[Dict, float]: 
                - solution: 最优解字典，包含z, x, y
                - objective_value: 最优目标函数值
                
        注意：
            - 如果求解失败，返回零解
            - 求解时间取决于问题规模和求解器性能
        """
        start_time = time.time()
        
        # 创建MILP问题（最大化问题）
        prob = pulp.LpProblem("ChargingStationOptimization", pulp.LpMaximize)
        
        # 步骤1: 定义决策变量
        # z_j: 二进制变量，表示是否在区域j建设（0或1）
        z = [pulp.LpVariable(f'z_{j}', cat='Binary') for j in range(self.m)]
        
        # x_j: 整数变量，表示区域j的充电桩数量，范围[0, U_j]
        # 注意：实际约束 x_j ≤ U_j * z_j 在后面添加
        x = [pulp.LpVariable(f'x_{j}', lowBound=0, upBound=self.U[j], cat='Integer') 
             for j in range(self.m)]
        
        # y_ij: 整数变量，表示楼栋i分配到区域j的用户数，非负
        y = [[pulp.LpVariable(f'y_{i}_{j}', lowBound=0, cat='Integer') 
              for j in range(self.m)] for i in range(self.n)]
        
        # 步骤2: 设置目标函数
        # max: Σ_i Σ_j (p_i * y_ij) - Σ_j (c_j * z_j)
        # 第一部分是总收益，第二部分是总成本
        prob += (pulp.lpSum([self.p[i] * y[i][j] 
                            for i in range(self.n) for j in range(self.m)]) -
                pulp.lpSum([self.c[j] * z[j] for j in range(self.m)]))
        
        # 步骤3: 添加约束条件
        # 约束1: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        # 如果a_ij=1，使用大M方法：y_ij ≤ D_i * z_j
        # 如果a_ij=0，则y_ij = 0
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1:
                    # 大M方法：当z_j=0时，y_ij必须为0；当z_j=1时，y_ij ≤ D_i
                    prob += y[i][j] <= self.D[i] * z[j]
                else:
                    # 如果区域j不能覆盖楼栋i，则y_ij必须为0
                    prob += y[i][j] == 0
        
        # 约束2: 需求约束 Σ_j y_ij ≤ D_i
        # 每栋楼被分配的用户总数不能超过其需求
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: 容量约束 Σ_i y_ij ≤ x_j
        # 区域j服务的用户总数不能超过其充电桩数量
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 约束4: 容量上限和建设约束 x_j ≤ U_j * z_j
        # 大M方法：当z_j=0时，x_j必须为0；当z_j=1时，x_j ≤ U_j
        for j in range(self.m):
            prob += x[j] <= self.U[j] * z[j]
        
        # 求解
        try:
            if self.solver_name == 'PULP_CBC_CMD':
                solver = pulp.PULP_CBC_CMD(msg=0)
            else:
                solver = pulp.getSolver(self.solver_name)
            
            prob.solve(solver)
            
            # 提取解并转换为整数
            z_sol = np.array([int(pulp.value(z[j])) if pulp.value(z[j]) is not None else 0 
                             for j in range(self.m)], dtype=int)
            x_sol = np.array([int(round(pulp.value(x[j]))) if pulp.value(x[j]) is not None else 0 
                             for j in range(self.m)], dtype=int)
            y_sol = np.array([[int(round(pulp.value(y[i][j]))) if pulp.value(y[i][j]) is not None else 0 
                              for j in range(self.m)] for i in range(self.n)], dtype=int)
            
            objective_value = pulp.value(prob.objective)
            if objective_value is None:
                objective_value = self.calculate_objective(z_sol, x_sol, y_sol)
            
        except Exception as e:
            print(f"MILP求解失败: {e}")
            # 返回零解（整数）
            z_sol = np.zeros(self.m, dtype=int)
            x_sol = np.zeros(self.m, dtype=int)
            y_sol = np.zeros((self.n, self.m), dtype=int)
            objective_value = 0.0
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z_sol.tolist(),
            'x': x_sol.tolist(),
            'y': y_sol.tolist()
        }
        self.best_objective = objective_value
        
        return self.best_solution, self.best_objective
