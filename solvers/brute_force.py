"""
暴力枚举法求解器

本模块实现了暴力枚举法来求解充电桩覆盖收益最大化问题。

算法思想：
    暴力枚举法通过枚举所有可能的建设决策组合来找到全局最优解。
    对于每种建设方案，使用线性规划求解最优的用户分配。
    
    时间复杂度：O(2^m × LP_time)，其中m是区域数量
    
优点：
    - 保证找到全局最优解（对于枚举的范围内）
    - 结果可靠
    
缺点：
    - 时间复杂度指数级增长，仅适用于小规模问题（m ≤ 15）
    - 计算时间随问题规模急剧增加

适用场景：
    - 小规模问题（区域数量 ≤ 15）
    - 需要精确最优解的场合
    - 作为基准解验证其他算法的正确性
"""

import numpy as np
import time
from itertools import product
from typing import Dict, Tuple
from .base_solver import BaseSolver


class BruteForceSolver(BaseSolver):
    """
    暴力枚举法求解器
    
    通过枚举所有可能的建设决策组合（2^m种），对每种组合求解最优的用户分配，
    最终选择目标函数值最大的解。
    """
    
    def solve(self, time_limit: float = 300.0) -> Tuple[Dict, float]:
        """
        使用暴力枚举法求解
        
        算法流程：
        1. 枚举所有可能的建设决策z（共2^m种组合）
        2. 对每种z，使用线性规划求解最优的x和y
        3. 计算每种方案的目标函数值
        4. 返回目标函数值最大的解
        
        Args:
            time_limit: 时间限制（秒），如果超过此时间则停止枚举
            
        Returns:
            Tuple[Dict, float]: 
                - solution: 全局最优解字典，包含z, x, y
                - objective_value: 最优目标函数值
                
        注意：
            - 当m > 15时，只枚举前1000种方案（因为2^15 = 32768已经很大）
            - 如果达到时间限制，返回当前找到的最优解
        """
        start_time = time.time()
        
        # 初始化最佳解
        best_z = None
        best_x = None
        best_y = None
        best_obj = float('-inf')  # 初始化为负无穷
        
        # 步骤1: 生成所有可能的建设决策组合
        # 每个区域都有建设和不建设两种选择，共2^m种组合
        if self.m > 15:
            # 如果区域数太多，只枚举前1000种（避免计算时间过长）
            print(f"警告: 区域数量 {self.m} 太大，暴力枚举不可行（需要枚举 2^{self.m} 种方案）")
            print("仅枚举前1000种方案...")
            z_combinations = [np.zeros(self.m, dtype=int)]  # 首先添加全0组合（不建设任何区域）
            count = 0
            # 使用itertools.product生成所有(0,1)组合
            for z in product([0, 1], repeat=self.m):
                if count >= 1000:
                    break
                z_combinations.append(np.array(z))
                count += 1
        else:
            # 对于小规模问题，枚举所有2^m种组合
            z_combinations = [np.array(z) for z in product([0, 1], repeat=self.m)]
        
        total_combinations = len(z_combinations)
        print(f"总共需要枚举 {total_combinations} 种建设方案...")
        
        # 步骤2: 对每种建设方案，求解最优的用户分配
        for idx, z in enumerate(z_combinations):
            # 检查是否超过时间限制
            if time.time() - start_time > time_limit:
                print(f"达到时间限制，已检查 {idx}/{total_combinations} 种方案")
                break
            
            # 每100次输出一次进度
            if idx % 100 == 0:
                print(f"进度: {idx}/{total_combinations}, 当前最优: {best_obj:.2f}")
            
            # 对当前的z，求解最优的x和y（使用线性规划或贪心方法）
            x, y, obj = self._solve_given_z(z)
            
            # 更新最佳解
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
        
        这是一个线性规划子问题。给定z后，需要确定x和y以最大化目标函数。
        子问题的目标函数：max Σ_i Σ_j (p_i * y_ij) - Σ_j (c_j * z_j)
        由于z是给定的，第二项是常数，因此只需最大化第一项。
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]∈{0,1}
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - x: 最优充电桩数量数组
                - y: 最优用户分配矩阵
                - objective: 目标函数值（包含常数项c_j*z_j）
                
        注意：
            - 优先使用线性规划求解（精确方法）
            - 如果PuLP不可用，则使用贪心方法（近似方法）
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
        使用线性规划求解给定z的最优x和y（精确方法）
        
        将子问题建模为线性规划问题：
        目标：max Σ_i Σ_j (p_i * y_ij)
        约束：
            - y_ij ≤ D_i * a_ij * z_j  (覆盖关系)
            - Σ_j y_ij ≤ D_i           (需求约束)
            - Σ_i y_ij ≤ x_j           (容量约束)
            - 0 ≤ x_j ≤ U_j * z_j      (容量上限)
        
        Args:
            z: 建设决策数组，形状为(m,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 最优的x, y和目标函数值
        """
        import pulp
        
        # 创建线性规划问题（最大化问题）
        prob = pulp.LpProblem("SubProblem", pulp.LpMaximize)
        
        # 定义决策变量
        # x_j: 区域j的充电桩数量，范围[0, U_j*z_j]
        x = [pulp.LpVariable(f'x_{j}', lowBound=0, upBound=self.U[j] * z[j], cat='Continuous')
             for j in range(self.m)]
        # y_ij: 楼栋i分配到区域j的用户数，非负
        y = [[pulp.LpVariable(f'y_{i}_{j}', lowBound=0, cat='Continuous')
              for j in range(self.m)] for i in range(self.n)]
        
        # 目标函数: max Σ_i Σ_j (p_i * y_ij)
        # 注意：Σ_j (c_j * z_j)是常数，不影响优化
        prob += pulp.lpSum([self.p[i] * y[i][j] 
                           for i in range(self.n) for j in range(self.m)])
        
        # 约束1: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        # 如果a_ij=1且z_j=1，则y_ij ≤ D_i；否则y_ij = 0
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    prob += y[i][j] <= self.D[i]
                else:
                    prob += y[i][j] == 0
        
        # 约束2: 需求约束 Σ_j y_ij ≤ D_i
        # 每栋楼被分配的用户总数不能超过其需求
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: 容量约束 Σ_i y_ij ≤ x_j
        # 区域j服务的用户总数不能超过其充电桩数量
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 约束4: 容量上限约束 x_j ≤ U_j * z_j
        # 已在变量定义中通过upBound处理
        
        # 使用CBC求解器求解
        solver = pulp.PULP_CBC_CMD(msg=0)  # msg=0表示不输出求解过程
        prob.solve(solver)
        
        # 提取解（处理可能的None值）
        x_sol = np.array([pulp.value(x[j]) if pulp.value(x[j]) is not None else 0.0
                         for j in range(self.m)])
        y_sol = np.array([[pulp.value(y[i][j]) if pulp.value(y[i][j]) is not None else 0.0
                          for j in range(self.m)] for i in range(self.n)])
        
        # 计算完整的目标函数值（包括成本项）
        objective = self.calculate_objective(z, x_sol, y_sol)
        
        return x_sol, y_sol, objective
    
    def _solve_given_z_greedy(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用贪心方法求解给定z的x和y（备选方法）
        
        当PuLP不可用时，使用贪心方法作为近似求解。
        贪心策略：按单位收益从高到低分配用户。
        
        Args:
            z: 建设决策数组，形状为(m,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 近似最优的x, y和目标函数值
            
        注意：
            此方法不能保证找到最优解，建议安装PuLP以获得精确结果。
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
