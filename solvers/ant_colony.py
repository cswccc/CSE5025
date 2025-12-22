"""
蚁群算法求解器

本模块实现了蚁群算法（Ant Colony Optimization, ACO）来求解充电桩覆盖收益最大化问题。

算法思想：
    蚁群算法模拟蚂蚁觅食行为，通过信息素机制来引导搜索。
    算法流程：
    1. 初始化信息素和启发式信息
    2. 每只蚂蚁根据信息素和启发式信息构建解
    3. 根据解的质量更新信息素
    4. 信息素挥发
    5. 重复步骤2-4直到达到最大迭代次数
    
    选择概率：P_j = (τ_j^α * η_j^β) / Σ(τ_k^α * η_k^β)
    其中：τ_j是信息素，η_j是启发式信息，α和β是权重参数
    
优点：
    - 具有良好的全局搜索能力
    - 适合组合优化问题
    - 信息素机制有助于避免过早收敛
    
缺点：
    - 收敛速度可能较慢
    - 参数敏感，需要调优
    - 需要多次迭代才能找到好解
"""

import numpy as np
import time
import random
from typing import Dict, Tuple, List
from .base_solver import BaseSolver


class AntColonySolver(BaseSolver):
    """
    蚁群算法求解器
    
    使用信息素和启发式信息引导蚂蚁构建解，通过信息素更新机制
    来学习最优解的构建模式。
    """
    
    def __init__(self, instance: Dict,
                 num_ants: int = 20,
                 max_iterations: int = 100,
                 alpha: float = 1.0,  # 信息素重要程度
                 beta: float = 2.0,   # 启发式信息重要程度
                 rho: float = 0.1,    # 信息素挥发系数
                 q: float = 100.0,    # 信息素强度
                 early_stop_patience: int = 30):  # 早停耐心值
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
            early_stop_patience: 早停耐心值，连续多少代没有改进则提前终止（默认30次迭代）
        """
        super().__init__(instance)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.early_stop_patience = early_stop_patience
        
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
        
        # 早停机制：记录连续没有改进的迭代次数
        no_improve_count = 0
        
        for iteration in range(self.max_iterations):
            # 所有蚂蚁构建解
            solutions = []
            objectives = []
            improved = False
            
            for ant in range(self.num_ants):
                z = self._construct_solution()
                x, y, obj = self._decode_individual(z)
                solutions.append(z)
                objectives.append(obj)
                
                if obj > best_objective:
                    best_objective = obj
                    best_solution = z.copy()
                    improved = True
            
            # 更新早停计数器
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 更新信息素
            self._update_pheromone(solutions, objectives, best_solution, best_objective)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 最佳目标值 = {best_objective:.2f}")
            
            # 早停检查：如果连续early_stop_patience次迭代没有改进，提前终止
            if no_improve_count >= self.early_stop_patience:
                print(f"迭代 {iteration}: 连续{self.early_stop_patience}次迭代无改进，提前终止（早停）")
                break
        
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
        """
        解码个体：对于给定的建设决策z，使用线性规划求解最优的x和y
        
        优先使用LP求解（精确方法），如果不可用则使用贪心方法（近似方法）
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]∈{0,1}
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - x: 最优充电桩数量数组
                - y: 最优用户分配矩阵
                - objective: 目标函数值
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
        目标：max Σ_i Σ_j (p_i * y_ij) - Σ_j (c_j * z_j)
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
        
        # 目标函数: max Σ_i Σ_j (p_i * y_ij) - Σ_j (c_j * z_j)
        prob += (pulp.lpSum([self.p[i] * y[i][j] 
                            for i in range(self.n) for j in range(self.m)])
                - pulp.lpSum([self.c[j] * z[j] for j in range(self.m)]))
        
        # 约束1: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    prob += y[i][j] <= self.D[i]
                else:
                    prob += y[i][j] == 0
        
        # 约束2: 需求约束 Σ_j y_ij ≤ D_i
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: 容量约束 Σ_i y_ij ≤ x_j
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 使用CBC求解器求解
        solver = pulp.PULP_CBC_CMD(msg=0)  # msg=0表示不输出求解过程
        prob.solve(solver)
        
        # 提取解（处理可能的None值）
        x_sol = np.array([pulp.value(x[j]) if pulp.value(x[j]) is not None else 0.0
                         for j in range(self.m)])
        y_sol = np.array([[pulp.value(y[i][j]) if pulp.value(y[i][j]) is not None else 0.0
                          for j in range(self.m)] for i in range(self.n)])
        
        # 计算目标函数值
        objective = self.calculate_objective(z, x_sol, y_sol)
        
        return x_sol, y_sol, objective
    
    def _solve_given_z_greedy(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用贪心方法求解给定z的x和y（近似方法，当LP不可用时使用）
        
        Args:
            z: 建设决策数组，形状为(m,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 近似的x, y和目标函数值
        """
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 贪心分配：按收益从高到低排序
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
        
        # 设置充电桩数量
        for j in range(self.m):
            if z[j] == 1:
                x[j] = np.sum(y[:, j])
        
        objective = self.calculate_objective(z, x, y)
        return x, y, objective
    
    def _update_pheromone(self, solutions: List[np.ndarray], objectives: List[float],
                         best_solution: np.ndarray, best_objective: float):
        """
        更新信息素
        
        信息素更新规则：
        1. 信息素挥发：τ_j = (1 - ρ) * τ_j
        2. 信息素增强：根据解的质量增强被选择的区域的信息素
        3. 最佳解额外增强：对全局最佳解中的区域额外增强信息素
        4. 信息素上下界限制：防止信息素过小或过大
        
        Args:
            solutions: 所有蚂蚁构建的解列表
            objectives: 对应的目标函数值列表
            best_solution: 全局最佳解
            best_objective: 全局最佳目标函数值
        """
        # 步骤1: 信息素挥发（模拟信息素的自然挥发）
        # 挥发系数rho控制信息素的持久性，rho越大挥发越快
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # 步骤2: 根据解的优劣更新信息素（正反馈机制）
        # 解的质量越好，增强的信息素越多
        for z, obj in zip(solutions, objectives):
            if obj > 0:
                for j in range(self.m):
                    if z[j] == 1:  # 如果区域j被选择
                        # 增强量与解的质量成正比
                        self.pheromone[j] += self.q * obj / best_objective if best_objective > 0 else 0
        
        # 步骤3: 最佳解额外增强（强化学习）
        # 对全局最佳解中的区域进行额外增强，加速收敛到最优解
        if best_objective > 0:
            for j in range(self.m):
                if best_solution[j] == 1:
                    self.pheromone[j] += self.q * 2.0  # 额外增强系数为2.0
        
        # 步骤4: 信息素上下界限制
        # 防止信息素过小（导致探索不足）或过大（导致过早收敛）
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
