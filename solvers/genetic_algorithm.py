"""
遗传算法求解器

本模块实现了遗传算法来求解充电桩覆盖收益最大化问题。

算法思想：
    遗传算法是一种模拟自然选择和遗传机制的元启发式优化算法。
    通过选择、交叉、变异等操作，在解空间中搜索最优解。
    
    算法流程：
    1. 初始化：生成初始种群（随机生成多个建设决策z）
    2. 评估：计算每个个体的适应度（目标函数值）
    3. 选择：根据适应度选择父代个体（轮盘赌选择）
    4. 交叉：父代个体交叉产生子代（单点交叉）
    5. 变异：子代个体以一定概率变异（位翻转）
    6. 精英保留：保留最优的个体到下一代
    7. 重复步骤2-6直到达到最大代数
    
优点：
    - 可以跳出局部最优
    - 适合大规模问题
    - 参数可调，适应性强
    
缺点：
    - 不能保证最优解
    - 参数需要调优
    - 可能需要较多迭代次数
"""

import numpy as np
import time
import random
from typing import Dict, Tuple, List
from .base_solver import BaseSolver


class GeneticAlgorithmSolver(BaseSolver):
    """
    遗传算法求解器
    
    使用遗传算法的选择、交叉、变异操作来搜索最优解。
    个体编码：只编码建设决策z（二进制串），x和y通过贪心分配得到。
    """
    
    def __init__(self, instance: Dict, 
                 pop_size: int = 50,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_rate: float = 0.1):
        """
        初始化遗传算法求解器
        
        Args:
            instance: 问题实例
            pop_size: 种群大小
            max_generations: 最大代数
            crossover_rate: 交叉率
            mutation_rate: 变异率
            elite_rate: 精英比例
        """
        super().__init__(instance)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用遗传算法求解
        
        算法主循环：
        1. 初始化种群
        2. 对每一代：
           a. 评估所有个体的适应度
           b. 选择父代个体
           c. 交叉产生子代
           d. 变异子代
           e. 精英保留
           f. 更新种群
        
        Returns:
            Tuple[Dict, float]: 最佳解和目标函数值
        """
        start_time = time.time()
        
        # 步骤1: 初始化种群（每个个体是一个建设决策z的编码）
        population = self._initialize_population()
        
        # 记录全局最佳个体
        best_individual = None
        best_fitness = float('-inf')
        
        # 步骤2: 主循环（进化过程）
        for generation in range(self.max_generations):
            # 步骤2a: 评估适应度（目标函数值）
            fitness_scores = []
            for individual in population:
                z = individual  # 个体就是建设决策z
                # 解码：根据z计算x和y，并得到目标函数值
                x, y, obj = self._decode_individual(z)
                fitness_scores.append(obj)
                
                # 更新全局最佳
                if obj > best_fitness:
                    best_fitness = obj
                    best_individual = z.copy()
            
            # 步骤2b: 选择（根据适应度选择父代）
            selected = self._select(population, fitness_scores)
            
            # 步骤2c: 交叉（父代交叉产生子代）
            offspring = self._crossover(selected)
            
            # 步骤2d: 变异（子代以一定概率变异）
            offspring = self._mutate(offspring)
            
            # 步骤2e: 精英保留（保留最优的个体直接进入下一代）
            elite_count = int(self.pop_size * self.elite_rate)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]  # 适应度最高的elite_count个
            elite = [population[i].copy() for i in elite_indices]
            
            # 步骤2f: 更新种群（精英 + 子代）
            population = elite + offspring[:self.pop_size - elite_count]
            
            # 每10代输出一次进度
            if generation % 10 == 0:
                print(f"代 {generation}: 最佳适应度 = {best_fitness:.2f}")
        
        # 解码最佳个体
        z_best = best_individual
        x_best, y_best, obj_best = self._decode_individual(z_best)
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z_best.tolist(),
            'x': x_best.tolist(),
            'y': y_best.tolist()
        }
        self.best_objective = obj_best
        
        return self.best_solution, self.best_objective
    
    def _initialize_population(self) -> List[np.ndarray]:
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            # 随机生成建设决策
            z = np.random.binomial(1, 0.3, size=self.m).astype(int)
            population.append(z)
        return population
    
    def _decode_individual(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        解码个体：对于给定的建设决策z，使用线性规划求解最优的x和y
        
        优先使用LP求解（精确方法），如果不可用则使用贪心方法（近似方法）
        这是遗传算法中的适应度评估过程，使用LP确保对每种建设方案都能找到最优解。
        
        Args:
            z: 建设决策数组，形状为(m,)，个体的编码
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 最优的x, y和目标函数值
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
        
        # 贪心分配用户
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 按收益排序分配
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
    
    def _select(self, population: List[np.ndarray], fitness: List[float]) -> List[np.ndarray]:
        """
        选择操作（轮盘赌选择）
        
        根据适应度值计算选择概率，适应度越高被选中的概率越大。
        使用轮盘赌选择方法选择父代个体。
        
        Args:
            population: 当前种群
            fitness: 每个个体的适应度值列表
            
        Returns:
            List[np.ndarray]: 选择出的父代个体列表
        """
        fitness_array = np.array(fitness)
        # 处理负值：如果适应度有负值，将所有值平移到正数区间
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            fitness_array = fitness_array - min_fitness + 1
        
        # 计算选择概率（归一化）
        prob = fitness_array / np.sum(fitness_array)
        
        # 根据概率选择pop_size个个体
        selected = []
        for _ in range(self.pop_size):
            idx = np.random.choice(len(population), p=prob)
            selected.append(population[idx].copy())
        
        return selected
    
    def _crossover(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        交叉操作（单点交叉）
        
        将种群中的个体两两配对，以crossover_rate的概率进行交叉。
        交叉方式：随机选择交叉点，交换交叉点后的基因片段。
        
        Args:
            population: 父代种群
            
        Returns:
            List[np.ndarray]: 子代个体列表
        """
        offspring = []
        # 两两配对
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            # 以crossover_rate的概率进行交叉
            if random.random() < self.crossover_rate:
                # 单点交叉：随机选择交叉点
                crossover_point = random.randint(1, self.m - 1)
                # 子代1：父代1的前半部分 + 父代2的后半部分
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                # 子代2：父代2的前半部分 + 父代1的后半部分
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                # 不交叉，直接复制父代
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring
    
    def _mutate(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        变异操作（位翻转）
        
        对种群中的每个个体，以mutation_rate的概率对每个基因位进行变异。
        变异方式：将0变为1，或将1变为0（位翻转）。
        
        Args:
            population: 种群个体列表
            
        Returns:
            List[np.ndarray]: 变异后的种群
        """
        mutated = []
        for individual in population:
            new_individual = individual.copy()
            # 对每个基因位，以mutation_rate的概率进行变异
            for j in range(self.m):
                if random.random() < self.mutation_rate:
                    # 位翻转：0变1，1变0
                    new_individual[j] = 1 - new_individual[j]
            mutated.append(new_individual)
        return mutated
