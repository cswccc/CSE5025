"""
遗传算法求解器
"""

import numpy as np
import time
import random
from typing import Dict, Tuple, List
from .base_solver import BaseSolver


class GeneticAlgorithmSolver(BaseSolver):
    """遗传算法求解器"""
    
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
        
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        # 初始化种群（只编码z变量，x和y通过贪心分配得到）
        population = self._initialize_population()
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.max_generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                z = individual
                x, y, obj = self._decode_individual(z)
                fitness_scores.append(obj)
                
                if obj > best_fitness:
                    best_fitness = obj
                    best_individual = z.copy()
            
            # 选择
            selected = self._select(population, fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutate(offspring)
            
            # 精英保留
            elite_count = int(self.pop_size * self.elite_rate)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = [population[i].copy() for i in elite_indices]
            
            # 更新种群
            population = elite + offspring[:self.pop_size - elite_count]
            
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
        解码个体（将z转换为完整的解）
        使用贪心方法分配用户
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
        """选择操作（轮盘赌选择）"""
        fitness_array = np.array(fitness)
        # 处理负值
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            fitness_array = fitness_array - min_fitness + 1
        
        # 计算选择概率
        prob = fitness_array / np.sum(fitness_array)
        
        selected = []
        for _ in range(self.pop_size):
            idx = np.random.choice(len(population), p=prob)
            selected.append(population[idx].copy())
        
        return selected
    
    def _crossover(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """交叉操作（单点交叉）"""
        offspring = []
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            if random.random() < self.crossover_rate:
                # 单点交叉
                crossover_point = random.randint(1, self.m - 1)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring
    
    def _mutate(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """变异操作（位翻转）"""
        mutated = []
        for individual in population:
            new_individual = individual.copy()
            for j in range(self.m):
                if random.random() < self.mutation_rate:
                    new_individual[j] = 1 - new_individual[j]
            mutated.append(new_individual)
        return mutated
