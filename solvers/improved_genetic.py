"""
改进的遗传算法求解器

优化策略：
1. 智能初始化：使用贪心解作为部分初始种群
2. 局部搜索改进：对优秀个体进行局部搜索
3. 自适应参数：根据进化情况调整变异率
4. 精英保留和多样性保持
"""

import numpy as np
import time
import random
from typing import Dict, Tuple, List
from .base_solver import BaseSolver


class ImprovedGeneticSolver(BaseSolver):
    """
    改进的遗传算法求解器
    
    使用智能初始化、局部搜索改进和自适应参数优化。
    """
    
    def __init__(self, instance: Dict, 
                 pop_size: int = 50,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_rate: float = 0.1,
                 local_search_rate: float = 0.2):
        """
        初始化改进的遗传算法求解器
        
        Args:
            instance: 问题实例
            pop_size: 种群大小
            max_generations: 最大代数
            crossover_rate: 交叉率
            mutation_rate: 初始变异率
            elite_rate: 精英比例
            local_search_rate: 局部搜索比例（对多少比例的个体进行局部搜索）
        """
        super().__init__(instance)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.local_search_rate = local_search_rate
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用改进的遗传算法求解
        
        Returns:
            Tuple[Dict, float]: 最佳解和目标函数值
        """
        start_time = time.time()
        
        # 智能初始化种群
        population = self._initialize_population_smart()
        
        best_individual = None
        best_fitness = float('-inf')
        stagnation_count = 0
        
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
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # 局部搜索改进（对优秀个体）
            population = self._local_search_improvement(population, fitness_scores)
            
            # 选择
            selected = self._select(population, fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 自适应变异
            current_mutation_rate = self._adaptive_mutation_rate(generation, stagnation_count)
            offspring = self._mutate(offspring, current_mutation_rate)
            
            # 精英保留
            elite_count = int(self.pop_size * self.elite_rate)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = [population[i].copy() for i in elite_indices]
            
            # 更新种群
            population = elite + offspring[:self.pop_size - elite_count]
            
            # 保持多样性（如果收敛太快，增加多样性）
            if stagnation_count > 20:
                population = self._maintain_diversity(population)
                stagnation_count = 0
            
            if generation % 10 == 0:
                print(f"代 {generation}: 最佳适应度 = {best_fitness:.2f}, 变异率 = {current_mutation_rate:.3f}")
        
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
    
    def _initialize_population_smart(self) -> List[np.ndarray]:
        """
        智能初始化种群
        
        使用多种策略：
        1. 贪心解
        2. 随机解（但偏向选择高效益区域）
        3. 完全随机解
        """
        population = []
        
        # 1. 添加贪心解（利用改进的贪心算法）
        greedy_z = self._get_greedy_solution()
        if greedy_z is not None:
            population.append(greedy_z)
        
        # 2. 添加基于效益的随机解
        num_biased = int(self.pop_size * 0.3)
        for _ in range(num_biased):
            z = self._generate_biased_solution()
            population.append(z)
        
        # 3. 添加完全随机解
        num_random = self.pop_size - len(population)
        for _ in range(num_random):
            # 随机选择30%的区域建设
            z = np.random.binomial(1, 0.3, size=self.m).astype(int)
            population.append(z)
        
        return population[:self.pop_size]
    
    def _get_greedy_solution(self) -> np.ndarray:
        """使用简单贪心获得初始解"""
        z = np.zeros(self.m, dtype=int)
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
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
        for j in sorted_regions:
            if benefits[j] > 0:
                test_z = z.copy()
                test_z[j] = 1
                test_x, test_y, test_obj = self._decode_individual(test_z)
                current_obj = self.calculate_objective(z, x, y)
                if test_obj > current_obj:
                    z[j] = 1
                    x, y = test_x.copy(), test_y.copy()
        
        return z
    
    def _generate_biased_solution(self) -> np.ndarray:
        """生成偏向高效益区域的随机解"""
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
        
        # 将效益归一化为选择概率
        benefits = benefits - np.min(benefits) + 1  # 确保为正
        probs = benefits / np.sum(benefits)
        
        # 根据概率选择区域
        for j in range(self.m):
            if np.random.random() < probs[j] * 2:  # 增加选择概率
                z[j] = 1
        
        return z
    
    def _local_search_improvement(self, population: List[np.ndarray], 
                                  fitness_scores: List[float]) -> List[np.ndarray]:
        """
        局部搜索改进
        
        对优秀个体进行局部搜索：尝试添加/删除/交换区域。
        """
        improved_population = []
        num_to_improve = int(len(population) * self.local_search_rate)
        
        # 选择适应度最高的个体进行改进
        sorted_indices = np.argsort(fitness_scores)[-num_to_improve:]
        
        for idx in range(len(population)):
            if idx in sorted_indices:
                # 对优秀个体进行局部搜索
                improved = self._local_search(population[idx])
                improved_population.append(improved)
            else:
                improved_population.append(population[idx].copy())
        
        return improved_population
    
    def _local_search(self, z: np.ndarray) -> np.ndarray:
        """
        局部搜索：尝试添加/删除/交换一个区域
        
        Args:
            z: 当前解
            
        Returns:
            改进后的解
        """
        best_z = z.copy()
        best_obj = self._decode_individual(z)[2]
        
        # 尝试添加一个区域
        for j in range(self.m):
            if z[j] == 0:
                test_z = z.copy()
                test_z[j] = 1
                _, _, test_obj = self._decode_individual(test_z)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_z = test_z.copy()
        
        # 尝试删除一个区域
        for j in range(self.m):
            if z[j] == 1:
                test_z = z.copy()
                test_z[j] = 0
                _, _, test_obj = self._decode_individual(test_z)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_z = test_z.copy()
        
        return best_z
    
    def _adaptive_mutation_rate(self, generation: int, stagnation_count: int) -> float:
        """
        自适应变异率
        
        如果长时间没有改进，增加变异率；如果改进很快，降低变异率。
        """
        base_rate = self.mutation_rate
        
        # 如果停滞，增加变异率
        if stagnation_count > 10:
            return min(base_rate * 2, 0.5)
        elif stagnation_count > 5:
            return base_rate * 1.5
        
        # 早期代次使用较高变异率，后期降低
        if generation < self.max_generations * 0.3:
            return base_rate * 1.2
        else:
            return base_rate * 0.8
    
    def _maintain_diversity(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """保持种群多样性：随机替换部分个体"""
        num_to_replace = int(self.pop_size * 0.2)
        new_population = population.copy()
        
        for _ in range(num_to_replace):
            idx = random.randint(0, len(new_population) - 1)
            new_population[idx] = self._generate_biased_solution()
        
        return new_population
    
    def _decode_individual(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """解码个体（与基础版本相同）"""
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
    
    def _select(self, population: List[np.ndarray], fitness: List[float]) -> List[np.ndarray]:
        """选择操作（轮盘赌选择）"""
        fitness_array = np.array(fitness)
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            fitness_array = fitness_array - min_fitness + 1
        
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
                crossover_point = random.randint(1, self.m - 1)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        return offspring
    
    def _mutate(self, population: List[np.ndarray], mutation_rate: float) -> List[np.ndarray]:
        """变异操作（位翻转）"""
        mutated = []
        for individual in population:
            new_individual = individual.copy()
            for j in range(self.m):
                if random.random() < mutation_rate:
                    new_individual[j] = 1 - new_individual[j]
            mutated.append(new_individual)
        return mutated
