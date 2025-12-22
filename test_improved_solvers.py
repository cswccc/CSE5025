"""
测试改进的求解器
"""

import json
import numpy as np
from data_generator import DataGenerator
from solvers import (
    GreedySolver,
    ImprovedGreedySolver,
    GeneticAlgorithmSolver,
    ImprovedGeneticSolver,
    LocalSearchSolver
)

# 生成测试实例
print("生成测试实例...")
generator = DataGenerator(seed=42)
instance = generator.generate_instance(n=15, m=8, unified_profit=5.0)

print(f"\n问题规模: n={instance['n']}, m={instance['m']}")
print(f"容量上限: {instance['U']}")
print(f"建设成本: {[f'{c:.2f}' for c in instance['c']]}")
cost_capacity_ratios = [instance['c'][j]/instance['U'][j] for j in range(len(instance['U']))]
print(f"成本/容量比: {[f'{r:.3f}' for r in cost_capacity_ratios]}")

methods = [
    ("基础贪心算法", GreedySolver(instance)),
    ("改进贪心算法", ImprovedGreedySolver(instance)),
    ("基础遗传算法", GeneticAlgorithmSolver(instance, pop_size=30, max_generations=50)),
    ("改进遗传算法", ImprovedGeneticSolver(instance, pop_size=30, max_generations=50)),
]

print("\n" + "="*80)
print("求解结果对比")
print("="*80)

results = []
for name, solver in methods:
    print(f"\n运行: {name}...")
    try:
        solution, objective = solver.solve()
        is_feasible, msg = solver.is_feasible(
            np.array(solution['z']),
            np.array(solution['x']),
            np.array(solution['y'])
        )
        print(f"  目标值: {objective:.2f}")
        print(f"  运行时间: {solver.solve_time:.4f}秒")
        print(f"  可行性: {'✓' if is_feasible else '✗'}")
        print(f"  建设区域数: {sum(solution['z'])}/{len(solution['z'])}")
        results.append((name, objective, solver.solve_time, is_feasible))
    except Exception as e:
        print(f"  错误: {e}")
        results.append((name, None, 0, False))

# 局部搜索：基于改进贪心的结果进行优化
print(f"\n运行: 局部搜索（基于改进贪心）...")
try:
    improved_greedy = ImprovedGreedySolver(instance)
    initial_sol, _ = improved_greedy.solve()
    local_search = LocalSearchSolver(instance, initial_solution=initial_sol, max_iterations=500)
    solution, objective = local_search.solve()
    is_feasible, msg = local_search.is_feasible(
        np.array(solution['z']),
        np.array(solution['x']),
        np.array(solution['y'])
    )
    print(f"  目标值: {objective:.2f}")
    print(f"  运行时间: {local_search.solve_time:.4f}秒")
    print(f"  可行性: {'✓' if is_feasible else '✗'}")
    results.append(("局部搜索", objective, local_search.solve_time, is_feasible))
except Exception as e:
    print(f"  错误: {e}")

print("\n" + "="*80)
print("结果汇总")
print("="*80)
print(f"{'方法':<20} {'目标值':<15} {'运行时间(秒)':<15} {'可行性':<10}")
print("-"*80)
results.sort(key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
for name, obj, time, feasible in results:
    obj_str = f"{obj:.2f}" if obj is not None else "N/A"
    print(f"{name:<20} {obj_str:<15} {time:<15.4f} {'✓' if feasible else '✗':<10}")

if results and results[0][1] is not None:
    print(f"\n最佳方法: {results[0][0]} (目标值: {results[0][1]:.2f})")
