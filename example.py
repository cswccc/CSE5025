"""
示例脚本：演示如何使用各个模块
"""

from data_generator import DataGenerator
from solvers import GreedySolver, GeneticAlgorithmSolver

# 1. 生成问题实例
print("=" * 60)
print("步骤1: 生成问题实例")
print("=" * 60)
generator = DataGenerator(seed=42)
instance = generator.generate_instance(
    n=15,  # 15栋楼
    m=8,   # 8个区域
    coverage_rate=0.3
)
print(f"生成完成: {instance['n']} 栋楼, {instance['m']} 个区域")
print(f"需求范围: [{min(instance['D'])}, {max(instance['D'])}]")
print(f"收益范围: [{min(instance['p']):.2f}, {max(instance['p']):.2f}]")
print(f"成本范围: [{min(instance['c']):.2f}, {max(instance['c']):.2f}]")

# 2. 使用贪心算法求解
print("\n" + "=" * 60)
print("步骤2: 使用贪心算法求解")
print("=" * 60)
greedy_solver = GreedySolver(instance)
solution, objective = greedy_solver.solve()
print(f"\n贪心算法结果:")
print(f"  目标函数值: {objective:.2f}")
print(f"  建设区域数: {sum(solution['z'])}/{len(solution['z'])}")
print(f"  总充电桩数: {sum(solution['x']):.0f}")

# 验证可行性
import numpy as np
is_feasible, msg = greedy_solver.is_feasible(
    np.array(solution['z']),
    np.array(solution['x']),
    np.array(solution['y'])
)
print(f"  可行性: {'✓ 可行' if is_feasible else '✗ 不可行: ' + msg}")

# 3. 使用遗传算法求解
print("\n" + "=" * 60)
print("步骤3: 使用遗传算法求解")
print("=" * 60)
ga_solver = GeneticAlgorithmSolver(
    instance,
    pop_size=30,
    max_generations=50
)
solution, objective = ga_solver.solve()
print(f"\n遗传算法结果:")
print(f"  目标函数值: {objective:.2f}")
print(f"  建设区域数: {sum(solution['z'])}/{len(solution['z'])}")
print(f"  总充电桩数: {sum(solution['x']):.0f}")

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
