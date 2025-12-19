"""
调试脚本：检查MILP和暴力枚举法之间的差异
"""

import json
import numpy as np
from data_generator import DataGenerator
from solvers import BruteForceSolver, MILPSolver

# 加载一个实例
generator = DataGenerator(seed=0)
instance = generator.generate_instance(n=10, m=5, coverage_rate=0.3)

print("=" * 80)
print("调试实例")
print("=" * 80)
print(f"n={instance['n']}, m={instance['m']}")

# 运行暴力枚举法
print("\n运行暴力枚举法...")
bf_solver = BruteForceSolver(instance)
bf_solution, bf_obj = bf_solver.solve()

print(f"暴力枚举法结果:")
print(f"  z = {bf_solution['z']}")
print(f"  x = {[f'{x:.1f}' for x in bf_solution['x']]}")
print(f"  目标值 = {bf_obj:.2f}")
print(f"  总充电桩数 = {sum(bf_solution['x']):.1f}")

# 验证可行性
bf_z = np.array(bf_solution['z'])
bf_x = np.array(bf_solution['x'])
bf_y = np.array(bf_solution['y'])
bf_feasible, bf_msg = bf_solver.is_feasible(bf_z, bf_x, bf_y)
print(f"  可行性: {bf_feasible}, {bf_msg}")

# 运行MILP
print("\n运行MILP求解器...")
milp_solver = MILPSolver(instance)
milp_solution, milp_obj = milp_solver.solve()

print(f"MILP结果:")
print(f"  z = {milp_solution['z']}")
print(f"  x = {[f'{x:.1f}' for x in milp_solution['x']]}")
print(f"  目标值 = {milp_obj:.2f}")
print(f"  总充电桩数 = {sum(milp_solution['x']):.1f}")

# 验证可行性
milp_z = np.array(milp_solution['z'])
milp_x = np.array(milp_solution['x'])
milp_y = np.array(milp_solution['y'])
milp_feasible, milp_msg = milp_solver.is_feasible(milp_z, milp_x, milp_y)
print(f"  可行性: {milp_feasible}, {milp_msg}")

# 检查差异
print("\n" + "=" * 80)
print("详细对比")
print("=" * 80)

print("\nz的差异:")
for j in range(instance['m']):
    if bf_z[j] != milp_z[j]:
        print(f"  区域{j}: 暴力={bf_z[j]}, MILP={milp_z[j]}")

print("\nx的差异:")
for j in range(instance['m']):
    if abs(bf_x[j] - milp_x[j]) > 0.01:
        print(f"  区域{j}: 暴力={bf_x[j]:.2f}, MILP={milp_x[j]:.2f}, 差异={milp_x[j] - bf_x[j]:.2f}")

print("\ny的差异（只显示非零部分）:")
total_diff = 0
for i in range(instance['n']):
    for j in range(instance['m']):
        diff = abs(bf_y[i, j] - milp_y[i, j])
        if diff > 0.01:
            print(f"  楼栋{i}->区域{j}: 暴力={bf_y[i, j]:.2f}, MILP={milp_y[i, j]:.2f}, 差异={diff:.2f}")
            total_diff += diff

print(f"\n总分配差异: {total_diff:.2f}")

# 重新计算目标函数
print("\n重新计算目标函数:")
bf_obj_recalc = bf_solver.calculate_objective(bf_z, bf_x, bf_y)
milp_obj_recalc = milp_solver.calculate_objective(milp_z, milp_x, milp_y)

print(f"  暴力枚举: {bf_obj:.2f} (原始) vs {bf_obj_recalc:.2f} (重新计算)")
print(f"  MILP: {milp_obj:.2f} (原始) vs {milp_obj_recalc:.2f} (重新计算)")

# 检查收益和成本
bf_revenue = np.sum(bf_solver.p[:, np.newaxis] * bf_y)
bf_cost = np.sum(bf_solver.c * bf_z)
milp_revenue = np.sum(milp_solver.p[:, np.newaxis] * milp_y)
milp_cost = np.sum(milp_solver.c * milp_z)

print("\n收益和成本分解:")
print(f"  暴力枚举: 收益={bf_revenue:.2f}, 成本={bf_cost:.2f}, 净收益={bf_obj:.2f}")
print(f"  MILP: 收益={milp_revenue:.2f}, 成本={milp_cost:.2f}, 净收益={milp_obj:.2f}")

# 检查MILP的z是否为整数
print("\n检查MILP的z值是否为整数:")
for j in range(instance['m']):
    if not (milp_z[j] == 0 or milp_z[j] == 1):
        print(f"  警告: z[{j}] = {milp_z[j]} 不是0或1!")
