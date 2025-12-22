"""
批量测试脚本：生成50组数据并使用不同方法求解
"""

import os
import json
import csv
import time
import numpy as np
from typing import List, Dict
from data_generator import DataGenerator
from solvers import (
    BruteForceSolver,
    GreedySolver,
    MILPSolver,
    AntColonySolver
)


def generate_50_instances():
    """
    生成50组数据
    5组不同的(n, m)组合，每组10个不同的seed
    """
    # 5组不同的(n, m)组合
    problem_sizes = [
        (10, 5),   # 小规模
        (15, 8),   # 中小规模
        (20, 10),  # 中等规模
        (25, 12),  # 中大规模
        (30, 15),  # 大规模
    ]
    
    instances = []
    
    for idx, (n, m) in enumerate(problem_sizes):
        print(f"\n生成第 {idx+1} 组: n={n}, m={m}")
        for seed_offset in range(10):
            seed = idx * 100 + seed_offset  # 使用不同的seed
            generator = DataGenerator(seed=seed)
            instance = generator.generate_instance(
                n=n,
                m=m,
                coverage_rate=0.3,
                unified_profit=5.0  # 使用统一收益5.0
            )
            instance['problem_id'] = f"P{idx+1}_S{seed_offset+1}"
            instance['seed'] = seed
            instances.append(instance)
            print(f"  生成实例 {instance['problem_id']}: seed={seed}")
    
    return instances


def test_one_instance(instance: Dict, methods: List[str]) -> Dict:
    """
    对一个实例运行所有求解器
    
    Returns:
        包含所有结果的字典
    """
    problem_id = instance['problem_id']
    n = instance['n']
    m = instance['m']
    seed = instance['seed']
    
    results = {
        'problem_id': problem_id,
        'n': n,
        'm': m,
        'seed': seed,
    }
    
    # 贪心算法
    if 'greedy' in methods:
        try:
            print(f"  [{problem_id}] 运行贪心算法...")
            solver = GreedySolver(instance)
            start_time = time.time()
            solution, objective = solver.solve()
            elapsed_time = time.time() - start_time
            is_feasible, msg = solver.is_feasible(
                np.array(solution['z']),
                np.array(solution['x']),
                np.array(solution['y'])
            )
            results['greedy_objective'] = objective
            results['greedy_time'] = elapsed_time
            results['greedy_feasible'] = is_feasible
            results['greedy_z_count'] = sum(solution['z'])
            results['greedy_x_total'] = sum(solution['x'])
        except Exception as e:
            print(f"  [{problem_id}] 贪心算法失败: {e}")
            results['greedy_objective'] = None
            results['greedy_time'] = None
            results['greedy_feasible'] = False
    
    # MILP求解器
    if 'milp' in methods:
        try:
            print(f"  [{problem_id}] 运行MILP求解器...")
            solver = MILPSolver(instance)
            start_time = time.time()
            solution, objective = solver.solve()
            elapsed_time = time.time() - start_time
            is_feasible, msg = solver.is_feasible(
                np.array(solution['z']),
                np.array(solution['x']),
                np.array(solution['y'])
            )
            results['milp_objective'] = objective
            results['milp_time'] = elapsed_time
            results['milp_feasible'] = is_feasible
            results['milp_z_count'] = sum(solution['z'])
            results['milp_x_total'] = sum(solution['x'])
        except Exception as e:
            print(f"  [{problem_id}] MILP求解器失败: {e}")
            results['milp_objective'] = None
            results['milp_time'] = None
            results['milp_feasible'] = False
    
    # 暴力枚举法（仅适用于小规模）
    if 'brute_force' in methods and m <= 15:
        try:
            print(f"  [{problem_id}] 运行暴力枚举法...")
            solver = BruteForceSolver(instance)
            start_time = time.time()
            solution, objective = solver.solve(time_limit=60)  # 限制60秒
            elapsed_time = time.time() - start_time
            is_feasible, msg = solver.is_feasible(
                np.array(solution['z']),
                np.array(solution['x']),
                np.array(solution['y'])
            )
            results['brute_force_objective'] = objective
            results['brute_force_time'] = elapsed_time
            results['brute_force_feasible'] = is_feasible
            results['brute_force_z_count'] = sum(solution['z'])
            results['brute_force_x_total'] = sum(solution['x'])
        except Exception as e:
            print(f"  [{problem_id}] 暴力枚举法失败: {e}")
            results['brute_force_objective'] = None
            results['brute_force_time'] = None
            results['brute_force_feasible'] = False
    elif 'brute_force' in methods:
        print(f"  [{problem_id}] 跳过暴力枚举法（m={m} > 15）")
        results['brute_force_objective'] = None
        results['brute_force_time'] = None
        results['brute_force_feasible'] = False
    
    # 蚁群算法
    if 'ant_colony' in methods:
        try:
            print(f"  [{problem_id}] 运行蚁群算法...")
            solver = AntColonySolver(
                instance,
                num_ants=20,
                max_iterations=50  # 为了加快速度，减少迭代次数
            )
            start_time = time.time()
            solution, objective = solver.solve()
            elapsed_time = time.time() - start_time
            is_feasible, msg = solver.is_feasible(
                np.array(solution['z']),
                np.array(solution['x']),
                np.array(solution['y'])
            )
            results['ant_colony_objective'] = objective
            results['ant_colony_time'] = elapsed_time
            results['ant_colony_feasible'] = is_feasible
            results['ant_colony_z_count'] = sum(solution['z'])
            results['ant_colony_x_total'] = sum(solution['x'])
        except Exception as e:
            print(f"  [{problem_id}] 蚁群算法失败: {e}")
            results['ant_colony_objective'] = None
            results['ant_colony_time'] = None
            results['ant_colony_feasible'] = False
    
    return results


def save_results_to_csv(results_list: List[Dict], filename: str = 'batch_test_results.csv'):
    """
    保存结果到CSV文件
    """
    if not results_list:
        print("没有结果可保存")
        return
    
    # 获取所有字段
    fieldnames = ['problem_id', 'n', 'm', 'seed']
    
    # 添加各方法的字段
    methods = ['greedy', 'milp', 'brute_force', 'ant_colony']
    for method in methods:
        fieldnames.extend([
            f'{method}_objective',
            f'{method}_time',
            f'{method}_feasible',
            f'{method}_z_count',
            f'{method}_x_total'
        ])
    
    # 写入CSV
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig for Excel
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)
    
    print(f"\n结果已保存到: {filename}")


def main():
    print("=" * 80)
    print("批量测试：生成50组数据并使用不同方法求解")
    print("=" * 80)
    
    # 选择要运行的方法
    methods = ['greedy', 'milp', 'brute_force', 'ant_colony']
    
    # 生成50组数据
    print("\n步骤1: 生成50组测试数据...")
    instances = generate_50_instances()
    print(f"\n共生成 {len(instances)} 组数据")
    
    # 保存实例数据（可选）
    os.makedirs('instances', exist_ok=True)
    for instance in instances:
        filename = f"instances/{instance['problem_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
    
    # 对每组数据运行求解器
    print("\n步骤2: 运行求解器...")
    all_results = []
    
    for idx, instance in enumerate(instances):
        print(f"\n[{idx+1}/{len(instances)}] 测试实例: {instance['problem_id']} (n={instance['n']}, m={instance['m']})")
        result = test_one_instance(instance, methods)
        all_results.append(result)
    
    # 保存结果
    print("\n步骤3: 保存结果...")
    save_results_to_csv(all_results, 'batch_test_results.csv')
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("测试完成！统计信息:")
    print("=" * 80)
    
    for method in methods:
        objectives = [r.get(f'{method}_objective') for r in all_results 
                     if r.get(f'{method}_objective') is not None]
        if objectives:
            times = [r.get(f'{method}_time') for r in all_results 
                    if r.get(f'{method}_time') is not None]
            print(f"\n{method.upper()}:")
            print(f"  成功运行: {len(objectives)}/{len(all_results)}")
            print(f"  平均目标值: {np.mean(objectives):.2f}")
            print(f"  最大目标值: {np.max(objectives):.2f}")
            print(f"  最小目标值: {np.min(objectives):.2f}")
            if times:
                print(f"  平均运行时间: {np.mean(times):.4f} 秒")
    
    print("\n详细结果请查看: batch_test_results.csv")


if __name__ == "__main__":
    main()