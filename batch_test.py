"""
批量测试脚本：从指定目录加载数据并使用不同方法求解
"""

import os
import json
import csv
import time
import argparse
import numpy as np
from typing import List, Dict
from solvers import (
    BruteForceSolver,
    GreedySolver,
    MILPSolver,
    AntColonySolver,
    GeneticAlgorithmSolver
)


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
            print(f"  [{problem_id}] 贪心算法完成: 目标值={objective:.2f}, 时间={elapsed_time:.4f}秒, 可行={'✓' if is_feasible else '✗'}")
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
            print(f"  [{problem_id}] MILP求解器完成: 目标值={objective:.2f}, 时间={elapsed_time:.4f}秒, 可行={'✓' if is_feasible else '✗'}")
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
            print(f"  [{problem_id}] 暴力枚举法完成: 目标值={objective:.2f}, 时间={elapsed_time:.4f}秒, 可行={'✓' if is_feasible else '✗'}")
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
                num_ants=30,  # 增加蚂蚁数量
                max_iterations=150  # 增加迭代次数以获得更好的结果
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
            print(f"  [{problem_id}] 蚁群算法完成: 目标值={objective:.2f}, 时间={elapsed_time:.4f}秒, 可行={'✓' if is_feasible else '✗'}")
        except Exception as e:
            print(f"  [{problem_id}] 蚁群算法失败: {e}")
            results['ant_colony_objective'] = None
            results['ant_colony_time'] = None
            results['ant_colony_feasible'] = False
    
    # 遗传算法
    if 'genetic' in methods or 'ga' in methods:
        try:
            method_name = 'genetic' if 'genetic' in methods else 'ga'
            print(f"  [{problem_id}] 运行遗传算法...")
            solver = GeneticAlgorithmSolver(
                instance,
                pop_size=50,
                max_generations=100
            )
            start_time = time.time()
            solution, objective = solver.solve()
            elapsed_time = time.time() - start_time
            is_feasible, msg = solver.is_feasible(
                np.array(solution['z']),
                np.array(solution['x']),
                np.array(solution['y'])
            )
            results[f'{method_name}_objective'] = objective
            results[f'{method_name}_time'] = elapsed_time
            results[f'{method_name}_feasible'] = is_feasible
            results[f'{method_name}_z_count'] = sum(solution['z'])
            results[f'{method_name}_x_total'] = sum(solution['x'])
            print(f"  [{problem_id}] 遗传算法完成: 目标值={objective:.2f}, 时间={elapsed_time:.4f}秒, 可行={'✓' if is_feasible else '✗'}")
        except Exception as e:
            print(f"  [{problem_id}] 遗传算法失败: {e}")
            method_name = 'genetic' if 'genetic' in methods else 'ga'
            results[f'{method_name}_objective'] = None
            results[f'{method_name}_time'] = None
            results[f'{method_name}_feasible'] = False
    
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
    methods = ['greedy', 'milp', 'brute_force', 'ant_colony', 'genetic']
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


def load_instances_from_dir(instances_dir: str) -> List[Dict]:
    """
    从指定目录加载所有JSON实例文件
    
    Args:
        instances_dir: 实例数据目录路径
    
    Returns:
        List[Dict]: 实例列表，按文件名排序
    """
    if not os.path.exists(instances_dir):
        raise FileNotFoundError(f"目录不存在: {instances_dir}")
    
    instances = []
    files = sorted([f for f in os.listdir(instances_dir) if f.endswith('.json')])
    
    if len(files) == 0:
        raise ValueError(f"目录中没有找到JSON文件: {instances_dir}")
    
    for filename in files:
        filepath = os.path.join(instances_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            instance = json.load(f)
            instances.append(instance)
    
    return instances


def main():
    parser = argparse.ArgumentParser(
        description='批量测试：从指定目录加载数据并使用不同方法求解',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认instances目录
  python batch_test.py
  
  # 指定数据目录
  python batch_test.py --data-dir my_instances
  
  # 指定运行的方法
  python batch_test.py --methods greedy milp genetic
  
  # 运行所有方法
  python batch_test.py --methods all

注意: 数据生成请使用 data_generator.py
  python data_generator.py --output instances
        """
    )
    parser.add_argument('--data-dir', type=str, default='instances',
                       help='数据目录路径（默认: instances）')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['greedy', 'milp', 'brute_force', 'ant_colony', 'genetic'],
                       choices=['greedy', 'milp', 'brute_force', 'ant_colony', 'genetic', 'ga', 'all'],
                       help='要运行的求解方法（默认: greedy milp brute_force ant_colony genetic）')
    parser.add_argument('--output', type=str, default='batch_test_results.csv',
                       help='结果输出CSV文件路径（默认: batch_test_results.csv）')
    
    args = parser.parse_args()
    
    # 处理methods参数
    methods = args.methods
    if 'all' in methods:
        methods = ['greedy', 'milp', 'brute_force', 'ant_colony', 'genetic']
    # 将'ga'统一为'genetic'
    if 'ga' in methods:
        methods = [m if m != 'ga' else 'genetic' for m in methods]
        if 'genetic' not in methods:
            methods.append('genetic')
    
    print("=" * 80)
    print("批量测试：加载数据并使用不同方法求解")
    print("=" * 80)
    
    # 加载实例数据
    print(f"\n步骤1: 从目录加载实例数据: {args.data_dir}")
    try:
        instances = load_instances_from_dir(args.data_dir)
        print(f"✓ 成功加载 {len(instances)} 个实例")
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ 错误: {e}")
        print(f"\n提示: 请先使用 data_generator.py 生成数据:")
        print(f"  python data_generator.py --output {args.data_dir}")
        return
    
    # 对每组数据运行求解器
    print("\n步骤2: 运行求解器...")
    all_results = []
    
    for idx, instance in enumerate(instances):
        print(f"\n[{idx+1}/{len(instances)}] 测试实例: {instance['problem_id']} (n={instance['n']}, m={instance['m']})")
        result = test_one_instance(instance, methods)
        all_results.append(result)
    
    # 保存结果
    print("\n步骤3: 保存结果...")
    save_results_to_csv(all_results, args.output)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("测试完成！统计信息:")
    print("=" * 80)
    
    # 统一处理genetic方法名
    for method in methods:
        display_name = 'genetic' if method == 'ga' else method
        objectives = [r.get(f'{display_name}_objective') for r in all_results 
                     if r.get(f'{display_name}_objective') is not None]
        if objectives:
            times = [r.get(f'{display_name}_time') for r in all_results 
                    if r.get(f'{display_name}_time') is not None]
            print(f"\n{display_name.upper()}:")
            print(f"  成功运行: {len(objectives)}/{len(all_results)}")
            print(f"  平均目标值: {np.mean(objectives):.2f}")
            print(f"  最大目标值: {np.max(objectives):.2f}")
            print(f"  最小目标值: {np.min(objectives):.2f}")
            if times:
                print(f"  平均运行时间: {np.mean(times):.4f} 秒")
    
    print("\n详细结果请查看: batch_test_results.csv")


if __name__ == "__main__":
    main()