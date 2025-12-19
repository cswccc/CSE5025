"""
主程序：运行所有求解方法并比较结果
"""

import argparse
import json
import time
import numpy as np
from data_generator import DataGenerator
from solvers import (
    BruteForceSolver,
    GreedySolver,
    MILPSolver,
    GeneticAlgorithmSolver,
    AntColonySolver,
    NeuralNetworkSolver
)


def run_solver(solver_name: str, solver, instance: dict):
    """运行求解器并返回结果"""
    print(f"\n{'='*60}")
    print(f"运行求解器: {solver_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        solution, objective = solver.solve()
        elapsed_time = time.time() - start_time
        
        # 验证解的可行性
        is_feasible, msg = solver.is_feasible(
            np.array(solution['z']),
            np.array(solution['x']),
            np.array(solution['y'])
        )
        
        result = {
            'solver': solver_name,
            'objective': objective,
            'time': elapsed_time,
            'feasible': is_feasible,
            'message': msg,
            'solution': solution
        }
        
        print(f"目标函数值: {objective:.2f}")
        print(f"运行时间: {elapsed_time:.4f} 秒")
        print(f"可行性: {'✓ 可行' if is_feasible else '✗ 不可行'}")
        if not is_feasible:
            print(f"错误信息: {msg}")
        print(f"建设区域数: {sum(solution['z'])}/{len(solution['z'])}")
        print(f"总充电桩数: {sum(solution['x']):.0f}")
        
        return result
        
    except Exception as e:
        print(f"求解器 {solver_name} 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'solver': solver_name,
            'objective': float('-inf'),
            'time': 0.0,
            'feasible': False,
            'message': str(e),
            'solution': None
        }


def main():
    parser = argparse.ArgumentParser(description='充电桩覆盖收益最大化问题求解')
    parser.add_argument('--instance', type=str, help='问题实例JSON文件路径')
    parser.add_argument('--n', type=int, default=20, help='楼栋数量')
    parser.add_argument('--m', type=int, default=10, help='区域数量')
    parser.add_argument('--generate', action='store_true', help='生成新的问题实例')
    parser.add_argument('--output', type=str, default='results.json', help='结果输出文件')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['greedy', 'milp', 'ga', 'aco', 'nn'],
                       choices=['brute_force', 'greedy', 'milp', 'ga', 'aco', 'nn', 'all'],
                       help='要运行的求解方法')
    
    args = parser.parse_args()
    
    # 生成或加载问题实例
    if args.generate or args.instance is None:
        print("生成新的问题实例...")
        generator = DataGenerator(seed=42)
        instance = generator.generate_instance(
            n=args.n,
            m=args.m,
            coverage_rate=0.3
        )
        instance_file = args.instance if args.instance else 'instance.json'
        generator.save_instance(instance, instance_file)
        print(f"问题实例已保存到: {instance_file}")
    else:
        print(f"从文件加载问题实例: {args.instance}")
        generator = DataGenerator()
        instance = generator.load_instance(args.instance)
    
    print(f"\n问题规模: {instance['n']} 栋楼, {instance['m']} 个区域")
    
    # 确定要运行的求解器
    methods = args.methods
    if 'all' in methods:
        methods = ['brute_force', 'greedy', 'milp', 'ga', 'aco', 'nn']
    
    results = []
    
    # 运行各种求解器
    if 'brute_force' in methods and instance['m'] <= 15:
        solver = BruteForceSolver(instance)
        result = run_solver('暴力枚举法', solver, instance)
        results.append(result)
    elif 'brute_force' in methods:
        print("\n跳过暴力枚举法：问题规模过大")
    
    if 'greedy' in methods:
        solver = GreedySolver(instance)
        result = run_solver('贪心算法', solver, instance)
        results.append(result)
    
    if 'milp' in methods:
        try:
            solver = MILPSolver(instance)
            result = run_solver('MILP求解器', solver, instance)
            results.append(result)
        except ImportError:
            print("\n跳过MILP求解器：PuLP未安装")
    
    if 'ga' in methods:
        solver = GeneticAlgorithmSolver(
            instance,
            pop_size=50,
            max_generations=100
        )
        result = run_solver('遗传算法', solver, instance)
        results.append(result)
    
    if 'aco' in methods:
        solver = AntColonySolver(
            instance,
            num_ants=20,
            max_iterations=100
        )
        result = run_solver('蚁群算法', solver, instance)
        results.append(result)
    
    if 'nn' in methods:
        try:
            solver = NeuralNetworkSolver(
                instance,
                num_episodes=500
            )
            result = run_solver('神经网络方法', solver, instance)
            results.append(result)
        except ImportError:
            print("\n跳过神经网络方法：PyTorch未安装")
    
    # 比较结果
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(f"{'求解器':<15} {'目标值':<15} {'运行时间(秒)':<15} {'可行性':<10}")
    print("-" * 60)
    
    feasible_results = [r for r in results if r['feasible']]
    feasible_results.sort(key=lambda x: x['objective'], reverse=True)
    
    for result in results:
        status = "✓" if result['feasible'] else "✗"
        print(f"{result['solver']:<15} {result['objective']:<15.2f} {result['time']:<15.4f} {status:<10}")
    
    if feasible_results:
        best = feasible_results[0]
        print(f"\n最佳解: {best['solver']} (目标值: {best['objective']:.2f})")
    
    # 保存结果
    output_data = {
        'instance': instance,
        'results': results,
        'best_solver': feasible_results[0]['solver'] if feasible_results else None,
        'best_objective': feasible_results[0]['objective'] if feasible_results else None
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
