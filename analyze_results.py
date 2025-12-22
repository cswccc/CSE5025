"""
结果分析脚本：分析不同求解方法的结果和运行时间
对每个问题实例进行详细对比，不计算平均值（因为每个问题的情况不同）
"""

import csv
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def load_results(csv_file: str) -> List[Dict]:
    """加载CSV结果文件"""
    results = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值字段
            for key, value in row.items():
                if key in ['n', 'm', 'seed']:
                    row[key] = int(value) if value else None
                elif key.endswith('_objective') or key.endswith('_time'):
                    try:
                        row[key] = float(value) if value else None
                    except (ValueError, TypeError):
                        row[key] = None
                elif key.endswith('_feasible'):
                    row[key] = value == 'True' if value else False
                elif key.endswith('_z_count') or key.endswith('_x_total'):
                    try:
                        row[key] = int(float(value)) if value else None
                    except (ValueError, TypeError):
                        row[key] = None
            results.append(row)
    return results


def get_method_name(field_name: str) -> Optional[str]:
    """从字段名提取方法名"""
    methods = ['greedy', 'milp', 'brute_force', 'genetic']
    for method in methods:
        if field_name.startswith(method):
            return method
    return None


def format_time(seconds: Optional[float]) -> str:
    """格式化时间显示"""
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return f"{seconds*1000:.3f}ms"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    else:
        return f"{seconds/60:.2f}min"


def format_objective(value: Optional[float]) -> str:
    """格式化目标值显示"""
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def find_best_methods(result: Dict) -> Dict[str, str]:
    """找出每个指标的最佳方法"""
    methods = ['greedy', 'milp', 'brute_force', 'genetic']
    
    # 找出目标值最大的方法（可行解中）
    best_obj_method = None
    best_obj_value = float('-inf')
    for method in methods:
        obj_key = f'{method}_objective'
        feasible_key = f'{method}_feasible'
        if result.get(feasible_key) and result.get(obj_key) is not None:
            if result[obj_key] > best_obj_value:
                best_obj_value = result[obj_key]
                best_obj_method = method
    
    # 找出时间最短的方法（可行解中）
    fastest_method = None
    fastest_time = float('inf')
    for method in methods:
        time_key = f'{method}_time'
        feasible_key = f'{method}_feasible'
        if result.get(feasible_key) and result.get(time_key) is not None:
            if result[time_key] < fastest_time:
                fastest_time = result[time_key]
                fastest_method = method
    
    return {
        'best_objective': best_obj_method,
        'fastest': fastest_method
    }


def analyze_single_problem(result: Dict, verbose: bool = True):
    """分析单个问题的结果"""
    problem_id = result['problem_id']
    n, m = result['n'], result['m']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"问题ID: {problem_id}  (n={n}, m={m}, seed={result['seed']})")
        print(f"{'='*80}")
    
    methods = ['greedy', 'milp', 'brute_force', 'genetic']
    method_names = {
        'greedy': '贪心算法',
        'milp': 'MILP求解器',
        'brute_force': '暴力枚举',
        'genetic': '遗传算法'
    }
    
    # 构建对比表格
    rows = []
    for method in methods:
        obj_key = f'{method}_objective'
        time_key = f'{method}_time'
        feasible_key = f'{method}_feasible'
        z_count_key = f'{method}_z_count'
        x_total_key = f'{method}_x_total'
        
        obj = result.get(obj_key)
        time_val = result.get(time_key)
        feasible = result.get(feasible_key, False)
        z_count = result.get(z_count_key)
        x_total = result.get(x_total_key)
        
        rows.append({
            'method': method_names[method],
            'objective': obj,
            'time': time_val,
            'feasible': feasible,
            'z_count': z_count,
            'x_total': x_total
        })
    
    # 打印表格
    if verbose:
        print(f"\n{'方法':<12} {'目标值':<15} {'运行时间':<15} {'可行':<8} {'建设区域数':<12} {'充电桩总数':<12}")
        print("-" * 80)
        for row in rows:
            feasible_str = "✓" if row['feasible'] else "✗"
            print(f"{row['method']:<12} {format_objective(row['objective']):<15} "
                  f"{format_time(row['time']):<15} {feasible_str:<8} "
                  f"{row['z_count'] if row['z_count'] is not None else 'N/A':<12} "
                  f"{row['x_total'] if row['x_total'] is not None else 'N/A':<12}")
    
    # 找出最佳方法
    best_methods = find_best_methods(result)
    
    if verbose:
        print("\n最佳结果:")
        if best_methods['best_objective']:
            best_obj = result[f"{best_methods['best_objective']}_objective"]
            print(f"  • 最佳目标值: {method_names[best_methods['best_objective']]} "
                  f"(目标值={format_objective(best_obj)})")
        if best_methods['fastest']:
            fastest_time = result[f"{best_methods['fastest']}_time"]
            print(f"  • 最快方法: {method_names[best_methods['fastest']]} "
                  f"(时间={format_time(fastest_time)})")
        
        # 对比不同方法的目标值差异
        print("\n目标值对比（相对于最佳值）:")
        if best_methods['best_objective']:
            best_obj_value = result[f"{best_methods['best_objective']}_objective"]
            for row in rows:
                if row['feasible'] and row['objective'] is not None:
                    diff = row['objective'] - best_obj_value
                    diff_pct = (diff / abs(best_obj_value)) * 100 if best_obj_value != 0 else 0
                    if diff == 0:
                        status = "（最佳）"
                    else:
                        status = f"（{diff:+.2f}, {diff_pct:+.2f}%）"
                    print(f"  • {row['method']:<12}: {format_objective(row['objective'])} {status}")
    
    return {
        'problem_id': problem_id,
        'rows': rows,
        'best_methods': best_methods
    }


def analyze_all_problems(results: List[Dict], verbose: bool = True):
    """分析所有问题"""
    print(f"\n{'='*80}")
    print(f"批量测试结果分析")
    print(f"总共 {len(results)} 个问题实例")
    print(f"{'='*80}")
    
    # 统计每个方法成为最佳的次数
    best_obj_counts = {'greedy': 0, 'milp': 0, 'brute_force': 0, 'genetic': 0}
    fastest_counts = {'greedy': 0, 'milp': 0, 'brute_force': 0, 'genetic': 0}
    
    method_names = {
        'greedy': '贪心算法',
        'milp': 'MILP求解器',
        'brute_force': '暴力枚举',
        'genetic': '遗传算法'
    }
    
    # 分析每个问题
    for result in results:
        analysis = analyze_single_problem(result, verbose=verbose)
        best_methods = analysis['best_methods']
        
        if best_methods['best_objective']:
            best_obj_counts[best_methods['best_objective']] += 1
        if best_methods['fastest']:
            fastest_counts[best_methods['fastest']] += 1
    
    # 打印统计摘要
    print(f"\n{'='*80}")
    print("统计摘要（各方法成为最佳的次数）")
    print(f"{'='*80}")
    
    print("\n最佳目标值次数:")
    for method, count in best_obj_counts.items():
        print(f"  • {method_names[method]:<12}: {count:3d} 次 ({count/len(results)*100:.1f}%)")
    
    print("\n最快速度次数:")
    for method, count in fastest_counts.items():
        print(f"  • {method_names[method]:<12}: {count:3d} 次 ({count/len(results)*100:.1f}%)")


def analyze_by_problem_size(results: List[Dict]):
    """
    按问题规模(n, m)分组分析：
    1. 每种方法在不同情况下（同一个m和n下）得到最优解的次数和概率
       - 对于每个实例，找出所有可行方法中目标值最大的作为"最优值"
       - 如果两个结果的目标值误差小于1e-5，则认为是同一值
    2. 绝对运行时间统计（平均、中位数、最小、最大）
    """
    methods = ['greedy', 'milp', 'brute_force', 'genetic']
    method_names = {
        'greedy': '贪心算法',
        'milp': 'MILP求解器',
        'brute_force': '暴力枚举',
        'genetic': '遗传算法'
    }
    
    # 按(n, m)分组
    groups = defaultdict(list)
    for result in results:
        key = (result['n'], result['m'])
        groups[key].append(result)
    
    print(f"\n{'='*80}")
    print("按问题规模分组分析")
    print(f"{'='*80}")
    
    # 对每个(n, m)组合进行分析
    for (n, m), group_results in sorted(groups.items()):
        print(f"\n{'='*80}")
        print(f"问题规模: n={n}, m={m}  (共 {len(group_results)} 个实例)")
        print(f"{'='*80}")
        
        # 统计每个方法得到最优解的次数
        best_obj_counts = defaultdict(int)
        total_valid = 0  # 有可行解的实例数
        
        # 统计时间信息
        time_stats = defaultdict(list)  # 每个方法的时间列表
        
        for result in group_results:
            # 找出该实例的最佳目标值（所有可行方法中的最大值）
            best_obj_value = float('-inf')
            best_obj_methods = []  # 可能有多个方法得到相同的最优值（误差小于1e-5）
            
            for method in methods:
                obj_key = f'{method}_objective'
                feasible_key = f'{method}_feasible'
                if result.get(feasible_key) and result.get(obj_key) is not None:
                    obj_val = result[obj_key]
                    if obj_val > best_obj_value + 1e-5:
                        # 明显更优，更新最优值
                        best_obj_value = obj_val
                        best_obj_methods = [method]
                    elif abs(obj_val - best_obj_value) < 1e-5:
                        # 误差小于1e-5，认为是同一值
                        best_obj_methods.append(method)
            
            # 如果有可行解，记录得到最优解的方法
            if best_obj_methods:
                total_valid += 1
                for method in best_obj_methods:
                    best_obj_counts[method] += 1
                
                # 统计绝对时间信息
                for method in methods:
                    time_key = f'{method}_time'
                    feasible_key = f'{method}_feasible'
                    if result.get(feasible_key) and result.get(time_key) is not None:
                        time_val = result[time_key]
                        time_stats[method].append(time_val)
        
        # 1. 打印最优解统计（优化排版）
        print(f"\n{'─'*80}")
        print(f"{'方法':<15} {'得到最优解次数':<18} {'概率':<12} {'百分比':<10}")
        print(f"{'─'*80}")
        
        if total_valid > 0:
            for method in methods:
                count = best_obj_counts[method]
                prob = count / total_valid
                print(f"{method_names[method]:<15} {count:<18} {prob:<12.4f} {prob*100:>6.2f}%")
        else:
            print("  无有效实例")
        
        # 2. 打印绝对运行时间统计（优化排版）
        print(f"\n{'─'*80}")
        print(f"{'方法':<15} {'平均时间':<18} {'中位数时间':<18} {'最小时间':<18} {'最大时间':<18}")
        print(f"{'─'*80}")
        
        for method in methods:
            if time_stats[method]:
                times = time_stats[method]
                avg_time = sum(times) / len(times)
                sorted_times = sorted(times)
                median_time = sorted_times[len(sorted_times) // 2]
                min_time = min(times)
                max_time = max(times)
                print(f"{method_names[method]:<15} {format_time(avg_time):<18} "
                      f"{format_time(median_time):<18} {format_time(min_time):<18} "
                      f"{format_time(max_time):<18}")
            else:
                print(f"{method_names[method]:<15} {'N/A':<18} {'N/A':<18} {'N/A':<18} {'N/A':<18}")
    
    # 总体统计（跨所有规模）
    print(f"\n{'='*80}")
    print("总体统计（跨所有问题规模）")
    print(f"{'='*80}")
    
    # 总体最优解概率
    total_best_obj_counts = defaultdict(int)
    total_valid_all = 0
    
    # 总体时间统计
    all_time_stats = defaultdict(list)
    
    for result in results:
        best_obj_value = float('-inf')
        best_obj_methods = []
        
        for method in methods:
            obj_key = f'{method}_objective'
            feasible_key = f'{method}_feasible'
            if result.get(feasible_key) and result.get(obj_key) is not None:
                obj_val = result[obj_key]
                if obj_val > best_obj_value + 1e-5:
                    best_obj_value = obj_val
                    best_obj_methods = [method]
                elif abs(obj_val - best_obj_value) < 1e-5:
                    best_obj_methods.append(method)
        
        if best_obj_methods:
            total_valid_all += 1
            for method in best_obj_methods:
                total_best_obj_counts[method] += 1
        
        # 统计绝对时间
        for method in methods:
            time_key = f'{method}_time'
            feasible_key = f'{method}_feasible'
            if result.get(feasible_key) and result.get(time_key) is not None:
                all_time_stats[method].append(result[time_key])
    
    print(f"\n{'─'*80}")
    print(f"{'方法':<15} {'得到最优解次数':<18} {'概率':<12} {'百分比':<10}")
    print(f"{'─'*80}")
    
    if total_valid_all > 0:
        for method in methods:
            count = total_best_obj_counts[method]
            prob = count / total_valid_all
            print(f"{method_names[method]:<15} {count:<18} {prob:<12.4f} {prob*100:>6.2f}%")
    else:
        print("  无有效实例")
    
    # 总体绝对时间统计
    print(f"\n{'─'*80}")
    print(f"{'方法':<15} {'平均时间':<18} {'中位数时间':<18} {'最小时间':<18} {'最大时间':<18}")
    print(f"{'─'*80}")
    
    for method in methods:
        if all_time_stats[method]:
            times = all_time_stats[method]
            avg_time = sum(times) / len(times)
            sorted_times = sorted(times)
            median_time = sorted_times[len(sorted_times) // 2]
            min_time = min(times)
            max_time = max(times)
            print(f"{method_names[method]:<15} {format_time(avg_time):<18} "
                  f"{format_time(median_time):<18} {format_time(min_time):<18} "
                  f"{format_time(max_time):<18}")
        else:
            print(f"{method_names[method]:<15} {'N/A':<18} {'N/A':<18} {'N/A':<18} {'N/A':<18}")


def compare_methods(results: List[Dict], method1: str, method2: str):
    """对比两个方法"""
    method_names = {
        'greedy': '贪心算法',
        'milp': 'MILP求解器',
        'brute_force': '暴力枚举',
        'genetic': '遗传算法'
    }
    
    print(f"\n{'='*80}")
    print(f"方法对比: {method_names[method1]} vs {method_names[method2]}")
    print(f"{'='*80}")
    
    print(f"\n{'问题ID':<12} {'方法1目标值':<15} {'方法2目标值':<15} "
          f"{'方法1时间':<15} {'方法2时间':<15} {'目标值差异':<15}")
    print("-" * 100)
    
    for result in results:
        problem_id = result['problem_id']
        obj1 = result.get(f'{method1}_objective')
        obj2 = result.get(f'{method2}_objective')
        time1 = result.get(f'{method1}_time')
        time2 = result.get(f'{method2}_time')
        feasible1 = result.get(f'{method1}_feasible', False)
        feasible2 = result.get(f'{method2}_feasible', False)
        
        if not feasible1:
            obj1 = None
        if not feasible2:
            obj2 = None
        
        obj_diff = None
        if obj1 is not None and obj2 is not None:
            obj_diff = obj1 - obj2
        
        print(f"{problem_id:<12} {format_objective(obj1):<15} {format_objective(obj2):<15} "
              f"{format_time(time1):<15} {format_time(time2):<15} "
              f"{format_objective(obj_diff) if obj_diff is not None else 'N/A':<15}")


def main():
    parser = argparse.ArgumentParser(description='分析批量测试结果')
    parser.add_argument('csv_file', type=str, help='结果CSV文件路径')
    parser.add_argument('--problem-id', type=str, help='只分析指定问题ID（如 P1_S1）')
    parser.add_argument('--compare', nargs=2, metavar=('METHOD1', 'METHOD2'),
                       choices=['greedy', 'milp', 'brute_force', 'genetic'],
                       help='对比两个方法（如: --compare greedy milp）')
    parser.add_argument('--brief', action='store_true', help='简要模式（不显示每个问题的详细信息）')
    parser.add_argument('--by-size', action='store_true', 
                       help='按问题规模(n,m)分组分析最优解概率和时间比率')
    
    args = parser.parse_args()
    
    # 加载结果
    try:
        results = load_results(args.csv_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载文件时出错: {e}")
        sys.exit(1)
    
    if not results:
        print("错误: 结果文件为空")
        sys.exit(1)
    
    # 如果指定了问题ID，只分析该问题
    if args.problem_id:
        filtered_results = [r for r in results if r['problem_id'] == args.problem_id]
        if not filtered_results:
            print(f"错误: 找不到问题ID {args.problem_id}")
            sys.exit(1)
        analyze_single_problem(filtered_results[0], verbose=True)
        return
    
    # 如果指定了对比，进行方法对比
    if args.compare:
        compare_methods(results, args.compare[0], args.compare[1])
        return
    
    # 如果指定了按规模分组分析
    if args.by_size:
        analyze_by_problem_size(results)
        return
    
    # 默认：分析所有问题
    analyze_all_problems(results, verbose=not args.brief)


if __name__ == '__main__':
    main()

