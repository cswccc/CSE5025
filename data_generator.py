"""
数据生成模块
用于生成充电桩覆盖收益最大化问题的测试数据
"""

import numpy as np
import random
import argparse
import os
from typing import Tuple, Dict, List
import json


class DataGenerator:
    """数据生成器类"""
    
    def __init__(self, seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子
        """
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_coverage_matrix(self, n: int, m: int, coverage_rate: float = 0.3) -> np.ndarray:
        """
        生成覆盖关系矩阵 a_ij
        
        Args:
            n: 楼栋数量
            m: 区域数量
            coverage_rate: 覆盖率，表示平均每个区域能覆盖多少比例的楼栋
            
        Returns:
            a_ij: n x m 的覆盖关系矩阵
        """
        # 使用伯努利分布生成覆盖关系
        a_ij = np.random.binomial(1, coverage_rate, size=(n, m))
        
        # 确保每个楼栋至少被一个区域覆盖
        for i in range(n):
            if np.sum(a_ij[i, :]) == 0:
                j = random.randint(0, m - 1)
                a_ij[i, j] = 1
        
        # 确保每个区域至少覆盖一个楼栋
        for j in range(m):
            if np.sum(a_ij[:, j]) == 0:
                i = random.randint(0, n - 1)
                a_ij[i, j] = 1
        
        return a_ij
    
    def generate_demands(self, n: int, min_demand: int = 10, max_demand: int = 100) -> np.ndarray:
        """
        生成每栋楼的潜在用户需求 D_i
        
        Args:
            n: 楼栋数量
            min_demand: 最小需求
            max_demand: 最大需求
            
        Returns:
            D: 长度为n的需求数组
        """
        return np.random.randint(min_demand, max_demand + 1, size=n)
    
    def generate_unit_profits(self, n: int, min_profit: float = 1.0, max_profit: float = 10.0, 
                              unified_profit: float = None) -> np.ndarray:
        """
        生成每栋楼的单位收益 p_i
        
        Args:
            n: 楼栋数量
            min_profit: 最小单位收益（当unified_profit为None时使用）
            max_profit: 最大单位收益（当unified_profit为None时使用）
            unified_profit: 统一收益值，如果提供则所有楼栋使用相同的单位收益
            
        Returns:
            p: 长度为n的单位收益数组
        """
        if unified_profit is not None:
            # 使用统一收益：所有楼栋的单位收益相同
            return np.ones(n) * unified_profit
        else:
            # 使用随机收益：在[min_profit, max_profit]范围内随机生成
            return np.random.uniform(min_profit, max_profit, size=n)
    
    def generate_costs(self, m: int, capacities: np.ndarray = None, 
                       cost_per_capacity: float = 2.0, 
                       noise_std: float = 0.2) -> np.ndarray:
        """
        生成每个区域的建设成本 c_j
        
        建设成本与容量上限近似成正比：c_j = cost_per_capacity * U_j + noise
        其中noise符合正态分布，标准差为cost_per_capacity * U_j * noise_std
        
        Args:
            m: 区域数量
            capacities: 容量上限数组，如果提供则基于此计算成本
            cost_per_capacity: 单位容量的成本系数（默认2.0）
            noise_std: 噪声的标准差系数（相对于基础成本的比例，默认0.2）
            
        Returns:
            c: 长度为m的成本数组（非负）
        """
        if capacities is not None:
            # 基于容量上限计算成本
            base_costs = cost_per_capacity * capacities
            # 添加正态分布的噪声
            noise = np.random.normal(0, base_costs * noise_std)
            costs = base_costs + noise
            # 确保成本非负
            costs = np.maximum(costs, base_costs * 0.1)  # 至少为基础成本的10%
        else:
            # 如果没有提供容量，使用默认方法（保持向后兼容）
            costs = np.random.uniform(50.0, 500.0, size=m)
        return costs
    
    def generate_capacity_limits(self, m: int, min_capacity: int = 20, max_capacity: int = 200) -> np.ndarray:
        """
        生成每个区域的容量上限 U_j
        
        Args:
            m: 区域数量
            min_capacity: 最小容量上限
            max_capacity: 最大容量上限
            
        Returns:
            U: 长度为m的容量上限数组
        """
        return np.random.randint(min_capacity, max_capacity + 1, size=m)
    
    def generate_instance(
        self, 
        n: int = 20, 
        m: int = 10,
        coverage_rate: float = 0.3,
        min_demand: int = 10,
        max_demand: int = 100,
        min_profit: float = 1.0,
        max_profit: float = 10.0,
        unified_profit: float = 5.0,  # 默认使用统一收益5.0
        min_capacity: int = 20,
        max_capacity: int = 200,
        cost_per_capacity: float = 2.0,  # 单位容量的成本系数
        cost_noise_std: float = 0.2  # 成本波动的标准差系数（相对于基础成本）
    ) -> Dict:
        """
        生成完整的问题实例
        
        Args:
            n: 楼栋数量
            m: 区域数量
            coverage_rate: 覆盖率
            min_demand: 最小需求
            max_demand: 最大需求
            min_profit: 最小单位收益（当unified_profit为None时使用）
            max_profit: 最大单位收益（当unified_profit为None时使用）
            unified_profit: 统一收益值，如果提供则所有楼栋使用相同的单位收益（默认5.0）
            min_capacity: 最小容量上限
            max_capacity: 最大容量上限
            cost_per_capacity: 单位容量的成本系数，建设成本 ≈ cost_per_capacity * 容量上限
            cost_noise_std: 成本波动的标准差系数（相对于基础成本的比例），控制波动程度
            
        Returns:
            包含所有问题参数的字典
        """
        # 先生成容量上限
        U = self.generate_capacity_limits(m, min_capacity, max_capacity)
        
        # 基于容量上限生成建设成本（成正比，但有正态分布的波动）
        c = self.generate_costs(m, capacities=U, 
                                cost_per_capacity=cost_per_capacity,
                                noise_std=cost_noise_std)
        
        instance = {
            'n': n,  # 楼栋数量
            'm': m,  # 区域数量
            'D': self.generate_demands(n, min_demand, max_demand).tolist(),  # 需求
            'p': self.generate_unit_profits(n, min_profit, max_profit, unified_profit).tolist(),  # 单位收益（统一）
            'c': c.tolist(),  # 建设成本（与容量上限近似成正比）
            'U': U.tolist(),  # 容量上限
            'a': self.generate_coverage_matrix(n, m, coverage_rate).tolist()  # 覆盖关系矩阵
        }
        
        return instance
    
    def save_instance(self, instance: Dict, filepath: str):
        """保存问题实例到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
    
    def load_instance(self, filepath: str) -> Dict:
        """从JSON文件加载问题实例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def generate_batch_instances(
    output_dir: str = "instances",
    problem_sizes: List[Tuple[int, int]] = None,
    instances_per_size: int = 10,
    coverage_rate: float = 0.3,
    unified_profit: float = 5.0,
    seed_base: int = 0
):
    """
    批量生成测试实例并保存到指定目录
    
    Args:
        output_dir: 输出目录路径
        problem_sizes: 问题规模列表，每个元素为(n, m)元组。如果为None，使用默认配置
        instances_per_size: 每个规模生成的实例数量
        coverage_rate: 覆盖率
        unified_profit: 统一收益值
        seed_base: 种子基础值，每个实例的seed = seed_base + idx
    """
    if problem_sizes is None:
        # 默认配置：5组不同的(n, m)组合
        problem_sizes = [
            (10, 5),   # 小规模
            (15, 8),   # 中小规模
            (20, 10),  # 中等规模
            (25, 12),  # 中大规模
            (30, 15),  # 大规模
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_instances = []
    total_count = 0
    
    print("=" * 80)
    print(f"批量生成测试数据到目录: {output_dir}")
    print("=" * 80)
    
    for group_idx, (n, m) in enumerate(problem_sizes):
        print(f"\n生成第 {group_idx + 1}/{len(problem_sizes)} 组: n={n}, m={m}")
        
        for instance_idx in range(instances_per_size):
            seed = group_idx * 1000 + seed_base + instance_idx
            generator = DataGenerator(seed=seed)
            
            instance = generator.generate_instance(
                n=n,
                m=m,
                coverage_rate=coverage_rate,
                unified_profit=unified_profit
            )
            
            # 添加元数据
            problem_id = f"P{group_idx+1}_S{instance_idx+1}"
            instance['problem_id'] = problem_id
            instance['seed'] = seed
            
            # 保存到文件
            filename = os.path.join(output_dir, f"{problem_id}.json")
            generator.save_instance(instance, filename)
            
            all_instances.append(instance)
            total_count += 1
            print(f"  ✓ {problem_id}: seed={seed}")
    
    print(f"\n共生成 {total_count} 个实例，已保存到 {output_dir}/")
    return all_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量生成充电桩覆盖收益最大化问题的测试数据')
    parser.add_argument('--output', '-o', type=str, default='instances',
                       help='输出目录路径（默认: instances）')
    parser.add_argument('--problem-sizes', type=str, 
                       help='问题规模，格式: "n1,m1;n2,m2;..." (例如: "10,5;15,8;20,10")')
    parser.add_argument('--instances-per-size', type=int, default=10,
                       help='每个规模生成的实例数量（默认: 10）')
    parser.add_argument('--coverage-rate', type=float, default=0.3,
                       help='覆盖率（默认: 0.3）')
    parser.add_argument('--unified-profit', type=float, default=5.0,
                       help='统一收益值（默认: 5.0）')
    parser.add_argument('--seed-base', type=int, default=0,
                       help='种子基础值（默认: 0）')
    parser.add_argument('--test', action='store_true',
                       help='生成单个测试实例用于验证')
    
    args = parser.parse_args()
    
    if args.test:
        # 测试模式：生成单个实例用于验证
        generator = DataGenerator(seed=42)
        instance = generator.generate_instance(
            n=10, 
            m=5,
            coverage_rate=0.4,
            unified_profit=args.unified_profit
        )
        
        print("生成的问题实例:")
        print(f"楼栋数量: {instance['n']}")
        print(f"区域数量: {instance['m']}")
        print(f"需求: {instance['D'][:5]}... (总共{len(instance['D'])}个)")
        print(f"单位收益: {instance['p'][:5]}... (统一收益: {instance['p'][0]:.2f})")
        print(f"建设成本: {[f'{c:.2f}' for c in instance['c']]}")
        print(f"容量上限: {instance['U']}")
        print(f"覆盖矩阵形状: {len(instance['a'])}x{len(instance['a'][0])}")
    else:
        # 批量生成模式
        problem_sizes = None
        if args.problem_sizes:
            # 解析问题规模
            problem_sizes = []
            for size_str in args.problem_sizes.split(';'):
                n, m = map(int, size_str.split(','))
                problem_sizes.append((n, m))
        
        generate_batch_instances(
            output_dir=args.output,
            problem_sizes=problem_sizes,
            instances_per_size=args.instances_per_size,
            coverage_rate=args.coverage_rate,
            unified_profit=args.unified_profit,
            seed_base=args.seed_base
        )
