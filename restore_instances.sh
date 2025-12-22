#!/bin/bash
# 重新生成instances目录和测试数据

echo "重新生成测试实例数据..."
cd /home/cswccc/work/Project

# 运行批量测试脚本（只生成数据，不运行所有求解器）
python3 -c "
from data_generator import DataGenerator
import json
import os

# 创建instances目录
os.makedirs('instances', exist_ok=True)

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
    print(f'生成第 {idx+1} 组: n={n}, m={m}')
    for seed_offset in range(10):
        seed = idx * 100 + seed_offset
        generator = DataGenerator(seed=seed)
        instance = generator.generate_instance(
            n=n,
            m=m,
            coverage_rate=0.3
        )
        instance['problem_id'] = f'P{idx+1}_S{seed_offset+1}'
        instance['seed'] = seed
        
        # 保存到文件
        filename = f'instances/{instance[\"problem_id\"]}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
        instances.append(instance)

print(f'\n成功生成 {len(instances)} 个测试实例到 instances/ 目录')
"

echo "完成！"
