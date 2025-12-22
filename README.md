# 充电桩覆盖收益最大化问题求解

本项目实现了多种算法来解决充电桩选址与配置的组合优化问题，目标是在预算和容量约束条件下，最大化覆盖用户带来的总收益减去建设成本的净收益。

## 问题描述

### 数学模型

给定：
- **n** 栋居民楼，每栋楼有潜在用户需求 $D_i$ 和单位收益 $p_i$
- **m** 个可选充电桩区域，每个区域有建设成本 $c_j$ 和容量上限 $U_j$
- 覆盖关系矩阵 $a_{ij}$：表示区域 $j$ 是否可以覆盖楼栋 $i$

决策变量：
- $z_j \in \{0,1\}$：是否在区域 $j$ 建设
- $x_j \geq 0$：区域 $j$ 设置的充电桩数量
- $y_{ij} \geq 0$：楼栋 $i$ 分配到区域 $j$ 的用户人数

目标函数：
$$\max \sum_{i=1}^{n} \sum_{j=1}^{m} p_i y_{ij} - \sum_{j=1}^{m} c_j z_j$$

约束条件：
1. 覆盖关系约束：$y_{ij} \leq D_i \cdot a_{ij} \cdot z_j$
2. 需求约束：$\sum_j y_{ij} \leq D_i$
3. 容量约束：$\sum_i y_{ij} \leq x_j$
4. 容量上限约束：$0 \leq x_j \leq U_j \cdot z_j$

## 项目结构

```
Project/
├── data_generator.py          # 数据生成模块（独立命令行工具）
├── main.py                    # 主程序（单个实例求解）
├── batch_test.py              # 批量测试脚本
├── requirements.txt           # 依赖包
├── README.md                  # 本文档
├── instances/                 # 生成的测试实例目录
├── solvers/                   # 求解器模块
│   ├── __init__.py
│   ├── base_solver.py        # 基础求解器类
│   ├── brute_force.py        # 暴力枚举法
│   ├── greedy.py             # 贪心算法
│   ├── milp_solver.py        # MILP求解器（PuLP）
│   ├── genetic_algorithm.py  # 遗传算法
│   └── ant_colony.py         # 蚁群算法
└── 充电桩覆盖收益最大化.pdf   # 问题描述文档
```

## 安装依赖

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

主要依赖：
- `numpy`: 数值计算
- `pulp`: MILP求解器（可选，用于精确求解）

## 使用方法

### 1. 生成问题实例

使用 `data_generator.py` 独立生成测试数据：

```bash
# 使用默认配置（生成50个实例到instances目录）
python data_generator.py

# 自定义输出目录和参数
python data_generator.py --output my_data --problem-sizes "10,5;15,8;20,10" --instances-per-size 5

# 生成单个测试实例用于验证
python data_generator.py --test
```

**data_generator.py 命令行参数**：
- `--output, -o`: 输出目录路径（默认: instances）
- `--problem-sizes`: 问题规模，格式 "n1,m1;n2,m2;..." （例如: "10,5;15,8;20,10"）
- `--instances-per-size`: 每个规模生成的实例数量（默认: 10）
- `--coverage-rate`: 覆盖率（默认: 0.3）
- `--unified-profit`: 统一收益值（默认: 5.0）
- `--seed-base`: 种子基础值（默认: 0）
- `--test`: 生成单个测试实例用于验证

### 2. 批量测试

使用 `batch_test.py` 对生成的数据进行批量测试：

```bash
# 使用默认instances目录和所有方法
python batch_test.py

# 指定数据目录
python batch_test.py --data-dir my_data

# 指定运行的方法
python batch_test.py --methods greedy milp genetic

# 指定输出文件
python batch_test.py --output results.csv
```

**batch_test.py 命令行参数**：
- `--data-dir`: 数据目录路径（默认: instances）
- `--methods`: 要运行的求解方法（默认: 所有方法）
  - `brute_force`: 暴力枚举法
  - `greedy`: 贪心算法
  - `milp`: MILP求解器
  - `genetic` 或 `ga`: 遗传算法
  - `ant_colony` 或 `aco`: 蚁群算法
  - `all`: 运行所有方法
- `--output`: 结果输出CSV文件（默认: batch_test_results.csv）

### 3. 单个实例求解

使用 `main.py` 对单个实例进行求解：

```bash
# 生成新实例并求解
python main.py --generate --n 20 --m 10 --methods greedy milp

# 使用已有问题实例
python main.py --instance instance.json --methods all
```

**main.py 命令行参数**：
- `--generate`: 生成新的问题实例
- `--instance FILE`: 指定问题实例JSON文件
- `--n N`: 楼栋数量（默认20）
- `--m M`: 区域数量（默认10）
- `--methods METHOD [METHOD ...]`: 指定要运行的求解方法
- `--output FILE`: 结果输出文件（默认results.json）

## 数据生成逻辑

### 参数说明

数据生成器（`data_generator.py`）生成以下参数：

1. **楼栋需求 $D_i$**: 
   - 范围：[`min_demand`, `max_demand`]（默认10-100）
   - 生成方式：均匀随机整数

2. **单位收益 $p_i$**: 
   - 默认统一收益 `unified_profit=5.0`（所有楼栋相同）
   - 如需随机收益，可设置 `unified_profit=None`，范围：[`min_profit`, `max_profit`]

3. **建设成本 $c_j$**: 
   - 与容量上限近似成正比：$c_j = \\text{cost\\_per\\_capacity} \\times U_j \\times (1 + \\mathcal{N}(0, \\sigma))$
   - 波动服从正态分布，默认系数 `cost_noise_std=0.2`
   - 默认比例系数 `cost_per_capacity=2.0`

4. **容量上限 $U_j$**: 
   - 范围：[`min_capacity`, `max_capacity`]（默认20-200）
   - 生成方式：均匀随机整数

5. **覆盖关系矩阵 $a_{ij}$**: 
   - 生成方式：伯努利分布，每个位置以 `coverage_rate` 的概率为1（默认0.3）
   - 保证每个楼栋至少被一个区域覆盖
   - 保证每个区域至少覆盖一个楼栋

### 数据特性

- **统一收益**: 所有楼栋使用相同的单位收益（默认5.0），符合实际应用场景
- **成本-容量关系**: 建设成本与容量上限近似成正比，带有正态分布波动，更贴近实际情况
- **随机种子控制**: 可通过seed参数控制随机性，确保实验可重复

### 使用示例

**命令行方式**：
```bash
# 生成50个测试实例（5组规模，每组10个）
python data_generator.py --output instances

# 生成自定义规模的测试数据
python data_generator.py --output test_data \
    --problem-sizes "10,5;15,8;20,10" \
    --instances-per-size 5 \
    --unified-profit 5.0 \
    --coverage-rate 0.3
```

**Python API方式**：
```python
from data_generator import DataGenerator

generator = DataGenerator(seed=42)
instance = generator.generate_instance(
    n=20,              # 20栋楼
    m=10,              # 10个区域
    coverage_rate=0.3,  # 30%覆盖率
    unified_profit=5.0  # 统一收益5.0
)
generator.save_instance(instance, "my_instance.json")
```

## 求解方法说明

本项目实现了以下求解方法：

### 1. 暴力枚举法 (Brute Force)

**文件**: `solvers/brute_force.py`

**算法逻辑**:
- 枚举所有可能的建设决策组合（$2^m$ 种方案）
- 对于每种建设方案，使用线性规划（LP）求解最优的用户分配和充电桩配置
- 选择目标函数值最大的解

**优点**:
- 保证找到最优解（对于枚举的范围内）
- 使用LP确保每种建设方案下的最优分配

**缺点**:
- 时间复杂度：$O(2^m)$，仅适用于小规模问题（$m \leq 15$）
- 对于较大规模问题，可能需要较长时间或达到时间限制

**适用场景**: 小规模问题（区域数量 $\leq 15$）

---

### 2. 贪心算法 (Greedy Algorithm)

**文件**: `solvers/greedy.py`

**算法逻辑**:
1. 计算每个区域的"性价比"（潜在收益/建设成本）
2. 按性价比从高到低排序区域
3. 依次尝试添加每个区域，如果目标函数提升则保留
4. 对选中的区域，贪心分配用户（按单位收益从高到低）

**优点**:
- 运行速度快
- 实现简单

**缺点**:
- 不能保证全局最优
- 可能陷入局部最优

**适用场景**: 快速获取可行解，作为其他算法的初始解

---

### 3. MILP求解器 (Mixed Integer Linear Programming)

**文件**: `solvers/milp_solver.py`

**算法逻辑**:
- 将问题建模为标准的混合整数线性规划模型
- 使用PuLP库调用CBC求解器求解
- 得到精确的最优解（或最优界）

**优点**:
- 保证找到最优解（如果求解器收敛）
- 可以给出最优性证明

**缺点**:
- 对于大规模问题可能求解时间很长
- 需要安装PuLP库

**适用场景**: 中等规模问题，需要精确解

---

### 4. 基础遗传算法 (Genetic Algorithm)

**文件**: `solvers/genetic_algorithm.py`

**算法逻辑**:
1. **编码**: 将建设决策 $z$ 编码为二进制串（染色体）
2. **初始化**: 随机生成初始种群
3. **适应度评估**: 对每个个体解码（贪心分配用户）并计算目标函数值
4. **选择**: 轮盘赌选择，适应度高的个体被选中概率大
5. **交叉**: 单点交叉，生成子代
6. **变异**: 随机翻转某些位，引入多样性
7. **精英保留**: 保留最优的个体到下一代
8. 重复步骤3-7直到达到最大代数

**参数**:
- `pop_size`: 种群大小（默认50）
- `max_generations`: 最大代数（默认100）
- `crossover_rate`: 交叉率（默认0.8）
- `mutation_rate`: 变异率（默认0.1）
- `elite_rate`: 精英比例（默认0.1）

**优点**:
- 可以跳出局部最优
- 适合大规模问题

**缺点**:
- 不能保证最优
- 参数需要调优

**适用场景**: 大规模问题，需要近似最优解

---

### 5. 蚁群算法 (Ant Colony Optimization)

**文件**: `solvers/ant_colony.py`

**算法逻辑**:
1. **信息素初始化**: 为每个区域初始化信息素浓度
2. **启发式信息**: 计算每个区域的性价比作为启发式信息
3. **解构造**: 每只蚂蚁根据信息素和启发式信息按概率选择区域
4. **信息素更新**: 
   - 信息素挥发：$\tau_j = (1-\rho)\tau_j$
   - 信息素增强：根据解的质量增加信息素
   - 最优解额外增强
5. 重复步骤3-4直到达到最大迭代次数

**参数**:
- `num_ants`: 蚂蚁数量（默认30，已优化）
- `max_iterations`: 最大迭代次数（默认150，已优化）
- `alpha`: 信息素重要程度（默认1.0）
- `beta`: 启发式信息重要程度（默认2.0）
- `rho`: 信息素挥发系数（默认0.1）
- `q`: 信息素强度（默认100.0）

**优点**:
- 具有良好的全局搜索能力
- 适合组合优化问题
- 参数已针对该问题优化

**缺点**:
- 收敛速度可能较慢
- 参数敏感

**适用场景**: 中等规模问题，需要高质量近似解

---

### 6. 动态规划方法

**注意**: 由于该问题的约束条件复杂（覆盖关系、容量限制、多楼栋多区域），传统的动态规划方法难以直接应用。问题的状态空间太大，且不具有典型的递归最优子结构。因此本项目未实现动态规划方法。

如果问题规模很小且具有特定的结构，可以考虑使用状态压缩动态规划，但通用性较差。

## 结果输出

### main.py 输出格式

程序会输出JSON格式的结果文件，包含：
- 问题实例参数
- 各求解器的结果（目标值、运行时间、可行性、解详情）
- 最佳求解器信息

示例输出：
```json
{
  "instance": {...},
  "results": [
    {
      "solver": "贪心算法",
      "objective": 1234.56,
      "time": 0.0123,
      "feasible": true,
      "solution": {
        "z": [1, 0, 1, ...],
        "x": [10.0, 0.0, 15.0, ...],
        "y": [[...], [...]]
      }
    },
    ...
  ],
  "best_solver": "MILP求解器",
  "best_objective": 1250.78
}
```

### batch_test.py 输出格式

批量测试会输出CSV格式的结果文件，包含：
- 问题ID、规模参数（n, m）、随机种子
- 各求解方法的目标值、运行时间、可行性、建设区域数、总充电桩数

CSV文件可以直接用于数据分析和可视化，便于比较不同方法在不同规模问题上的表现。

## 性能比较

不同方法的适用场景和建议：

| 方法 | 适用规模 | 求解时间 | 解质量 | 保证最优 |
|------|---------|---------|--------|---------|
| 暴力枚举 | 很小 (m≤15) | 长 | 最优 | ✓ |
| 贪心算法 | 任意 | 很短 | 较好 | ✗ |
| MILP | 中小 | 中等 | 最优 | ✓ |
| 遗传算法 | 大规模 | 中等 | 好 | ✗ |
| 蚁群算法 | 中大规模 | 中等 | 很好 | ✗ |

## 工作流程示例

### 完整测试流程

```bash
# 1. 生成测试数据（50个实例，5组规模，每组10个）
python data_generator.py --output instances

# 2. 运行批量测试（所有方法）
python batch_test.py --data-dir instances --methods all

# 3. 查看结果
cat batch_test_results.csv
```

### 快速测试单个实例

```bash
# 生成并测试单个实例
python main.py --generate --n 20 --m 10 --methods greedy milp
```

## 扩展建议

1. **并行化**: 遗传算法和蚁群算法可以并行化加速
2. **混合算法**: 结合多种方法的优点，如用贪心生成初始解，再用元启发式算法优化
3. **启发式改进**: 对得到的解进行局部搜索改进
4. **问题变体**: 考虑更多约束，如预算限制、时间窗口等
5. **数据分析**: 使用batch_test输出的CSV文件进行性能分析和可视化

## 参考文献

- 问题描述见：`充电桩覆盖收益最大化.pdf`
- MILP建模参考：混合整数线性规划相关文献
- 元启发式算法：遗传算法、蚁群算法相关文献

## 许可证

本项目仅用于学术研究和教学目的。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。
