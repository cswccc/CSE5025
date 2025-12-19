"""
测试所有模块是否可以正常导入
"""

print("测试模块导入...")

try:
    print("1. 导入数据生成器...")
    from data_generator import DataGenerator
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("2. 导入基础求解器...")
    from solvers.base_solver import BaseSolver
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("3. 导入贪心算法...")
    from solvers.greedy import GreedySolver
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("4. 导入暴力枚举法...")
    from solvers.brute_force import BruteForceSolver
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("5. 导入MILP求解器...")
    from solvers.milp_solver import MILPSolver
    print("   ✓ 成功（PuLP可用）")
except ImportError:
    print("   ⚠ 跳过（PuLP未安装，可选依赖）")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("6. 导入遗传算法...")
    from solvers.genetic_algorithm import GeneticAlgorithmSolver
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("7. 导入蚁群算法...")
    from solvers.ant_colony import AntColonySolver
    print("   ✓ 成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("8. 导入神经网络方法...")
    from solvers.neural_network import NeuralNetworkSolver
    print("   ✓ 成功（PyTorch可用）")
except ImportError:
    print("   ⚠ 跳过（PyTorch未安装，可选依赖）")
except Exception as e:
    print(f"   ✗ 失败: {e}")

print("\n模块导入测试完成！")
