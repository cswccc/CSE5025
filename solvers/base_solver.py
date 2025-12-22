"""
基础求解器类

本模块定义了所有求解器的基类，提供了公共的方法和接口。
所有具体的求解器都继承自BaseSolver，并实现solve()方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class BaseSolver(ABC):
    """
    求解器基类
    
    这是一个抽象基类，定义了求解器的基本接口和公共方法。
    所有具体的求解算法都应该继承此类并实现solve()方法。
    
    属性说明:
        instance: 问题实例字典，包含所有输入参数
        n: 楼栋数量
        m: 区域数量  
        D: 每栋楼的潜在用户需求数组 (n,)
        p: 每栋楼的单位收益数组 (n,)
        c: 每个区域的建设成本数组 (m,)
        U: 每个区域的容量上限数组 (m,)
        a: 覆盖关系矩阵 (n, m)，a[i,j]=1表示区域j可以覆盖楼栋i
        best_solution: 找到的最佳解（字典格式，包含z, x, y）
        best_objective: 最佳解的目标函数值
        solve_time: 求解耗时（秒）
    """
    
    def __init__(self, instance: Dict):
        """
        初始化求解器
        
        Args:
            instance: 问题实例字典，必须包含以下键:
                - 'n': 楼栋数量 (int)
                - 'm': 区域数量 (int)
                - 'D': 需求列表 (list[float])
                - 'p': 单位收益列表 (list[float])
                - 'c': 建设成本列表 (list[float])
                - 'U': 容量上限列表 (list[int])
                - 'a': 覆盖关系矩阵 (list[list[int]])
        """
        self.instance = instance
        self.n = instance['n']  # 楼栋数量
        self.m = instance['m']  # 区域数量
        self.D = np.array(instance['D'])  # 需求数组 (n,)
        self.p = np.array(instance['p'])  # 单位收益数组 (n,)
        self.c = np.array(instance['c'])  # 建设成本数组 (m,)
        self.U = np.array(instance['U'])  # 容量上限数组 (m,)
        self.a = np.array(instance['a'])  # 覆盖关系矩阵 (n, m)
        
        # 初始化最佳解相关属性
        self.best_solution = None  # 最佳解字典 {'z': [...], 'x': [...], 'y': [[...]]}
        self.best_objective = float('-inf')  # 最佳目标函数值
        self.solve_time = 0.0  # 求解耗时
    
    @abstractmethod
    def solve(self) -> Tuple[Dict, float]:
        """
        求解问题的抽象方法
        
        所有子类必须实现此方法，用于执行具体的求解算法。
        
        Returns:
            Tuple[Dict, float]: 
                - solution: 解字典，包含:
                    - 'z': 建设决策列表 (list[int])，z[j]=1表示在区域j建设
                    - 'x': 充电桩数量列表 (list[float])，x[j]表示区域j的充电桩数
                    - 'y': 用户分配矩阵 (list[list[float]])，y[i][j]表示楼栋i分配到区域j的用户数
                - objective_value: 目标函数值（净收益），float类型
        """
        pass
    
    def calculate_objective(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算目标函数值（净收益）
        
        目标函数 = 总收益 - 总成本
        总收益 = Σ_i Σ_j (p_i * y_ij)  所有用户带来的收益总和
        总成本 = Σ_j (c_j * z_j)        所有区域的建设成本总和
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]∈{0,1}表示是否在区域j建设
            x: 充电桩数量数组，形状为(m,)，x[j]≥0表示区域j的充电桩数量
            y: 用户分配矩阵，形状为(n, m)，y[i,j]≥0表示楼栋i分配到区域j的用户人数
            
        Returns:
            float: 净收益值 = 总收益 - 总成本
            
        注意: x参数在本函数中实际上没有被使用（充电桩数量不影响目标函数），
             但保留此参数以保持接口一致性。
        """
        # 计算总收益: 对每个楼栋i和区域j，收益 = p_i * y_ij，然后求和
        # self.p[:, np.newaxis]将p从(n,)扩展为(n,1)，以便与y(n,m)进行广播运算
        total_revenue = np.sum(self.p[:, np.newaxis] * y)
        
        # 计算总成本: 对所有建设的区域（z[j]=1），累加建设成本c_j
        total_cost = np.sum(self.c * z)
        
        # 净收益 = 总收益 - 总成本
        net_profit = total_revenue - total_cost
        
        return net_profit
    
    def is_feasible(self, z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """
        检查解的可行性
        
        验证解是否满足所有约束条件：
        1. 变量类型约束
        2. 覆盖关系约束: y_ij ≤ D_i * a_ij * z_j
        3. 需求约束: Σ_j y_ij ≤ D_i
        4. 容量约束: Σ_i y_ij ≤ x_j
        5. 容量上限约束: 0 ≤ x_j ≤ U_j * z_j
        
        Args:
            z: 建设决策数组，形状为(m,)，应为0-1整数
            x: 充电桩数量数组，形状为(m,)，应为非负
            y: 用户分配矩阵，形状为(n, m)，应为非负
            
        Returns:
            Tuple[bool, str]: 
                - is_feasible: 布尔值，True表示可行，False表示不可行
                - error_message: 字符串，如果不可行则说明违反的约束，否则为"可行"
        """
        # 约束1: 检查z是否为0-1变量
        # z中的所有元素必须为0或1
        if not np.all((z == 0) | (z == 1)):
            return False, "z必须为0-1变量"
        
        # 约束2: 检查x是否非负
        if np.any(x < 0):
            return False, "x必须非负"
        
        # 约束3: 检查y是否非负
        if np.any(y < 0):
            return False, "y必须非负"
        
        # 约束4: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        # 含义：只有区域j被建设（z_j=1）且可以覆盖楼栋i（a_ij=1）时，才能分配用户
        # 并且分配的用户数不能超过楼栋i的需求D_i
        for i in range(self.n):
            for j in range(self.m):
                # 计算允许的最大分配量：如果a_ij=1且z_j=1，则为D_i；否则为0
                max_allocation = self.D[i] * self.a[i, j] * z[j]
                # 使用1e-6容差来处理浮点数精度问题
                if y[i, j] > max_allocation + 1e-6:
                    return False, f"违反覆盖关系约束: y[{i},{j}] > D[{i}] * a[{i},{j}] * z[{j}]"
        
        # 约束5: 需求约束 Σ_j y_ij ≤ D_i
        # 含义：每栋楼被分配的用户总数不能超过其潜在需求
        for i in range(self.n):
            total_assigned = np.sum(y[i, :])  # 楼栋i被分配到所有区域的总用户数
            if total_assigned > self.D[i] + 1e-6:
                return False, f"违反需求约束: 楼栋{i}的用户分配超过需求"
        
        # 约束6: 容量约束 Σ_i y_ij ≤ x_j
        # 含义：区域j服务的用户总数不能超过其设置的充电桩数量
        for j in range(self.m):
            total_served = np.sum(y[:, j])  # 区域j服务的所有楼栋的总用户数
            if total_served > x[j] + 1e-6:
                return False, f"违反容量约束: 区域{j}的服务量超过容量"
        
        # 约束7: 容量上限约束 0 ≤ x_j ≤ U_j * z_j
        # 含义：充电桩数量必须非负，且只有在区域被建设时才能设置充电桩
        # 如果z_j=0，则x_j必须为0；如果z_j=1，则x_j不能超过U_j
        for j in range(self.m):
            if x[j] < 0:
                return False, f"违反非负约束: x[{j}] < 0"
            max_capacity = self.U[j] * z[j]  # 如果z_j=0则为0，如果z_j=1则为U_j
            if x[j] > max_capacity + 1e-6:
                return False, f"违反容量上限约束: x[{j}] > U[{j}] * z[{j}]"
        
        # 所有约束都满足
        return True, "可行"
    
    def optimal_assign_given_z(self, z: np.ndarray, method: str = 'mincostflow') -> Tuple[np.ndarray, np.ndarray, float]:
        """
        给定建设决策z，最优分配用户y（x直接装满为U_j*z_j）
        
        策略：
        1. 设置x_j = U_j * z_j（选址后充电桩直接装满）
        2. 使用最小费用最大流、MIP或贪心方法最优分配y
        
        Args:
            z: 建设决策数组，形状为(m,)，z[j]∈{0,1}表示是否在区域j建设
            method: 求解方法，可选：
                - 'mincostflow': 最小费用最大流（默认，多项式时间，理论上最快）
                - 'mip': 混合整数规划（保证最优解，但可能较慢）
                - 'greedy': 贪心方法（快速但可能不是最优）
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - x: 充电桩数量数组，x[j] = U[j] * z[j]（整数）
                - y: 用户分配矩阵（整数），最优分配
                - objective: 目标函数值
        """
        # 步骤1: 设置x_j = U_j * z_j（直接装满）
        x = (self.U * z).astype(int)
        
        # 步骤2: 根据指定方法分配y
        if method == 'mincostflow':
            try:
                import networkx as nx
                return self._optimal_assign_given_z_mincostflow(z, x)
            except ImportError:
                # 如果没有NetworkX，降级到MIP
                method = 'mip'
        
        if method == 'mip':
            try:
                import pulp
                return self._optimal_assign_given_z_mip(z, x)
            except ImportError:
                # 如果没有PuLP，降级到贪心方法
                return self._optimal_assign_given_z_greedy(z, x)
        
        if method == 'greedy':
            return self._optimal_assign_given_z_greedy(z, x)
        
        # 默认使用最小费用最大流
        return self._optimal_assign_given_z_mincostflow(z, x)
    
    def _optimal_assign_given_z_mincostflow(self, z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用最小费用最大流算法精确求解给定z和x的最优y分配
        
        网络结构：
        - 源点s
        - 楼栋节点i（每个楼栋一个节点）
        - 区域节点j（只包括已选中的区域j∈S，其中z[j]=1）
        - 汇点t
        
        边：
        - s → i: 容量D_i，费用0
        - i → j: 容量D_i（如果a_ij=1且z[j]=1），费用-p_i
        - j → t: 容量U_j，费用0
        - i → t: 容量D_i，费用0（未被服务的用户）
        
        Args:
            z: 建设决策数组
            x: 充电桩数量数组（x[j] = U[j] * z[j]）
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: x, y, objective
        """
        import networkx as nx
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 节点命名
        source = 's'
        sink = 't'
        
        # 获取已选中的区域集合S
        S = [j for j in range(self.m) if z[j] == 1]
        
        # 计算总需求（用于源点和汇点的供给/需求）
        total_demand = int(np.sum(self.D))
        
        # 添加边
        
        # 1. 源点到楼栋的边：s → i，容量D_i，费用0
        for i in range(self.n):
            G.add_edge(source, f'i_{i}', capacity=int(self.D[i]), weight=0)
        
        # 2. 楼栋到区域的边：i → j（如果a_ij=1且j∈S），容量D_i，费用-p_i
        for i in range(self.n):
            for j in S:
                if self.a[i, j] == 1:
                    # NetworkX中weight表示费用，费用为-p_i（因为要最小化费用以最大化收益）
                    G.add_edge(f'i_{i}', f'j_{j}', capacity=int(self.D[i]), weight=-float(self.p[i]))
        
        # 3. 区域到汇点的边：j → t，容量U_j，费用0
        for j in S:
            G.add_edge(f'j_{j}', sink, capacity=int(x[j]), weight=0)
        
        # 4. 楼栋直接到汇点的边：i → t，容量D_i，费用0（未被服务的用户）
        for i in range(self.n):
            G.add_edge(f'i_{i}', sink, capacity=int(self.D[i]), weight=0)
        
        # 求解最小费用最大流
        # flowDict是一个字典，flowDict[u][v]表示边(u,v)上的流量
        flowDict = nx.max_flow_min_cost(G, source, sink)
        
        # 从流中恢复y分配
        y = np.zeros((self.n, self.m), dtype=int)
        
        for i in range(self.n):
            node_i = f'i_{i}'
            for j in S:
                node_j = f'j_{j}'
                if node_j in flowDict.get(node_i, {}):
                    y[i, j] = int(flowDict[node_i][node_j])
        
        # 计算总费用（注意：NetworkX返回的是最小费用，但我们需要最大收益）
        min_cost = nx.cost_of_flow(G, flowDict)
        # 由于我们在边i→j上设置的费用是-p_i，所以：
        # min_cost = -Σ p_i * y_ij
        # 因此最大收益 = -min_cost
        max_revenue = -min_cost
        
        # 计算完整的目标函数值（包括成本项）
        objective = self.calculate_objective(z, x, y)
        
        return x, y, objective
    
    def _optimal_assign_given_z_mip(self, z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用MIP精确求解给定z和x的最优y分配
        """
        import pulp
        
        # 创建MIP问题（最大化问题）
        prob = pulp.LpProblem("OptimalAssign", pulp.LpMaximize)
        
        # 定义决策变量：y_ij（整数）
        y = [[pulp.LpVariable(f'y_{i}_{j}', lowBound=0, cat='Integer')
              for j in range(self.m)] for i in range(self.n)]
        
        # 目标函数: max Σ_i Σ_j (p_i * y_ij)
        prob += pulp.lpSum([self.p[i] * y[i][j] 
                           for i in range(self.n) for j in range(self.m)])
        
        # 约束1: 覆盖关系约束 y_ij ≤ D_i * a_ij * z_j
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    prob += y[i][j] <= self.D[i]
                else:
                    prob += y[i][j] == 0
        
        # 约束2: 需求约束 Σ_j y_ij ≤ D_i
        for i in range(self.n):
            prob += pulp.lpSum([y[i][j] for j in range(self.m)]) <= self.D[i]
        
        # 约束3: 容量约束 Σ_i y_ij ≤ x_j
        for j in range(self.m):
            prob += pulp.lpSum([y[i][j] for i in range(self.n)]) <= x[j]
        
        # 求解
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        # 提取解
        y_sol = np.array([[int(round(pulp.value(y[i][j]))) if pulp.value(y[i][j]) is not None else 0
                          for j in range(self.m)] for i in range(self.n)], dtype=int)
        
        # 计算目标函数值（包括成本项）
        objective = self.calculate_objective(z, x, y_sol)
        
        return x, y_sol, objective
    
    def _optimal_assign_given_z_greedy(self, z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用贪心方法近似分配y（当MIP不可用时使用）
        """
        # 初始化y为全零
        y = np.zeros((self.n, self.m), dtype=int)
        
        # 构建可分配的(楼栋, 区域)对列表，按单位收益排序
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                # 只有当区域j已建设且可以覆盖楼栋i时，才能分配
                if self.a[i, j] == 1 and z[j] == 1:
                    allocations.append((i, j, self.p[i]))
        
        # 按单位收益从高到低排序
        allocations.sort(key=lambda item: item[2], reverse=True)
        
        # 贪心分配用户
        remaining_demand = self.D.copy().astype(int)  # 剩余需求
        remaining_capacity = x.copy()  # 剩余容量（初始为x，即U_j*z_j）
        
        for i, j, profit in allocations:
            if remaining_demand[i] <= 0 or remaining_capacity[j] <= 0:
                continue
            
            # 分配尽可能多的用户（受剩余需求和剩余容量限制）
            amount = min(remaining_demand[i], remaining_capacity[j])
            y[i, j] = amount
            remaining_demand[i] -= amount
            remaining_capacity[j] -= amount
        
        # 计算目标函数值
        objective = self.calculate_objective(z, x, y)
        
        return x, y, objective
