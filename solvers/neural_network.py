"""
神经网络方法求解器
使用强化学习方法训练策略网络
"""

import numpy as np
import time
from typing import Dict, Tuple
from .base_solver import BaseSolver

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，神经网络求解器不可用。请运行: pip install torch")


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkSolver(BaseSolver):
    """神经网络求解器（使用策略梯度方法）"""
    
    def __init__(self, instance: Dict,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 num_episodes: int = 1000,
                 batch_size: int = 32):
        """
        初始化神经网络求解器
        
        Args:
            instance: 问题实例
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            num_episodes: 训练回合数
            batch_size: 批次大小
        """
        super().__init__(instance)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，请运行: pip install torch")
        
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        
        # 创建策略网络（输入为区域特征，输出为选择概率）
        # 特征维度：区域j的特征包括成本、容量、覆盖楼栋数、平均收益等
        input_dim = 4  # [cost, capacity, num_covered, avg_profit]
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, 1)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
    
    def solve(self) -> Tuple[Dict, float]:
        """
        使用神经网络方法求解
        
        Returns:
            (solution, objective_value)
        """
        start_time = time.time()
        
        # 提取区域特征
        region_features = self._extract_features()
        
        # 训练策略网络
        self._train_policy(region_features)
        
        # 使用训练好的网络生成解
        z = self._generate_solution(region_features)
        
        # 解码解
        x, y, obj = self._decode_individual(z)
        
        self.solve_time = time.time() - start_time
        self.best_solution = {
            'z': z.tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        }
        self.best_objective = obj
        
        return self.best_solution, self.best_objective
    
    def _extract_features(self) -> np.ndarray:
        """提取区域特征"""
        features = []
        for j in range(self.m):
            covered_buildings = np.where(self.a[:, j] == 1)[0]
            num_covered = len(covered_buildings)
            avg_profit = np.mean(self.p[covered_buildings]) if num_covered > 0 else 0
            
            # 归一化特征
            cost_norm = self.c[j] / (np.max(self.c) + 1e-6)
            capacity_norm = self.U[j] / (np.max(self.U) + 1e-6)
            num_covered_norm = num_covered / (self.n + 1e-6)
            avg_profit_norm = avg_profit / (np.max(self.p) + 1e-6)
            
            features.append([cost_norm, capacity_norm, num_covered_norm, avg_profit_norm])
        
        return np.array(features, dtype=np.float32)
    
    def _train_policy(self, features: np.ndarray):
        """训练策略网络"""
        features_tensor = torch.FloatTensor(features)
        
        for episode in range(self.num_episodes):
            # 前向传播
            probs = self.policy_net(features_tensor).squeeze()
            
            # 采样决策
            z = (torch.rand(self.m) < probs).float()
            
            # 计算奖励（目标函数值）
            z_np = z.detach().numpy().astype(int)
            x_np, y_np, reward = self._decode_individual(z_np)
            
            # 计算损失（策略梯度）
            log_probs = torch.log(probs + 1e-8) * z + torch.log(1 - probs + 1e-8) * (1 - z)
            loss = -torch.sum(log_probs) * reward  # 简单的策略梯度
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {reward:.2f}")
    
    def _generate_solution(self, features: np.ndarray) -> np.ndarray:
        """使用训练好的网络生成解"""
        features_tensor = torch.FloatTensor(features)
        self.policy_net.eval()
        
        with torch.no_grad():
            probs = self.policy_net(features_tensor).squeeze().numpy()
        
        # 根据概率选择区域（贪心：选择概率高的）
        # 可以尝试多次采样，选择最好的
        best_z = None
        best_obj = float('-inf')
        
        for _ in range(10):  # 采样10次
            z = (np.random.rand(self.m) < probs).astype(int)
            x, y, obj = self._decode_individual(z)
            if obj > best_obj:
                best_obj = obj
                best_z = z.copy()
        
        return best_z if best_z is not None else (probs > 0.5).astype(int)
    
    def _decode_individual(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """解码个体"""
        x = np.zeros(self.m, dtype=float)
        y = np.zeros((self.n, self.m), dtype=float)
        
        remaining_demand = self.D.copy()
        remaining_capacity = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if z[j] == 1:
                remaining_capacity[j] = self.U[j]
        
        # 贪心分配
        allocations = []
        for i in range(self.n):
            for j in range(self.m):
                if self.a[i, j] == 1 and z[j] == 1:
                    allocations.append((i, j, self.p[i]))
        
        allocations.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, profit in allocations:
            if remaining_demand[i] <= 0 or remaining_capacity[j] <= 0:
                continue
            amount = min(remaining_demand[i], remaining_capacity[j])
            y[i, j] = amount
            remaining_demand[i] -= amount
            remaining_capacity[j] -= amount
        
        for j in range(self.m):
            if z[j] == 1:
                x[j] = np.sum(y[:, j])
        
        objective = self.calculate_objective(z, x, y)
        return x, y, objective
