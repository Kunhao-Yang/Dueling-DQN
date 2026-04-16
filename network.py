"""
Dueling DQN 网络结构

将Q(s,a)分解为 V(s) + A(s,a)
- V(s): 状态价值函数
- A(s,a): 优势函数
"""

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture
    
    结构:
        state → 共享特征层 → Value流 → V(s)
                              ↘
                                合并 → Q(s,a)
                              ↗
                    → Advantage流 → A(s,a)
    
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value流: 输出标量 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage流: 输出向量 A(s,a1), ..., A(s,an)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态, shape [batch, state_dim]
        
        Returns:
            Q值, shape [batch, action_dim]
        """
        # 提取共享特征
        feature = self.feature(x)
        
        # 计算Value和Advantage
        value = self.value_stream(feature)           # [batch, 1]
        advantage = self.advantage_stream(feature)   # [batch, action_dim]
        
        # 合并: Q = V + A - mean(A)
        # 中心化Advantage使分解唯一
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q


class DuelingDQNv2(nn.Module):
    """
    另一种实现: 使用单层hidden，更轻量
    
    适用于简单环境如CartPole
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # 共享层
        self.shared = nn.Linear(state_dim, hidden_dim)
        
        # Value流
        self.value = nn.Linear(hidden_dim, 1)
        
        # Advantage流
        self.advantage = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.shared(x))
        
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


if __name__ == "__main__":
    # 测试网络
    import numpy as np
    
    # 创建网络
    state_dim = 4
    action_dim = 2
    net = DuelingDQN(state_dim, action_dim)
    
    # 测试前向传播
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    q_values = net(states)
    
    print(f"输入: {states.shape}")
    print(f"输出: {q_values.shape}")
    print(f"Q值示例: {q_values[0].detach().numpy()}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n总参数量: {total_params}")
