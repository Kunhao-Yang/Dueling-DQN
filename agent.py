"""
Dueling DQN Agent

支持:
- 经验回放 (Experience Replay)
- 目标网络 (Target Network)
- Double DQN (可选)
- Dueling架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from network import DuelingDQN


class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储一个转移"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Dueling DQN 智能体
    
    特性:
    - Dueling网络架构: Q = V + A - mean(A)
    - 经验回放
    - 目标网络
    - 可选Double DQN
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        double_dqn: bool = False,
        device: str = "auto"
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            epsilon_start: ε初始值
            epsilon_end: ε最终值
            epsilon_decay: ε衰减率
            buffer_capacity: 经验回放池容量
            batch_size: 批大小
            target_update_freq: 目标网络更新频率
            double_dqn: 是否使用Double DQN
            device: 设备 ("auto", "cpu", "cuda")
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        # 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 网络
        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 计数器
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        选择动作 (ε-greedy)
        
        Args:
            state: 当前状态
            eval_mode: 是否为评估模式 (不探索)
        
        Returns:
            选择的动作
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移到经验回放池"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """
        更新网络
        
        Returns:
            损失值 (如果未更新返回None)
        """
        # 检查是否有足够样本
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转为Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 在线网络选动作，目标网络评估
                best_actions = self.q_network(next_states).argmax(dim=1)
                q_next = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准DQN: 目标网络选最大
                q_next = self.target_network(next_states).max(dim=1)[0]
            
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        # 计算损失
        loss = nn.MSELoss()(q_values, q_target)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


if __name__ == "__main__":
    # 测试Agent
    agent = DuelingDQNAgent(
        state_dim=4,
        action_dim=2,
        double_dqn=True
    )
    
    print(f"设备: {agent.device}")
    print(f"网络结构:\n{agent.q_network}")
    
    # 测试动作选择
    state = np.random.randn(4)
    action = agent.select_action(state)
    print(f"\n测试动作选择: state → action {action}")
    
    # 测试存储和更新
    agent.store_transition(state, action, 1.0, np.random.randn(4), False)
    print(f"经验回放池大小: {len(agent.replay_buffer)}")
