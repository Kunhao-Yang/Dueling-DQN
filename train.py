"""
训练脚本

支持环境:
- CartPole-v1
- LunarLander-v3

用法:
    python train.py --env CartPole-v1 --episodes 500
    python train.py --env LunarLander-v3 --episodes 1000 --double
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from agent import DuelingDQNAgent


def train(
    env_name: str = "CartPole-v1",
    episodes: int = 500,
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
    seed: int = 42,
    save_model: bool = True,
    save_plot: bool = True
):
    """
    训练Dueling DQN
    
    Args:
        env_name: Gym环境名称
        episodes: 训练轮数
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
        seed: 随机种子
        save_model: 是否保存模型
        save_plot: 是否保存训练曲线
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 智能体
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        double_dqn=double_dqn
    )
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"设备: {agent.device}")
    print(f"Double DQN: {double_dqn}")
    print("-" * 50)
    
    # 训练记录
    rewards_history = []
    moving_avg = []
    
    # 训练循环
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储转移
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            agent.update()
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 记录
        rewards_history.append(episode_reward)
        moving_avg.append(np.mean(rewards_history[-100:]))
        
        # 打印进度
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Moving Avg: {moving_avg[-1]:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    
    # 统计
    final_avg = np.mean(rewards_history[-100:])
    best_reward = max(rewards_history)
    print("-" * 50)
    print(f"训练完成!")
    print(f"最终100轮平均奖励: {final_avg:.2f}")
    print(f"最佳单轮奖励: {best_reward:.2f}")
    
    # 保存模型
    if save_model:
        os.makedirs("models", exist_ok=True)
        model_name = f"dueling_dqn_{env_name.replace('-', '_').lower()}"
        if double_dqn:
            model_name += "_double"
        model_name += ".pth"
        agent.save(f"models/{model_name}")
        print(f"模型已保存: models/{model_name}")
    
    # 绘制训练曲线
    if save_plot:
        plt.figure(figsize=(12, 5))
        
        # 奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
        plt.plot(moving_avg, color='red', label='Moving Average (100)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Dueling DQN - {env_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最后100轮分布
        plt.subplot(1, 2, 2)
        plt.hist(rewards_history[-100:], bins=20, edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Count')
        plt.title(f'Last 100 Episodes Distribution\nMean: {final_avg:.2f}')
        
        plt.tight_layout()
        
        os.makedirs("plots", exist_ok=True)
        plot_name = f"dueling_dqn_{env_name.replace('-', '_').lower()}"
        if double_dqn:
            plot_name += "_double"
        plot_name += ".png"
        plt.savefig(f"plots/{plot_name}", dpi=150)
        print(f"训练曲线已保存: plots/{plot_name}")
    
    return rewards_history, moving_avg


def evaluate(env_name: str, model_path: str, episodes: int = 100, double_dqn: bool = False):
    """
    评估训练好的模型
    
    Args:
        env_name: 环境名称
        model_path: 模型路径
        episodes: 评估轮数
        double_dqn: 是否使用Double DQN
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=double_dqn
    )
    agent.load(model_path)
    
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    
    env.close()
    
    print(f"评估结果 ({episodes} 轮):")
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"最小/最大: {min(rewards):.2f} / {max(rewards):.2f}")
    
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dueling DQN Training")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--target-update", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--double", action="store_true", help="Use Double DQN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate model path")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args.env, args.eval, double_dqn=args.double)
    else:
        train(
            env_name=args.env,
            episodes=args.episodes,
            hidden_dim=args.hidden,
            lr=args.lr,
            gamma=args.gamma,
            buffer_capacity=args.buffer,
            batch_size=args.batch,
            target_update_freq=args.target_update,
            double_dqn=args.double,
            seed=args.seed
        )
