import argparse
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.butterfly import pistonball_v6
import tianshou as ts
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo.utils.conversions import parallel_wrapper_fn

def get_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--device', type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser.parse_args()

def make_env(render_mode=None):
    """Create the Pistonball environment with Gymnasium-compatible wrapper."""
    # Create PettingZoo environment
    env = pistonball_v6.env(
        n_pistons=20, 
        time_penalty=-0.1, 
        continuous=True, 
        render_mode=render_mode
    )
    
    # Convert to parallel environment
    parallel_env = parallel_wrapper_fn(env)()
    
    # Wrap as a Gymnasium environment
    class PistonballGymnasiumWrapper(gym.Env):
        def __init__(self):
            self.parallel_env = parallel_env
            
            # Assuming all agents have same observation and action spaces
            agent_names = list(self.parallel_env.agents)
            self.observation_space = self.parallel_env.observation_spaces[agent_names[0]]
            self.action_space = self.parallel_env.action_spaces[agent_names[0]]
            
        def reset(self, seed=None):
            self.parallel_env.reset(seed=seed)
            obs = {}
            for agent in self.parallel_env.agents:
                obs[agent] = self.parallel_env.state[agent]
            return obs, {}
        
        def step(self, actions):
            obs, rewards, terminations, truncations, infos = self.parallel_env.step(actions)
            
            # Aggregate rewards and check if all agents are done
            total_reward = sum(rewards.values())
            done = all(terminations.values()) or all(truncations.values())
            
            # Return in Gymnasium format
            return obs, total_reward, done, False, infos
        
        def render(self):
            return self.parallel_env.render()
        
        def close(self):
            self.parallel_env.close()
    
    return PistonballGymnasiumWrapper()

def create_network(state_shape, action_shape, hidden_sizes):
    """Create a neural network for the Q-network."""
    return torch.nn.Sequential(
        torch.nn.Linear(np.prod(state_shape), hidden_sizes[0]),
        torch.nn.ReLU(),
        *[torch.nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])],
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_sizes[-1], np.prod(action_shape))
    )

def main():
    # Parse arguments
    args = get_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create training and test environments
    train_envs = ts.env.DummyVectorEnv([make_env for _ in range(args.training_num)])
    test_envs = ts.env.DummyVectorEnv([make_env for _ in range(args.test_num)])
    
    # Get environment information
    env = make_env()
    obs, _ = env.reset()
    
    # Convert observation and action spaces
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    # Create network
    net = create_network(state_shape, action_shape, args.hidden_sizes)
    
    # Create optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # Create policy
    policy = ts.policy.DQNPolicy(
        net, 
        optim, 
        discount_factor=args.gamma, 
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq
    )
    
    # Create buffer
    buffer = ts.data.PrioritizedVectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=len(train_envs),
        prioritized=True
    )
    
    # Create collector for data collection
    train_collector = ts.data.Collector(
        policy, 
        train_envs, 
        buffer, 
        exploration_noise=True
    )
    
    test_collector = ts.data.Collector(policy, test_envs)
    
    # Create logger
    log_path = os.path.join(args.logdir, 'pistonball', 'dqn')
    writer = SummaryWriter(log_path)
    logger = ts.utils.TensorboardLogger(writer)
    
    # Trainer
    result = ts.trainer.offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        collect_per_step=args.collect_per_step,
        episode_per_collect=args.collect_per_step,
        batch_size=args.batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(args.eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(args.eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= 100,
        logger=logger,
        verbose=True,
        test_in_train=True,
    )
    
    # Print training results
    print(f"Training finished. Best reward: {result['best_reward']}")
    
    # Optional: save model
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    
    # Close environments
    train_envs.close()
    test_envs.close()

if __name__ == '__main__':
    main()