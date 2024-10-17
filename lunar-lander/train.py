import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init()

# Create the configuration for the PPO algorithm
config = (
    PPOConfig()
    .environment(env="LunarLander-v2")  # Gymnasium's LunarLander environment
    .framework("torch")  # Use PyTorch
    .rollouts(num_rollout_workers=2)  # Set the number of workers to parallelize
    .training(train_batch_size=4000, sgd_minibatch_size=64, num_sgd_iter=10)
    .resources(num_gpus=1)  # Adjust this if you have a GPU
)

# Build the algorithm from the configuration
algo = config.build()
result = algo.train()
print(f"Iteration {0}, reward: {result['env_runners']['episode_reward_mean']}")

# Train the algorithm for a certain number of iterations
#for i in range(10):  # Number of training iterations
 #   result = algo.train()

  #  print(f"Iteration {i}, reward: {result['env_runners']['episode_reward_mean']}")

# Save the trained model
algo.save("lunar_lander_ppo_model")

# Clean up Ray resources
ray.shutdown()
