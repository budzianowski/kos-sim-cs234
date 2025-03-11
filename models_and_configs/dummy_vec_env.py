import numpy as np
import gym
from gym import spaces

class DummyVecEnv:
    def __init__(self, num_envs=1, obs_dim=36, action_dim=10):
        self.num_envs = num_envs
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_dim)
        self.current_step = 0
        self.num_privileged_obs = None
        self.num_obs = obs_dim
        self.num_actions = action_dim

    def reset(self):
        """Resets the environment and returns an initial observation."""
        self.current_step = 0
        return np.random.uniform(-1, 1, (self.num_envs, self.observation_space.shape[0])).astype(np.float32), None

    def step(self, actions):
        """Takes an action and returns next_obs, reward, done, and info."""
        next_obs = np.random.uniform(-1, 1, (self.num_envs, self.observation_space.shape[0])).astype(np.float32)
        reward = np.random.randn(self.num_envs)
        done = np.array([self.current_step >= 100 for _ in range(self.num_envs)])
        info = [{} for _ in range(self.num_envs)]
        self.current_step += 1
        return next_obs, reward, done, info

    def close(self):
        """Closes the environment (dummy implementation)."""
        pass

