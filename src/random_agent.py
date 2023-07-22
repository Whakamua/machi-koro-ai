import gym
import numpy as np
from torch import nn
import torch.functional as F

class RandomAgent:
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, env_state = None):
        return

    def compute_action(self, observation, env_state = None):
        action_mask = observation["action_mask"]
        prob_dist = action_mask/sum(action_mask)
        return np.random.choice(range(self.action_space.n), p=prob_dist), prob_dist

class Policy(nn.Module):
    def __init__(self, n_in, n_out):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_in, n_in)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(n_in, n_out)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)