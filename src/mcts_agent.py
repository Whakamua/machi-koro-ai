import gym
import numpy as np
from torch import nn
# import torch.functional as F
from mcts import MCTS


class RandomAgent:
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

    def compute_action(self, observation):
        action_mask = observation["action_mask"]
        prob_dist = action_mask/sum(action_mask)
        return np.random.choice(range(self.action_space.n), p=prob_dist)

class PVNet(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
    ):
        super(PVNet, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def predict(self, observation):
        return observation["action_mask"] / observation["action_mask"].sum(), 0

class MCTSAgent:
    def __init__(
            self,
            env,
            num_mcts_sims,
            c_puct,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        nnet = PVNet(observation_space=self.observation_space, action_space=self.action_space)
        self.mcts = MCTS(env, nnet, num_mcts_sims, c_puct)

    def reset(self, env_state):
        self.mcts.reset(env_state=env_state)
    
    def compute_action(self, observation, env_state):
        probs = self.mcts.compute_probs(observation, env_state)
        action = np.argmax(probs)
        self.mcts.set_root(self.mcts.root.children[action])
        return action