import gym
import numpy as np
from torch import nn
import torch
import copy
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
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        num_inputs = gym.spaces.flatten_space(observation_space).shape[0]
        num_outputs = action_space.n
        
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_outputs)
        self.softmax = nn.Softmax(dim=1)

        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.softmax(self.fc3(x))
        x = torch.relu(self.fc4(x))
        value = torch.sigmoid(self.fc5(x))
        return policy, value

    def predict(self, observation):
        input = torch.tensor(gym.spaces.flatten(self.observation_space, observation)).unsqueeze(0).to(torch.float32)
        policy_pred, value_pred = self.forward(input)
        return policy_pred.detach().numpy() * observation["action_mask"], value_pred.detach().numpy()

class MCTSAgent:
    def __init__(
            self,
            env,
            num_mcts_sims,
            c_puct,
    ):
        env = copy.deepcopy(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        nnet = PVNet(observation_space=self.observation_space, action_space=self.action_space)
        self.mcts = MCTS(env, nnet, num_mcts_sims, c_puct)

    def reset(self, env_state):
        self.mcts.reset(env_state=env_state)
    
    def compute_action(self, observation, env_state):
        probs = self.mcts.compute_probs(observation, env_state)
        action = np.argmax(probs)
        return action, probs
    
    def train(self, buffer):
        buffer.compute_values()
        optimizer = torch.optim.Adam(self.mcts.pvnet.parameters(), lr=0.001)

        for epoch in range(10):
            batches = buffer.get_random_batches(batch_size = 64)
            for batch in batches:
                obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, probs = batch
                
                preds = self.mcts.pvnet.forward(torch.tensor(obss))

        obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, probs = buffer.sample(100)

        breakpoint()