import gym
import numpy as np
from torch import nn
import torch
import copy

from mcts import MCTS

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

        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        x = torch.relu(self.fc2(x))
        policy = self.fc3(x)
        x = torch.relu(self.fc4(x))
        value = torch.tanh(self.fc5(x))
        return policy, value

    def predict(self, observation):
        input = torch.tensor(gym.spaces.flatten(self.observation_space, observation)).unsqueeze(0).to(torch.float32)
        policy_pred, value_pred = self.forward(input)

        policy_pred = torch.nn.functional.softmax(policy_pred, 1)

        return policy_pred.squeeze().detach().numpy(), value_pred.detach().numpy().item()

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
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")

    def update_pvnet(self, pvnet):
        if isinstance(pvnet, str):
            pvnet = torch.load(pvnet)
        self.mcts.update_pvnet(pvnet)

    def reset(self, env_state):
        self.mcts.reset(env_state=env_state)
    
    def compute_action(self, observation, env_state):
        probs = self.mcts.compute_probs(observation, env_state)
        action = np.argmax(probs)
        return action, probs
    
    def _loss(self, policy_preds, value_preds, policy_targets, value_targets):
        policy_loss = self.KLDiv(torch.nn.functional.log_softmax(policy_preds), torch.tensor(policy_targets).to(torch.float32))
        value_loss = torch.nn.functional.mse_loss(value_preds, torch.tensor(value_targets).to(torch.float32))
        return policy_loss + value_loss
    
    def train(self, buffer, batch_size):
        buffer.post_process()
        optimizer = torch.optim.Adam(self.mcts.pvnet.parameters(), lr=0.001)

        for epoch in range(10):
            batches = buffer.get_random_batches(batch_size = batch_size)
            for batch in batches:
                obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, probs = batch
                
                policy_preds, value_preds = self.mcts.pvnet.forward(torch.tensor(obss).to(torch.float32))
                loss = self._loss(
                    policy_preds=policy_preds,
                    value_preds=value_preds,
                    policy_targets=probs,
                    value_targets=values
                )
                print(f"epoch: {epoch} | loss: {loss}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.mcts.pvnet