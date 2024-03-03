import gym
import numpy as np
from torch import nn
import torch
import warnings

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
        
        # self.fc1 = nn.Linear(num_inputs, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, num_outputs)

        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 1)
        self.fctrunk = nn.Linear(num_inputs, 10)
        self.fclogits = nn.Linear(10, num_outputs)
        self.fcvalue = nn.Linear(10, 1)


        self.is_trained = False
        
        self.is_trained = False
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")
        # REMOVE
        self.ones = np.ones(self.action_space.n)/self.action_space.n
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        
        # x = torch.relu(self.fc2(x))
        # policy = self.fc3(x)
        # x = torch.relu(self.fc4(x))
        # value = torch.tanh(self.fc5(x))
        # return policy, value
        trunk = torch.relu(self.fctrunk(x))
        logits = self.fclogits(trunk)
        value = torch.tanh(self.fcvalue(trunk))

        return logits, value

    def predict(self, observation, flattened=False):
        # REMOVE
        return self.ones, 0
        if not flattened:
            observation = gym.spaces.flatten(self.observation_space, observation)
        input = torch.tensor(observation).unsqueeze(0).to(torch.float32)
        policy_pred, value_pred = self.forward(input)

        policy_pred = torch.nn.functional.softmax(policy_pred, 1)

        return policy_pred.squeeze().detach().numpy(), value_pred.detach().numpy().item()
    
    def _loss(self, policy_preds, value_preds, policy_targets, value_targets):
        policy_loss = self.KLDiv(torch.nn.functional.log_softmax(policy_preds), torch.tensor(policy_targets).to(torch.float32))
        value_loss = torch.nn.functional.mse_loss(value_preds, torch.tensor(value_targets).to(torch.float32))
        return policy_loss + value_loss

    def train(self, buffer, batch_size, epochs):
        buffer.compute_values()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            batches = buffer.get_random_batches(batch_size = batch_size, exclude_terminal_states=True)
            for batch in batches:
                obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, probs = batch
                
                policy_preds, value_preds = self.forward(torch.tensor(obss).to(torch.float32))
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

class MCTSAgent:
    def __init__(
            self,
            env,
            num_mcts_sims,
            c_puct,
            pvnet,
            dirichlet_to_root_node = True
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.mcts = MCTS(env, pvnet, num_mcts_sims, c_puct, dirichlet_to_root_node)
        warnings.warn("Not using any temperature in probs, might need that for first n actions")

    def update_pvnet(self, state_dict):
        self.mcts.update_pvnet(state_dict)

    def reset(self, env_state):
        self.mcts.reset(env_state=env_state)
    
    def compute_action(self, observation, env_state):
        probs = self.mcts.compute_probs(observation, env_state)
        action = np.argmax(probs)
        return action, probs
    
    def train(self, buffer, batch_size):
        return self.mcts.pvnet.train(buffer, batch_size)
    
if __name__ == "__main__":
    import pickle
    from env import GymMachiKoro, MachiKoro

    pvnet = torch.load("checkpoints2/4.pt")
    with open(f"checkpoints2/4.pkl", "rb") as file:
        buffer = pickle.load(file)

    env = MachiKoro(n_players=2)
    env = GymMachiKoro(env)
    observation, info = env.reset()

    agent = MCTSAgent(env=env, num_mcts_sims=100, c_puct=2, pvnet=pvnet, dirichlet_to_root_node=True)
    agent.reset(info["state"])

    # pred = pvnet.predict(buffer[-100][0], flattened=True)
    breakpoint()
    pred = agent.compute_action(observation=observation, env_state=info["state"])
    probs_preds, value_preds = pvnet.forward(torch.tensor(buffer.obss[0:1000]).to(torch.float32))
