import gym
import numpy as np
from torch import nn
import torch
import warnings

from mcts import MCTS
import itertools


class MultiDimensionalOneHot:
    def __init__(self, values):
        # np array where each row is padded with nans
        self.values = torch.tensor(list(itertools.zip_longest(*values, fillvalue=np.nan))).T

        # the one_hot_start_indices represent the indices in a flattened array. Each index is the
        # start index of a onehot dimension.
        values_lengths = torch.sum(~torch.isnan( self.values), axis=1)
        self.one_hot_start_indices = values_lengths.cumsum(0) - values_lengths
        
        # the lenght of the flattened onehot array
        self.one_hot_len = sum(map(len, values))
        # self.start_indices = (np.cumsum(n_elements) - n_elements)

    def to_onehot(self, array):
        # one hot initialized by zeros
        one_hot = torch.zeros((len(array), self.one_hot_len))

        # The following line figures out what indices in the flattened onehot array need to be 
        # marked as `1`. It does that by taking the array and figure out the indices at which it
        # equals the self.values array. This results 3 array but only the last one is interesting,
        # therefore [-1] is used. This one is flattened for each row in `array` however, so it is
        # reshaped to (len(array), len(self.values)). Now only the indices for each row in 
        # self.values are known, so the one_hot_start_indices are added to find the indices in the
        # flattened one_hot.
        try:
            onehot_indices = torch.where(self.values == array[:,:,None])[-1].reshape(len(array), len(self.values)) + self.one_hot_start_indices
        except:
            # [print(row) for row in torch.cat((self.values, array.T), axis=1)]
            breakpoint()
        
        one_hot[torch.arange(len(array))[:, None], onehot_indices] = 1
        return one_hot


class PVNet(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            info: dict,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._info = info
        self._landmark_indices_in_action = [self._info["action_str_to_idx"][landmark] for landmark in self._info["landmarks"]]
        # num_inputs = gym.spaces.flatten_space(observation_space).shape[0]

        one_hot_indices = []
        one_hot_values = []
        identity_indices = []
        for player in info["observation_indices"]["player_info"].keys():
            for card in info["observation_indices"]["player_info"][player]["cards"].keys():
                card_index = info["observation_indices"]["player_info"][player]["cards"][card]
                card_values = info["observation_values"]["player_info"][player]["cards"][card]
                one_hot_indices.append(card_index)
                one_hot_values.append(card_values)

            identity_indices.append(info["observation_indices"]["player_info"][player]["coins"])
            identity_indices.append(info["observation_indices"]["player_info"][player]["tech_startup_investment"])

        for alley in info["observation_indices"]["marketplace"].keys():
            for pos in info["observation_indices"]["marketplace"][alley].keys():
                card_index = info["observation_indices"]["marketplace"][alley][pos]["card"]
                card_values = info["observation_values"]["marketplace"][alley][pos]["card"]
                one_hot_indices.append(card_index)
                one_hot_values.append(card_values)

                count_index = info["observation_indices"]["marketplace"][alley][pos]["count"]
                count_values = info["observation_values"]["marketplace"][alley][pos]["count"]
                one_hot_indices.append(count_index)
                one_hot_values.append(count_values)

        one_hot_indices.append(info["observation_indices"]["current_player_index"])
        one_hot_values.append(info["observation_values"]["current_player_index"])
        one_hot_indices.append(info["observation_indices"]["current_stage_index"])
        one_hot_values.append(info["observation_values"]["current_stage_index"])

        self._one_hot_indices = list(one_hot_indices)
        self._identity_indices = list(identity_indices)
        self._mdoh = MultiDimensionalOneHot(one_hot_values)

        num_inputs = self._mdoh.one_hot_len + len(self._identity_indices)
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
    def forward(self, x):
        x = torch.cat((self._mdoh.to_onehot(x[:, self._one_hot_indices]), x[:, self._identity_indices]), axis=1)
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

    def predict(self, observation):

        input = torch.tensor(observation).unsqueeze(0).to(torch.float32)
        policy_pred, value_pred = self.forward(input)

        current_stage = self._info["stage_order"][observation[self._info["observation_indices"]["current_stage_index"]]]
        if current_stage == "build":
            current_player = self._info["player_order"][observation[self._info["observation_indices"]["current_player_index"]]]
            landmark_indices = [self._info["observation_indices"]["player_info"][current_player]["cards"][landmark] for landmark in self._info["landmarks"]]
            cost_for_all_unowned_landmarks = np.sum(~observation[landmark_indices].astype(bool)*self._info["landmarks_cost"])
            player_coins = observation[self._info["observation_indices"]["player_info"][current_player]["coins"]]
            if player_coins >= cost_for_all_unowned_landmarks:
                policy_pred[:, self._landmark_indices_in_action] = 1e10

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
    
    def compute_action(self, observation):
        probs = self.mcts.compute_probs(observation)
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
    pred = agent.compute_action(observation=observation)
    probs_preds, value_preds = pvnet.forward(torch.tensor(buffer.obss[0:1000]).to(torch.float32))
