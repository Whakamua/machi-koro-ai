import numpy as np
from torch import nn
import torch
import warnings

from mcts import MCTS
import itertools
import copy

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
        onehot_indices = torch.where(self.values == array[:,:,None])[-1].reshape(len(array), len(self.values)) + self.one_hot_start_indices
        
        one_hot[torch.arange(len(array))[:, None], onehot_indices] = 1
        return one_hot


class PVNet(nn.Module):
    def __init__(
            self,
            env_cls,
            env_kwargs,
            uniform_pvnet: bool = False,
            custom_policy_edit: bool = False,
            custom_value: bool = False,
    ):
        super().__init__()
        env = env_cls(**env_kwargs)
        self.env = env
        self.use_uniform_pvnet = uniform_pvnet
        self.use_custom_policy_edit = custom_policy_edit
        self.use_custom_value = custom_value
        self._landmark_indices_in_action = [env._action_str_to_idx[landmark] for landmark in env._env._landmarks]
        self.landmarks_cost = [env.card_info[landmark]["cost"] for landmark in env._env._landmarks]
        # num_inputs = gym.spaces.flatten_space(observation_space).shape[0]
        
        self.one_dice_rolls = np.array([1,2,3,4,5,6])
        self.one_dice_probs = np.array([1,1,1,1,1,1])/6
        self.two_dice_rolls = np.array([2,3,4,5,6,7,8,9,10,11,12])
        self.two_dice_probs = np.array([1,2,3,4,5,6,5,4,3,2,1])/36
        

        one_hot_indices = []
        one_hot_values = []
        identity_indices = []
        for player in env.observation_indices["player_info"].keys():
            for card in env.observation_indices["player_info"][player]["cards"].keys():
                card_index = env.observation_indices["player_info"][player]["cards"][card]
                card_values = env.observation_values["player_info"][player]["cards"][card]
                one_hot_indices.append(card_index)
                one_hot_values.append(card_values)

            identity_indices.append(env.observation_indices["player_info"][player]["coins"])

        for alley in env.observation_indices["marketplace"].keys():
            for pos in env.observation_indices["marketplace"][alley].keys():
                card_index = env.observation_indices["marketplace"][alley][pos]["card"]
                card_values = env.observation_values["marketplace"][alley][pos]["card"]
                one_hot_indices.append(card_index)
                one_hot_values.append(card_values)

                count_index = env.observation_indices["marketplace"][alley][pos]["count"]
                count_values = env.observation_values["marketplace"][alley][pos]["count"]
                one_hot_indices.append(count_index)
                one_hot_values.append(count_values)

        one_hot_indices.append(env.observation_indices["current_player_index"])
        one_hot_values.append(env.observation_values["current_player_index"])
        one_hot_indices.append(env.observation_indices["current_stage_index"])
        one_hot_values.append(env.observation_values["current_stage_index"])
        one_hot_indices.append(env.observation_indices["another_turn"])
        one_hot_values.append(env.observation_values["another_turn"])
        one_hot_indices.append(env.observation_indices["build_rounds_left"])
        one_hot_values.append(env.observation_values["build_rounds_left"])

        self._one_hot_indices = list(one_hot_indices)
        self._identity_indices = list(identity_indices)
        self._mdoh = MultiDimensionalOneHot(one_hot_values)

        num_inputs = self._mdoh.one_hot_len + len(self._identity_indices)
        num_outputs = self.env.action_space.n

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

    def expected_coins_diceroll(self, current_player, n_dice, observation):
        self.env.set_state(copy.deepcopy(observation))
        self.env.current_player = copy.deepcopy(current_player)

        current_player = copy.deepcopy(self.env.current_player)
        other_player = copy.deepcopy(self.env.next_players()[0])
        current_player_coins = self.env._env.player_coins(current_player)
        other_player_coins = self.env._env.player_coins(other_player)

        current_player_expected_coins_after_dice_roll = 0
        other_player_expected_coins_after_dice_roll = 0

        assert n_dice in [1, 2]
        roll_amount = self.one_dice_rolls if n_dice == 1 else self.two_dice_rolls
        roll_probs = self.one_dice_probs if n_dice == 1 else self.two_dice_probs
        assert len(self.env.next_players()) == 1

        for diceroll, prob in zip(roll_amount, roll_probs):
            self.env.set_state(copy.deepcopy(observation))
            self.env.current_player = copy.deepcopy(current_player)
            other_player = copy.deepcopy(self.env.next_players()[0])
            self.env._env._earn_income(diceroll)
            current_player_expected_coins_after_dice_roll += self.env._env.player_coins(current_player) * prob
            other_player_expected_coins_after_dice_roll += self.env._env.player_coins(other_player) * prob

        expected_coins_current_player = current_player_expected_coins_after_dice_roll - current_player_coins
        expected_coins_other_player = other_player_expected_coins_after_dice_roll - other_player_coins
        return expected_coins_current_player, expected_coins_other_player

    def custom_policy_edit(self, observation, policy_pred):
        self.env.set_state(copy.deepcopy(observation))
        current_player = copy.deepcopy(self.env.current_player)
        current_stage = copy.deepcopy(self.env.current_stage)
        
        # if all landmarks that are left to be bought, can be bought, at least 1 must be bought.
        if current_stage == "build":
            landmark_indices = [self.env.observation_indices["player_info"][current_player]["cards"][landmark] for landmark in self.env._env._landmarks]
            cost_for_all_unowned_landmarks = np.sum(~observation[landmark_indices].astype(bool)*self.landmarks_cost)
            player_coins = observation[self.env.observation_indices["player_info"][current_player]["coins"]]
            if player_coins >= cost_for_all_unowned_landmarks:
                policy_pred[:, self._landmark_indices_in_action] = 1e10

        # maximize expected income when choosing dice.
        elif current_stage == "diceroll":
            expected_gain_current_player, expected_gain_other_player = self.expected_coins_diceroll(current_player=current_player, n_dice=1, observation=observation)
            expected_gain_one_dice = expected_gain_current_player - expected_gain_other_player
            expected_gain_current_player, expected_gain_other_player = self.expected_coins_diceroll(current_player=current_player, n_dice=2, observation=observation)
            expected_gain_two_dice = expected_gain_current_player - expected_gain_other_player
            
            if expected_gain_one_dice >= expected_gain_two_dice:
                policy_pred[:, self.env._action_str_to_idx["1 dice"]] = 1e10
            else:
                policy_pred[:, self.env._action_str_to_idx["2 dice"]] = 1e10

        else:
            assert False, "unexpected stage"

        return policy_pred

    
    def custom_value(self, observation):
        self.env.set_state(copy.deepcopy(observation))
        current_player = copy.deepcopy(self.env.current_player)
        other_player = copy.deepcopy(self.env.next_players()[0])
        current_player_coins = self.env._env.player_coins(current_player)
        other_player_coins = self.env._env.player_coins(other_player)

        # from current player's perspective
        current_p_coins_when_current_p_throw_1d, other_p_coins_when_current_p_throw_1d = self.expected_coins_diceroll(current_player=current_player, n_dice=1, observation=observation)
        current_player_expected_gain_one_dice = current_p_coins_when_current_p_throw_1d - other_p_coins_when_current_p_throw_1d
        current_p_coins_when_current_p_throw_2d, other_p_coins_when_current_p_throw_2d = self.expected_coins_diceroll(current_player=current_player, n_dice=2, observation=observation)
        current_player_expected_gain_two_dice = current_p_coins_when_current_p_throw_2d - other_p_coins_when_current_p_throw_2d

        if current_player_expected_gain_one_dice > current_player_expected_gain_two_dice:
            current_player_expected_coins_per_turn = current_p_coins_when_current_p_throw_1d
        elif current_player_expected_gain_one_dice == current_player_expected_gain_two_dice:
            current_player_expected_coins_per_turn = current_p_coins_when_current_p_throw_1d
        else:
            current_player_expected_coins_per_turn = current_p_coins_when_current_p_throw_2d

        # from other player's perspective
        other_p_coins_when_other_p_throw_1d, current_p_coins_when_other_p_throw_1d = self.expected_coins_diceroll(current_player=other_player, n_dice=1, observation=observation)
        other_player_expected_gain_one_dice = other_p_coins_when_other_p_throw_1d - current_p_coins_when_other_p_throw_1d
        other_p_coins_when_other_p_throw_2d, current_p_coins_when_other_p_throw_2d = self.expected_coins_diceroll(current_player=other_player, n_dice=2, observation=observation)
        other_player_expected_gain_two_dice = other_p_coins_when_other_p_throw_2d - current_p_coins_when_other_p_throw_2d

        if other_player_expected_gain_one_dice > other_player_expected_gain_two_dice:
            other_player_expected_coins_per_turn = other_p_coins_when_other_p_throw_1d
        elif other_player_expected_gain_one_dice == other_player_expected_gain_two_dice:
            other_player_expected_coins_per_turn = other_p_coins_when_other_p_throw_1d
        else:
            other_player_expected_coins_per_turn = other_p_coins_when_other_p_throw_2d

        landmark_indices_current_player = [self.env.observation_indices["player_info"][current_player]["cards"][landmark] for landmark in self.env._env._landmarks]
        landmark_indices_other_player = [self.env.observation_indices["player_info"][other_player]["cards"][landmark] for landmark in self.env._env._landmarks]
        current_player_cost_for_all_unowned_landmarks = np.sum(~observation[landmark_indices_current_player].astype(bool)*self.landmarks_cost)
        other_player_cost_for_all_unowned_landmarks = np.sum(~observation[landmark_indices_other_player].astype(bool)*self.landmarks_cost)
        
        # Use sigmoid to make current_player_expected_coins_per_turn non-negative
        current_player_expected_gain_per_turn_sigmoid = 1 / (1 + np.exp(-current_player_expected_coins_per_turn))
        other_player_expected_gain_per_turn_sigmoid = 1 / (1 + np.exp(-other_player_expected_coins_per_turn))

        # because of the sigmoid, this is just a proxy, taking the max so that values of 0 are not possible in the log later on.
        expected_turns_for_winning_money_current_player = max(1, current_player_cost_for_all_unowned_landmarks - current_player_coins) / current_player_expected_gain_per_turn_sigmoid
        expected_turns_for_winning_money_other_player = max(1, other_player_cost_for_all_unowned_landmarks - other_player_coins) / other_player_expected_gain_per_turn_sigmoid


        if expected_turns_for_winning_money_other_player == expected_turns_for_winning_money_current_player:
            value = 0
        elif expected_turns_for_winning_money_current_player == np.inf and expected_turns_for_winning_money_other_player != np.inf:
            value = -1
        elif expected_turns_for_winning_money_other_player == np.inf and expected_turns_for_winning_money_current_player != np.inf:
            value = 1
        else:
            # taking log of the ratio to center ratio around 0.
            # (log(1/10) = -2.3 and log(10/1) = 2.3)
            x = np.clip(np.log(expected_turns_for_winning_money_other_player / expected_turns_for_winning_money_current_player), -100, 100)
            # taking tanh to squash log(ratio) between -1 and 1.
            value = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return torch.tensor(value)


    def predict(self, observation):
        if self.use_uniform_pvnet:
            policy_pred, value_pred = torch.ones((1, self.env.action_space.n)), torch.zeros((1, 1))
            policy_pred = policy_pred + 0.01*torch.randn((1, self.env.action_space.n))
        else:
            input = torch.tensor(observation).unsqueeze(0).to(torch.float32)
            policy_pred, value_pred = self.forward(input)
        # overwrite policy or value if specified.
        if self.use_custom_policy_edit:
            policy_pred = self.custom_policy_edit(observation, policy_pred)
        if self.use_custom_value:
            value_pred = self.custom_value(observation)

        policy_pred = torch.nn.functional.softmax(policy_pred, 1)
        return policy_pred.squeeze().detach().numpy(), value_pred.detach().numpy().item()
    
    def _loss(self, policy_preds, value_preds, policy_targets, value_targets):
        policy_loss = self.KLDiv(torch.nn.functional.log_softmax(policy_preds), torch.tensor(policy_targets).to(torch.float32))
        value_loss = torch.nn.functional.mse_loss(value_preds, torch.tensor(value_targets).to(torch.float32))
        return policy_loss + value_loss

    def train(
            self,
            buffer,
            batch_size,
            epochs,
            train_val_split,
            lr,
            weight_decay,
        ):
        buffer.compute_values()
        train_buffer, val_buffer = buffer.split_buffer_by_episode(train_val_split)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            train_batches = train_buffer.get_random_batches(batch_size = batch_size)

            tot_train_loss = 0
            train_steps_since_last_val_step = 0
            for i, batch in enumerate(train_batches):
                train_steps_since_last_val_step += 1
                obss, actions, rewards, next_obss, dones, player_ids, action_masks, values, value_preds, probs = batch
                
                # policy_preds, value_preds = self.forward(torch.tensor(obss).to(torch.float32))
                prob_preds, value_preds = self.forward(torch.tensor(obss).to(torch.float32))
                loss = self._loss(
                    policy_preds=prob_preds,
                    value_preds=value_preds,
                    policy_targets=probs,
                    value_targets=values
                )
                tot_train_loss += loss
                if i % 100 == 0:
                    obss, _, _, _, _, _, _, values, probs = val_buffer[:]
                    # policy_preds, value_preds = self.forward(torch.tensor(obss).to(torch.float32))
                    prob_preds, value_preds = self.forward(torch.tensor(obss).to(torch.float32))
                    avg_val_loss = self._loss(
                        policy_preds=prob_preds,
                        value_preds=value_preds,
                        policy_targets=probs,
                        value_targets=values
                    )
                    avg_train_loss = tot_train_loss/train_steps_since_last_val_step
                    print(f"epoch: {epoch} | train_loss: {avg_train_loss} | val_loss: {avg_val_loss}", end="\r")
                    tot_train_loss = 0
                    train_steps_since_last_val_step = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f"epoch: {epoch} | train_loss: {avg_train_loss} | val_loss: {avg_val_loss}")
        return train_buffer, val_buffer, avg_train_loss, avg_val_loss

class MCTSAgent:
    def __init__(
            self,
            env,
            num_mcts_sims,
            c_puct,
            pvnet,
            dirichlet_to_root_node = True,
            thinking_time: int = None,
            print_info: bool = False,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.mcts = MCTS(env, pvnet, num_mcts_sims, c_puct, dirichlet_to_root_node, thinking_time, print_info)
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

    # pred = self.predict(buffer[-100][0], flattened=True)
    breakpoint()
    pred = agent.compute_action(observation=observation)
    probs_preds, value_preds = self.forward(torch.tensor(buffer.obss[0:1000]).to(torch.float32))
