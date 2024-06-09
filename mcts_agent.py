import numpy as np
from torch import nn
import torch
import warnings

from mcts import MCTS
import itertools
from buffer import Buffer
import copy
import mlflow
import os

class MultiDimensionalOneHot:
    def __init__(self, values, device: str = None):
        self.device = device
        # np array where each row is padded with nans
        self.values = torch.tensor(list(itertools.zip_longest(*values, fillvalue=np.nan)), device=self.device, requires_grad=False).T

        # the one_hot_start_indices represent the indices in a flattened array. Each index is the
        # start index of a onehot dimension.
        values_lengths = torch.sum(~torch.isnan( self.values), axis=1)
        self.one_hot_start_indices = values_lengths.cumsum(0) - values_lengths
        
        # the lenght of the flattened onehot array
        self.one_hot_len = sum(map(len, values))
        # self.start_indices = (np.cumsum(n_elements) - n_elements)

    def to_onehot(self, array, batch_size=512):
        # one hot initialized by zeros
        one_hot = torch.zeros((len(array), self.one_hot_len), device=self.device, requires_grad=False)

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

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.best_epoch = None
        self.best_params = None
        self.wait = 0

    def update(self, loss, epoch, params=None):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_params = params
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_params = params
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            return True
        return False

    def reset(self):
        self.best_loss = None
        self.best_epoch = None
        self.best_params = None
        self.wait = 0

def add_layer(module, name, f_in, f_out, nonlinearity, device, layer_norm=True):
    layer = nn.Linear(f_in, f_out, device=device)
    if nonlinearity in ["softmax", "sigmoid", "tanh"]:
        nn.init.xavier_normal_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    module.add_module(name, layer)
    if layer_norm:
        linear_norm = nn.LayerNorm(f_out, device=device)
        module.add_module(f"{name}_norm", linear_norm)
    return module

class PolicyNet(nn.Module):
    def __init__(
            self,
            f_in,
            f_out,
            device: str = None,
    ):
        super().__init__()
        self.device = device
        self.logits = nn.Sequential()
        self.logits = add_layer(self.logits, "logits1", f_in, 128, "relu", self.device)
        self.logits.add_module("logits_relu1", nn.ReLU())
        self.logits = add_layer(self.logits, "logits2", 128, 128, "relu", self.device)
        self.logits.add_module("logits_relu2", nn.ReLU())
        self.logits = add_layer(self.logits, "logits3", 128, 128, "relu", self.device)
        self.logits.add_module("logits_relu3", nn.ReLU())
        self.logits = add_layer(self.logits, "logits4", 128, 128, "relu", self.device)
        self.logits.add_module("logits_relu4", nn.ReLU())
        self.logits = add_layer(self.logits, "logits5", 128, f_out, "softmax", self.device)
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.logits(x)
        
    
    def loss(self, policy_preds, policy_targets):
        policy_loss = self.KLDiv(torch.nn.functional.log_softmax(policy_preds), policy_targets)
        return policy_loss
    
class ValueNet(nn.Module):
    def __init__(
            self,
            f_in,
            f_out,
            device: str = None,
    ):
        super().__init__()
        self.device = device
        self.value = nn.Sequential()
        self.value = add_layer(self.value, "value1", f_in, 10, "relu", self.device)
        self.value.add_module("value_relu1", nn.ReLU())
        self.value = add_layer(self.value, "value2", 10, 128, "relu", self.device)
        self.value.add_module("value_relu2", nn.ReLU())
        self.value = add_layer(self.value, "value3", 128, 128, "relu", self.device)
        self.value.add_module("value_relu3", nn.ReLU())
        self.value = add_layer(self.value, "value4", 128, 10, "relu", self.device)
        self.value.add_module("value_relu4", nn.ReLU())
        self.value = add_layer(self.value, "value5", 10, f_out, "tanh", self.device, layer_norm=False)
        self.value.add_module("value_tanh5", nn.Tanh())

    def forward(self, x):
        return self.value(x)
    
    def loss(self, value_preds, value_targets):
        value_loss = torch.nn.functional.mse_loss(value_preds, value_targets)
        return value_loss

class PVNet:
    def __init__(
            self,
            env_cls,
            env_kwargs,
            uniform_pvnet: bool = False,
            custom_policy_edit: bool = False,
            custom_value: bool = False,
            device: str = None,
    ):
        env = env_cls(**env_kwargs)
        self.env = env
        self.use_uniform_pvnet = uniform_pvnet
        self.use_custom_policy_edit = custom_policy_edit
        self.use_custom_value = custom_value
        self.device = device
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
        one_hot_index_names = []
        identity_index_names = []
        for player in env.observation_indices["player_info"].keys():
            for card in env.observation_indices["player_info"][player]["cards"].keys():
                card_index = env.observation_indices["player_info"][player]["cards"][card]
                # card_values = env.observation_values["player_info"][player]["cards"][card]
                # one_hot_indices.append(card_index)
                # one_hot_values.append(card_values)
                identity_indices.append(card_index)
                identity_index_names.append(f"player_info/{player}/{card}")

            identity_indices.append(env.observation_indices["player_info"][player]["coins"])
            identity_index_names.append(f"player_info/{player}/coins")

        for alley in env.observation_indices["marketplace"].keys():
            for pos in env.observation_indices["marketplace"][alley].keys():
                card_index = env.observation_indices["marketplace"][alley][pos]["card"]
                card_values = env.observation_values["marketplace"][alley][pos]["card"]
                one_hot_indices.append(card_index)
                one_hot_values.append(card_values)
                one_hot_index_names.extend([f"marketplace/{alley}/{pos}/card/{env._env._card_num_to_name[num]}" for num in card_values])

                if alley != "landmarks":
                    count_index = env.observation_indices["marketplace"][alley][pos]["count"]
                    # count_values = env.observation_values["marketplace"][alley][pos]["count"]
                    # one_hot_indices.append(count_index)
                    # one_hot_values.append(count_values)
                    identity_indices.append(count_index)
                    identity_index_names.append(f"marketplace/{alley}/{pos}/count")

        one_hot_indices.append(env.observation_indices["current_player_index"])
        one_hot_values.append(env.observation_values["current_player_index"])
        one_hot_index_names.extend([f"current_player_index/{i}" for i in env.observation_values["current_player_index"]])

        one_hot_indices.append(env.observation_indices["current_stage_index"])
        one_hot_values.append(env.observation_values["current_stage_index"])
        one_hot_index_names.extend([f"current_stage_index/{i}" for i in env.observation_values["current_stage_index"]])

        one_hot_indices.append(env.observation_indices["another_turn"])
        one_hot_values.append(env.observation_values["another_turn"])
        one_hot_index_names.extend([f"another_turn/{i}" for i in env.observation_values["another_turn"]])

        one_hot_indices.append(env.observation_indices["build_rounds_left"])
        one_hot_values.append(env.observation_values["build_rounds_left"])
        one_hot_index_names.extend([f"build_rounds_left/{i}" for i in env.observation_values["build_rounds_left"]])

        self._one_hot_indices = one_hot_indices
        self._identity_indices = identity_indices
        self._input_names = one_hot_index_names + identity_index_names

        self._one_hot_indices_tensor = torch.tensor(np.array(one_hot_indices), device=self.device, requires_grad=False)
        self._identity_indices_tensor = torch.tensor(np.array(identity_indices), device=self.device, requires_grad=False)
        self._mdoh = MultiDimensionalOneHot(one_hot_values)

        num_inputs = self._mdoh.one_hot_len + len(self._identity_indices)
        num_outputs = self.env.action_space.n

        self.policy_net = PolicyNet(num_inputs, num_outputs, device=self.device)
        self.value_net = ValueNet(num_inputs, 1, device=self.device)
        
        self.is_trained = False
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")

    def add_layer(self, module, name, f_in, f_out, nonlinearity, device, layer_norm=True):
        layer = nn.Linear(f_in, f_out, device=device)
        if nonlinearity in ["softmax", "sigmoid", "tanh"]:
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        module.add_module(name, layer)
        if layer_norm:
            linear_norm = nn.LayerNorm(f_out, device=device)
            module.add_module(f"{name}_norm", linear_norm)
        return module

    def obss_to_onehot(self, obss):
        return torch.cat((self._mdoh.to_onehot(obss[:, self._one_hot_indices]), obss[:, self._identity_indices]), axis=1)

    def pred_policy(self, observation):
        logits = self.logits(observation)
        return logits
    
    def pred_value(self, observation):
        value = self.value(observation)
        return value

    def predict(self, observation):
        with torch.no_grad():
            if self.use_uniform_pvnet:
                policy_pred, value_pred = torch.ones((1, self.env.action_space.n)), torch.zeros((1, 1))
                policy_pred = policy_pred + 0.01*torch.randn((1, self.env.action_space.n))
            else:
                input = self.obss_to_onehot(torch.tensor(observation.astype(np.float32)).unsqueeze(0)).to(self.device)
                policy_pred, value_pred = self.policy_net(input), self.value_net(input)
                
            # overwrite policy or value if specified.
            if self.use_custom_policy_edit:
                policy_pred = self.custom_policy_edit(observation, policy_pred)
            if self.use_custom_value:
                value_pred = self.custom_value(observation)

            policy_pred = torch.nn.functional.softmax(policy_pred, 1)
            if policy_pred.device.type != "cpu":
                policy_pred = policy_pred.cpu()
            if value_pred.device.type != "cpu":
                value_pred = value_pred.cpu()
            return policy_pred.squeeze().detach().numpy(), value_pred.squeeze(0).detach().numpy()

    def scale_x(self, x):
        x[:, self._identity_indices_tensor] = (x[:, self._identity_indices_tensor] - self.x_mean) / self.x_std
        return x

    def train(
            self,
            batch_size,
            epochs,
            train_val_split,
            lr,
            weight_decay,
            buffer: Buffer | None = None,
            train_buffer: Buffer | None = None,
            val_buffer: Buffer | None = None,
        ):
        if buffer is not None:
            buffer.compute_values()
            train_buffer, val_buffer = buffer.split_buffer_by_episode(train_val_split)
        else:
            assert train_buffer is not None and val_buffer is not None
        self.x_mean = torch.tensor(train_buffer.obss[:, self._identity_indices].mean(axis=0).astype(np.float32), device=self.device, requires_grad=False)
        self.x_std = torch.tensor(train_buffer.obss[:, self._identity_indices].std(axis=0).astype(np.float32), device=self.device, requires_grad=False)


        avg_train_loss = None
        avg_val_loss = None
        epoch = 0

        obss_train, _, _, _, _, _, _, values_train, _, _, probs_train = train_buffer[:]
        obss_train = torch.tensor(obss_train.astype(np.float32))
        obss_train = self.obss_to_onehot(obss_train)
        values_train = torch.tensor(values_train.astype(np.float32))
        probs_train = torch.tensor(probs_train.astype(np.float32))
        obss_train, values_train, probs_train = obss_train.to(self.device), values_train.to(self.device), probs_train.to(self.device)

        obss_val, _, _, _, _, _, _, values_val, _, _, probs_val = val_buffer[:]
        obss_val = torch.tensor(obss_val.astype(np.float32))
        obss_val = self.obss_to_onehot(obss_val)
        values_val = torch.tensor(values_val.astype(np.float32))
        probs_val = torch.tensor(probs_val.astype(np.float32))
        obss_val, values_val, probs_val = obss_val.to(self.device), values_val.to(self.device), probs_val.to(self.device)


        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=weight_decay)

        policy_early_stopping = EarlyStopping(patience=5)
        value_early_stopping = EarlyStopping(patience=5)

        policy_done = False
        value_done = False

        for epoch in range(epochs):

            with torch.no_grad():
                if not policy_done:
                    prob_preds = self.policy_net(torch.tensor(obss_val))
                    avg_val_policy_loss = self.policy_net.loss(prob_preds, torch.tensor(probs_val))

                    if policy_early_stopping.update(avg_val_policy_loss, epoch, self.policy_net.state_dict()):
                        print("policy net is done!")
                        policy_done = True

                if not value_done:
                    value_preds = self.value_net(torch.tensor(obss_val))
                    avg_val_value_loss = self.value_net.loss(value_preds, torch.tensor(values_val))

                    if value_early_stopping.update(avg_val_value_loss, epoch, self.value_net.state_dict()):
                        print("value net is done!")
                        value_done = True
            
            if policy_done and value_done:
                break

            indices_for_all_batches = train_buffer.get_random_batch_indices(batch_size)

            tot_train_policy_loss = 0
            tot_train_value_loss = 0
            train_steps_since_last_val_step = 0
            n_batches = len(indices_for_all_batches)
            print(n_batches)
            for i, batch_indices in enumerate(indices_for_all_batches):
                train_steps_since_last_val_step += 1

                if not policy_done:
                    prob_preds = self.policy_net(torch.tensor(obss_train[batch_indices]))
                    policy_loss = self.policy_net.loss(prob_preds, torch.tensor(probs_train[batch_indices]))

                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    tot_train_policy_loss += policy_loss.detach()
                    avg_train_policy_loss = tot_train_policy_loss/train_steps_since_last_val_step

                if not value_done:
                    value_preds = self.value_net(torch.tensor(obss_train[batch_indices]))
                    value_loss = self.value_net.loss(value_preds, torch.tensor(values_train[batch_indices]))

                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

                    tot_train_value_loss += value_loss.detach()
                    avg_train_value_loss = tot_train_value_loss/train_steps_since_last_val_step

                avg_train_loss = avg_train_policy_loss + avg_train_value_loss

                if i % 100 == 0:
                    with torch.no_grad():
                        if not policy_done:
                            prob_preds = self.policy_net(torch.tensor(obss_val))
                            avg_val_policy_loss = self.policy_net.loss(prob_preds, torch.tensor(probs_val))

                        if not value_done:
                            value_preds = self.value_net(torch.tensor(obss_val))
                            avg_val_value_loss = self.value_net.loss(value_preds, torch.tensor(values_val))

                        avg_val_loss = avg_val_policy_loss + avg_val_value_loss

                        print(f"epoch: {epoch} {round(i/n_batches * 100)}% | train_loss | {avg_train_loss} | val_loss: {avg_val_loss} | train_policy_loss: {avg_train_policy_loss} | train_value_loss: {avg_train_value_loss} | val_policy_loss: {avg_val_policy_loss} | val_value_loss: {avg_val_value_loss} | policy_done: {policy_done} | value_done: {value_done}", end="\r")
                        mlflow.log_metric("epoch", epoch, step=i)
                        mlflow.log_metric("train_policy_loss", avg_train_policy_loss, step=i)
                        mlflow.log_metric("train_value_loss", avg_train_value_loss, step=i)
                        mlflow.log_metric("val_policy_loss", avg_val_policy_loss, step=i)
                        mlflow.log_metric("val_value_loss", avg_val_value_loss, step=i)
                        mlflow.log_metric("train_loss", avg_train_loss, step=i)
                        mlflow.log_metric("val_loss", avg_val_loss, step=i)
                        tot_train_policy_loss = 0
                        tot_train_value_loss = 0
                        train_steps_since_last_val_step = 0
            
            if policy_done and value_done:
                break

        self.policy_net.load_state_dict(policy_early_stopping.best_params)
        self.value_net.load_state_dict(value_early_stopping.best_params)

        print(f"epoch: {epoch} | train_loss | {avg_train_loss} | val_loss: {avg_val_loss} | train_policy_loss: {avg_train_policy_loss} | train_value_loss: {avg_train_value_loss} | val_policy_loss: {avg_val_policy_loss} | val_value_loss: {avg_val_value_loss} | policy_done: {policy_done} | value_done: {value_done}")
        return train_buffer, val_buffer, avg_train_loss, avg_val_loss

    def save(self, folder):
        os.mkdir(folder)
        torch.save(self.policy_net.state_dict(), folder+"/policy_net.ckpt")
        torch.save(self.value_net.state_dict(), folder+"/value_net.ckpt")

    def load(self, folder):
        self.policy_net.load_state_dict(torch.load(folder+"/policy_net.ckpt"))
        self.value_net.load_state_dict(torch.load(folder+"/value_net.ckpt"))

    def load_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict["policy_weights"])
        self.value_net.load_state_dict(state_dict["value_weights"])

    def state_dict(self):
        return {
            "policy_weights": self.policy_net.state_dict(),
            "value_weights": self.value_net.state_dict(),
        }

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
