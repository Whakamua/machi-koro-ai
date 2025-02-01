import h5py
import numpy as np
from torch import nn
import torch
import warnings
import re
import time
import random
import pickle
import sys

from mcts import MCTS
import itertools
import copy
import mlflow
import os
from env_machi_koro_2 import GymMachiKoro2
import logging

logger = logging.getLogger("SelfPlayFileLogger")

class TrainLogger:
    def __init__(self):
        self.first_call = True
        self.file_logger = logging.getLogger("SelfPlayFileLogger")

    def log(
        self,
        epoch: int,
        steps: int,
        avg_train_loss: float,
        avg_val_loss: float,
        avg_train_policy_loss: float,
        avg_train_value_loss: float,
        avg_val_policy_loss: float,
        avg_val_value_loss: float,
        policy_done: bool,
        value_done: bool,
        final: bool = False,
    ):
        # Cursor handling: Clear previous output if not the first call
        if not self.first_call:
            self.clear_stdout()
        else:
            self.first_call = False

        if not final:
            # Log metrics to mlflow
            mlflow.log_metric("epoch", epoch, step=steps)
            mlflow.log_metric("train_policy_loss", avg_train_policy_loss, step=steps)
            mlflow.log_metric("train_value_loss", avg_train_value_loss, step=steps)
            mlflow.log_metric("val_policy_loss", avg_val_policy_loss, step=steps)
            mlflow.log_metric("val_value_loss", avg_val_value_loss, step=steps)
            mlflow.log_metric("train_loss", avg_train_loss, step=steps)
            mlflow.log_metric("val_loss", avg_val_loss, step=steps)

        # Generate the log summary
        summary = f"""
############################################
Training PV-net:
############################################
Epoch: {epoch}
Policy:
  Train Loss: {avg_train_policy_loss:.4f}
  Val Loss: {avg_val_policy_loss:.4f}
  Completion: {'Done' if policy_done else 'In Progress'}
Value:
  Train Loss: {avg_train_value_loss:.4f}
  Val Loss: {avg_val_value_loss:.4f}
  Completion: {'Done' if value_done else 'In Progress'}
Total:
  Avg Val Loss: {avg_val_loss:.4f}
  Avg Train Loss: {avg_train_loss:.4f}
############################################
"""

        # Print updated training information to stdout
        sys.stdout.write(summary + "\n")
        sys.stdout.flush()

        if final:
            self.clear_stdout()
            sys.stdout.flush()
            self.file_logger.info(summary + "\n")

    def clear_stdout(self):
        for _ in range(18):
            sys.stdout.write("\033[F\033[K")

class NotEnoughDataError(Exception):
    pass

class HDF5DataLoader:
    def __init__(self, file_path, subset_rules, chunk_size, num_workers=1):
        """
        A class to handle HDF5 datasets and generate train and validation DataLoaders.

        Args:
            file_path (str): Path to the HDF5 file.
            subset_rules (dict): Rules for selecting rows from iterations/buffers.
                                 Example: {1: 0.5, 2: 0.5, ..., 6: 1.0, 7: 1.0}
            chunk_size (int): Chunk size for DataLoaders.
            num_workers (int): Number of worker threads for DataLoader.
        """
        self.file_path = file_path
        self.subset_rules = subset_rules
        self.chunk_size = chunk_size
        self.num_workers = num_workers

        # Prepare row indices
        self.train_indices, self.val_indices, self.column_indices = self._prepare_indices()
        if len(self.train_indices) == 0 or len(self.val_indices) == 0:
            raise NotEnoughDataError("Not enough data to create train and validation sets.")

        # Create DataLoaders
        self.train_loader = self._create_dataloader(self.train_indices, self.column_indices, "train")
        self.val_loader = self._create_dataloader(self.val_indices, self.column_indices, "val")

    def get_column_indices_regex(self, column_names, regex):
        r_filter = re.compile(regex)
        vmatch = np.vectorize(lambda x:bool(r_filter.match(x)))
        return np.where(vmatch(column_names))[0]
    
    def _prepare_indices_for_split(self, split):
        with h5py.File(self.file_path, "r") as h5f:
            vds_sources = []
            for iteration, fraction in self.subset_rules.items():
                if fraction == 0.0:
                    continue
                grp = h5f[split][iteration]
                games = list(grp.keys())
                vds_sources.extend([(iteration, game) for game in games])
        return vds_sources


    def _prepare_indices(self):
        """Generate row indices based on subset rules."""

        train_row_indices = self._prepare_indices_for_split("train")
        val_row_indices = self._prepare_indices_for_split("val")

        with h5py.File(self.file_path, "r") as h5f:
            column_indices = {
                "obs": self.get_column_indices_regex(h5f.attrs["columns"], r"obs\d+"),
                "values": self.get_column_indices_regex(h5f.attrs["columns"], r"\bvalue\b"),
                "probs": self.get_column_indices_regex(h5f.attrs["columns"], r"prob\d+"),
            }

        return train_row_indices, val_row_indices, column_indices

    def _create_dataloader(self, indices, column_indices, split):
        """Create a DataLoader from the given indices."""
        return HDF5Dataset(self.file_path, indices, column_indices, self.chunk_size, split)

    def get_dataloaders(self):
        """Return the train and validation DataLoaders."""
        return self.train_loader, self.val_loader



class HDF5Dataset:
    def __init__(self, file_path, buffer_indices, column_indices: dict[str, np.ndarray[int]], chunk_size, split):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            buffer_indices (list): List of tuples (iteration, buffer).
        """
        self.file_path = file_path
        self.buffer_indices = buffer_indices
        self.column_indices = column_indices
        self.sorted_column_indices_permutation = np.argsort(np.concatenate(list(self.column_indices.values())))
        self.sorted_column_indices = np.concatenate(list(self.column_indices.values()))[self.sorted_column_indices_permutation]
        self.vds_indices = {}
        col_counter = 0

        for name, cols in column_indices.items():
            self.vds_indices[name] = self.sorted_column_indices_permutation[col_counter: col_counter+len(cols)]
            col_counter += len(cols)

        self.n_columns = sum(len(indices) for indices in self.column_indices.values())
        self.chunk_size = chunk_size
        self.split = split

        self.shuffle()


    def shuffle(self):
        self.vds_names = []
        self.tot_rows = 0
        def create_vds(h5f: h5py.File, buffers_for_chunk, total_rows):
            vds_name = f"VDS_{self.split}_{len(self.vds_names)}"
            self.vds_names.append(vds_name)
            vds_layout = h5py.VirtualLayout(shape=(total_rows, self.n_columns), dtype='float64')
            row_number = 0
            for (iteration, buffer_name) in buffers_for_chunk:
                virtual_source = h5py.VirtualSource(h5f[self.split][iteration][buffer_name])
                # filling only the columns that are in the column_indices
                vds_layout[row_number:row_number+virtual_source.shape[0], :] = virtual_source[:, self.sorted_column_indices]
                row_number += virtual_source.shape[0]

            h5f.create_virtual_dataset(vds_name, vds_layout)

        random.shuffle(self.buffer_indices)
        with h5py.File(self.file_path, "a") as h5f:
            for key in h5f.keys():
                if key.startswith(f"VDS_{self.split}_"):
                    del h5f[key]
            n_rows = 0
            buffers_for_chunk = []
            for buffer_index in self.buffer_indices:
                iteration, buffer_name = buffer_index
                rows_in_bufffer = h5f[self.split][iteration][buffer_name].shape[0]
                self.tot_rows += rows_in_bufffer
                n_rows += rows_in_bufffer
                buffers_for_chunk.append(buffer_index)
                if n_rows >= self.chunk_size:
                    create_vds(h5f, buffers_for_chunk, n_rows)
                    n_rows = 0
                    buffers_for_chunk = []
            if n_rows > 0:
                create_vds(h5f, buffers_for_chunk, n_rows)


    def __len__(self):
        """Return the total number of chunks in the dataset."""
        return len(self.vds_names)

    def __getitem__(self, idx):
        """Load a single row by index."""

        vds_name = self.vds_names[idx]
        with h5py.File(self.file_path, "r") as h5f:
            # Note to self, it is not needed to filter out rows where done is 1, since the data
            # is stored as obs, 
            data = h5f[vds_name][:]

        return {
            data_name: torch.tensor(data[:, indices], dtype=torch.float32)
            for data_name, indices in self.vds_indices.items()
        }
    
    def __iter__(self):
        """Initialize the iterator."""
        self.current_index = 0
        return self

    def __next__(self):
        """Fetch the next chunk of data during iteration."""
        if self.current_index >= len(self):
            raise StopIteration
        result = self[self.current_index]
        self.current_index += 1
        return result

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
        policy_loss = self.KLDiv(torch.nn.functional.log_softmax(policy_preds, dim=1), policy_targets)
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
            env: GymMachiKoro2,
            uniform_pvnet: bool = False,
            custom_policy_edit: bool = False,
            custom_value: bool = False,
            device: str = None,
            mlflow_experiment_name: str = None,
    ):
        self.mlflow_experiment_name = mlflow_experiment_name
        if self.mlflow_experiment_name is not None:
            mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
            mlflow.set_experiment(mlflow_experiment_name)

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

        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")

        self.reset_weights()

    def reset_weights(self):
        num_inputs = self._mdoh.one_hot_len + len(self._identity_indices)
        num_outputs = self.env.action_space.n
        self.policy_net = PolicyNet(num_inputs, num_outputs, device=self.device)
        self.value_net = ValueNet(num_inputs, 1, device=self.device)
        self.is_trained = False

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
    
    def _process_chunk(self, chunk: dict[str, torch.Tensor]):
        obss = chunk["obs"].float()
        obss = self.obss_to_onehot(obss)
        values = chunk["values"].float()
        probs = chunk["probs"].float()
        obss, values, probs = obss.to(self.device), values.to(self.device), probs.to(self.device)
        return obss, values, probs

    def train_hdf5(
        self,
        batch_size: int,
        epochs: int,
        train_val_split: float,
        lr: float,
        weight_decay: float,
        hdf5_file_path: str,
        subset_rules: dict,
        reset_weights: bool = False,
    ):
        try:
            # Initialize the data loader manager
            data_manager = HDF5DataLoader(
                file_path=hdf5_file_path,
                subset_rules=subset_rules,
                chunk_size=int(64e4),
                num_workers=1,
            )
        except NotEnoughDataError:
            warnings.warn("Not enough data to train the model.")
            return None
        
        train_loader, val_loader = data_manager.get_dataloaders()

        if reset_weights:
            self.reset_weights()
        
        avg_train_loss = None
        avg_val_loss = None

        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=weight_decay)

        policy_early_stopping = EarlyStopping(patience=5)
        value_early_stopping = EarlyStopping(patience=5)

        policy_done = False
        value_done = False
        with mlflow.start_run() as run:
            train_logger = TrainLogger()
            for epoch in range(epochs):
                with torch.no_grad():
                    if not policy_done:
                        sum_of_avg_val_policy_loss = 0
                        time_val = time.time()
                        for val_chunk in val_loader:
                            obss_val, values_val, probs_val = self._process_chunk(val_chunk)
                            prob_preds = self.policy_net(obss_val)
                            sum_of_avg_val_policy_loss += self.policy_net.loss(prob_preds, probs_val)
                        
                        avg_val_policy_loss = sum_of_avg_val_policy_loss / len(val_loader)

                        if policy_early_stopping.update(avg_val_policy_loss, epoch, self.policy_net.state_dict()):
                            policy_done = True

                    if not value_done:
                        sum_of_avg_val_value_loss = 0
                        for val_chunk in val_loader:
                            obss_val, values_val, probs_val = self._process_chunk(val_chunk)
                            value_preds = self.value_net(obss_val)
                            sum_of_avg_val_value_loss += self.value_net.loss(value_preds, values_val)
                        avg_val_value_loss = sum_of_avg_val_value_loss / len(val_loader)
                        if value_early_stopping.update(avg_val_value_loss, epoch, self.value_net.state_dict()):
                            value_done = True
                
                if policy_done and value_done:
                    break

                train_loader.shuffle()
                tot_train_policy_loss = 0
                tot_train_value_loss = 0
                train_steps_since_last_val_step = 0
                n_rows_in_train_loader = train_loader.tot_rows
                n_rows_trained_on_in_epoch = 0
                steps = 0
                for train_chunk in train_loader:
                    obss_train, values_train, probs_train = self._process_chunk(train_chunk)

                    def split_array(arr, chunk_size):
                        return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

                    indices_for_all_batches = split_array(np.arange(obss_train.shape[0]), batch_size)
                    
                    for i, batch_indices in enumerate(indices_for_all_batches):
                        steps+=1
                        n_rows_trained_on_in_epoch += len(batch_indices)
                        train_steps_since_last_val_step += 1

                        if not policy_done:
                            prob_preds = self.policy_net(obss_train[batch_indices])
                            policy_loss = self.policy_net.loss(prob_preds, probs_train[batch_indices])

                            policy_optimizer.zero_grad()
                            policy_loss.backward()
                            policy_optimizer.step()

                            tot_train_policy_loss += policy_loss.detach()
                            avg_train_policy_loss = tot_train_policy_loss/train_steps_since_last_val_step
                        
                        if not value_done:
                            value_preds = self.value_net(obss_train[batch_indices])
                            value_loss = self.value_net.loss(value_preds, values_train[batch_indices])

                            value_optimizer.zero_grad()
                            value_loss.backward()
                            value_optimizer.step()

                            tot_train_value_loss += value_loss.detach()
                            avg_train_value_loss = tot_train_value_loss/train_steps_since_last_val_step
                        
                        avg_train_loss = avg_train_policy_loss + avg_train_value_loss

                        if i % 100 == 0:
                            with torch.no_grad():
                                if not policy_done:
                                    sum_of_avg_val_policy_loss = 0
                                    for val_chunk in val_loader:
                                        obss_val, values_val, probs_val = self._process_chunk(val_chunk)
                                        prob_preds = self.policy_net(obss_val)
                                        sum_of_avg_val_policy_loss += self.policy_net.loss(prob_preds, probs_val)
                                        avg_val_policy_loss = self.policy_net.loss(prob_preds, probs_val)
                                        
                                    avg_val_policy_loss = sum_of_avg_val_policy_loss / len(val_loader)

                                if not value_done:
                                    sum_of_avg_val_value_loss = 0
                                    for val_chunk in val_loader:
                                        obss_val, values_val, probs_val = self._process_chunk(val_chunk)
                                        value_preds = self.value_net(obss_val)
                                        sum_of_avg_val_value_loss += self.value_net.loss(value_preds, values_val)
                                    avg_val_value_loss = sum_of_avg_val_value_loss / len(val_loader)
                                
                                avg_val_loss = avg_val_policy_loss + avg_val_value_loss

                                train_logger.log(
                                    epoch=epoch,
                                    steps=steps,
                                    avg_train_loss=avg_train_loss,
                                    avg_val_loss=avg_val_loss,
                                    avg_train_policy_loss=avg_train_policy_loss,
                                    avg_train_value_loss=avg_train_value_loss,
                                    avg_val_policy_loss=avg_val_policy_loss,
                                    avg_val_value_loss=avg_val_value_loss,
                                    policy_done=policy_done,
                                    value_done=value_done,
                                )
                                tot_train_policy_loss = 0
                                tot_train_value_loss = 0
                                train_steps_since_last_val_step = 0

        # Finalize logging at the end
        train_logger.log(
        epoch=epoch,
            steps=steps,
            avg_train_loss=avg_train_loss,
            avg_val_loss=avg_val_loss,
            avg_train_policy_loss=avg_train_policy_loss,
            avg_train_value_loss=avg_train_value_loss,
            avg_val_policy_loss=avg_val_policy_loss,
            avg_val_value_loss=avg_val_value_loss,
            policy_done=policy_done,
            value_done=value_done,
            final=True
        )

        self.policy_net.load_state_dict(policy_early_stopping.best_params)
        self.value_net.load_state_dict(value_early_stopping.best_params)

        logger.info("training done")
        return avg_train_loss, avg_val_loss

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
            env: GymMachiKoro2,
            num_mcts_sims: int,
            c_puct: float,
            pvnet: PVNet,
            dirichlet_to_root_node = True,
            thinking_time: int = None,
            print_info: bool = False,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.mcts = MCTS(env, pvnet, num_mcts_sims, c_puct, dirichlet_to_root_node, thinking_time, print_info)
        warnings.warn("Not using any temperature in probs, might need that for first n actions")

    def get_state_dict(self):
        return {
            "uniform_pvnet": self.mcts.pvnet.use_uniform_pvnet,
            "custom_policy_edit": self.mcts.pvnet.use_custom_policy_edit,
            "custom_value": self.mcts.pvnet.use_custom_value,
            "weights": self.mcts.pvnet.state_dict()
        }

    def set_state_dict(self, state_dict):
        self.mcts.pvnet.use_uniform_pvnet = state_dict["uniform_pvnet"]
        self.mcts.pvnet.use_custom_policy_edit = state_dict["custom_policy_edit"]
        self.mcts.pvnet.use_custom_value = state_dict["custom_value"]
        self.mcts.pvnet.load_state_dict(state_dict["weights"])

    def reset(self, env_state):
        self.mcts.reset(env_state=env_state)
    
    def compute_action(self, observation):
        probs = self.mcts.compute_probs(observation)
        action = np.argmax(probs)
        return action, probs
    
    def train(self, buffer, batch_size):
        return self.mcts.pvnet.train(buffer, batch_size)
    
    def pickle(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_pickle(path):
        with open(path, "rb") as file:
            return pickle.load(file)
