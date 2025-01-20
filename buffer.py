import numpy as np
from gym.spaces import flatten, flatten_space, Discrete, MultiBinary
import polars as pl

class NoWinnerException(Exception):
    pass

class Buffer:
    def __init__(self, observation_space, action_space, capacity: int = 0):
        assert isinstance(action_space, Discrete), "other action spaces are not supported due to the reset method assuming a dimension of 1" 
        assert isinstance(observation_space, MultiBinary)
        self._observation_space = observation_space
        self._action_space = action_space
        self._capacity = capacity
        self._size = 0

        self.reset(self._capacity)

        self._obs_col_names = ["obs" + str(i) for i in range(self._obss.shape[1])]
        self._action_col_names = ["action"]
        self._reward_col_names = ["reward"]
        self._next_obs_col_names = ["next_obs" + str(i) for i in range(self._next_obss.shape[1])]
        self._done_col_names = ["done"]
        self._player_id_col_names = ["player_id"]
        self._action_mask_col_names = ["action_mask" + str(i) for i in range(self._action_masks.shape[1])]
        self._value_col_names = ["value"]
        self._value_pred_col_names = ["value_pred"]
        self._value_mcts_col_names = ["value_mcts"]
        self._prob_col_names = ["prob" + str(i) for i in range(self._probs.shape[1])]
        self._winner_col_names = ["winner"]

        self._flattened_column_names_and_types_dict = {
            obs_col_name: pl.Int16 for obs_col_name in self._obs_col_names
        } | {
            action_col_name: pl.UInt8 for action_col_name in self._action_col_names
        } | {
            reward_col_name: pl.Int8 for reward_col_name in self._reward_col_names
        } | {
            next_obs_col_name: pl.Int16 for next_obs_col_name in self._next_obs_col_names
        } | {
            done_col_name: pl.Boolean for done_col_name in self._done_col_names
        } | {
            player_id_col_name: pl.UInt8 for player_id_col_name in self._player_id_col_names
        } | {
            action_mask_col_name: pl.Boolean for action_mask_col_name in self._action_mask_col_names
        } | {
            value_col_name: pl.Float32 for value_col_name in self._value_col_names
        } | {
            value_pred_col_name: pl.Float32 for value_pred_col_name in self._value_pred_col_names
        } | {
            value_mcts_col_name: pl.Float32 for value_mcts_col_name in self._value_mcts_col_names
        } | {
            prob_col_name: pl.Float32 for prob_col_name in self._prob_col_names
        } | {
            winner_col_name: pl.UInt8 for winner_col_name in self._winner_col_names
        }

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def isfull(self):
        return self._size == self._capacity
    
    @property
    def size(self):
        return self._size
    
    @property
    def effective_size(self):
        done_indices = np.arange(self.size)[:,None][self.dones.astype(bool)]
        if len(done_indices) == 0:
            return 0
        return done_indices.max() + 1

    @property
    def obss(self):
        return self._obss[:self.size]
    
    @property
    def actions(self):
        return self._actions[:self.size]
    
    @property
    def rewards(self):
        return self._rewards[:self.size]
    
    @property
    def next_obss(self):
        return self._next_obss[:self.size]
    
    @property
    def dones(self):
        return self._dones[:self.size]
    
    @property
    def player_ids(self):
        return self._player_ids[:self.size]
    
    @property
    def action_masks(self):
        return self._action_masks[:self.size]
    
    @property
    def values(self):
        return self._values[:self.size]
    
    @property
    def value_preds(self):
        return self._value_preds[:self.size]
    
    @property
    def values_mcts(self):
        return self._values_mcts[:self.size]
    
    @property
    def probs(self):
        return self._probs[:self.size]

    def reset(self, capacity):
        self._obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._actions = np.zeros((capacity, 1))
        self._rewards = np.zeros((capacity, 1))
        self._next_obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._dones = np.zeros((capacity, 1))
        self._player_ids = np.zeros((capacity, 1))
        self._action_masks = np.zeros((capacity, self._action_space.n))
        self._values = np.zeros((capacity, 1))
        self._value_preds = np.zeros((capacity, 1))
        self._values_mcts = np.zeros((capacity, 1))
        self._probs = np.zeros((capacity, self._action_space.n))

        self.values_computed = False
        
        
    @property
    def flattened_column_names_and_types_dict(self):
        return self._flattened_column_names_and_types_dict

    def export_flattened(self):
        if not self.values_computed:
            self.compute_values()

        return np.concatenate([
            self._obss[:self._size],
            self._actions[:self._size],
            self._rewards[:self._size],
            self._next_obss[:self._size],
            self._dones[:self._size],
            self._player_ids[:self._size],
            self._action_masks[:self._size],
            self._values[:self._size],
            self._value_preds[:self._size],
            self._values_mcts[:self._size],
            self._probs[:self._size],
        ], axis=1)

    def import_flattened(self, flattened):
        self._obss = flattened[self._obs_col_names].to_numpy()
        self._actions = flattened[self._action_col_names].to_numpy()
        self._rewards = flattened[self._reward_col_names].to_numpy()
        self._next_obss = flattened[self._next_obs_col_names].to_numpy()
        self._dones = flattened[self._done_col_names].to_numpy()
        self._player_ids = flattened[self._player_id_col_names].to_numpy()
        self._action_masks = flattened[self._action_mask_col_names].to_numpy()
        self._values = flattened[self._value_col_names].to_numpy()
        self._value_preds = flattened[self._value_pred_col_names].to_numpy()
        self._values_mcts = flattened[self._value_mcts_col_names].to_numpy()
        self._probs = flattened[self._prob_col_names].to_numpy()
        self._size = len(flattened)
        self._capacity = self._size
        self.values_computed = True

    def add(self, obs, action, reward, next_obs, done, probs, current_player_index, action_mask, value_pred, value_mcts):
        assert self._size < self._capacity
        index = self._size
        self[index] = (
            flatten(self._observation_space, obs),
            action,
            reward,
            flatten(self._observation_space, next_obs),
            done,
            current_player_index,
            action_mask,
            0,
            value_pred,
            value_mcts,
            probs,
        )
        self.values_computed = False
        
        self._size += 1

    def compute_values(self):
        if self._dones[self._size-1] != 1:
            raise NoWinnerException("No winner has been determined yet")

        winner = self._player_ids[self._size-1]
        for i in range(self._size-1, -1, -1):
            self._values[i] = 1 if self._player_ids[i] == winner else -1
        self.values_computed = True

    def __getitem__(self, indices):
        return (
            self._obss[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obss[indices],
            self._dones[indices],
            self._player_ids[indices],
            self._action_masks[indices],
            self._values[indices],
            self._value_preds[indices],
            self._values_mcts[indices],
            self._probs[indices],
        )

    def __setitem__(self, indices, items):
        self._obss[indices] = items[0]
        self._actions[indices] = items[1]
        self._rewards[indices] = items[2]
        self._next_obss[indices] = items[3]
        self._dones[indices] = items[4]
        self._player_ids[indices] = items[5]
        self._action_masks[indices] = items[6]
        self._values[indices] = items[7]
        self._value_preds[indices] = items[8]
        self._values_mcts[indices] = items[9]
        self._probs[indices] = items[10]

    def sample(self, batch_size):
        batch_indices = np.random.randint(low=0, high=self._size, size=batch_size, dtype=int)
        return self[batch_indices]