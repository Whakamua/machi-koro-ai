import numpy as np
from gym.spaces import flatten, flatten_space
import random

class Buffer:
    def __init__(self, gamma, observation_space, action_space, capacity):
        self._gamma = gamma
        self._observation_space = observation_space
        self._action_space = action_space
        self._capacity = capacity
        self._size = 0
        self.reset()

    @property
    def isfull(self):
        return self._size == self._capacity
    
    def reset(self):
        self._obss = np.zeros((self._capacity, flatten_space(self._observation_space).shape[0]))
        self._actions = np.zeros((self._capacity, flatten_space(self._action_space).shape[0]))
        self._rewards = np.zeros((self._capacity, 1))
        self._next_obss = np.zeros((self._capacity, flatten_space(self._observation_space).shape[0]))
        self._dones = np.zeros((self._capacity, 1))
        self._player_ids = np.zeros((self._capacity, 1))
        self._action_masks = np.zeros((self._capacity, flatten_space(self._action_space).shape[0]))
        self._values = np.zeros((self._capacity, 1))
    
    def _add_to_idx(self, idx, obs, action, reward, next_obs, done):
        self._obss[idx] = flatten(self._observation_space, obs)
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obss[idx] = flatten(self._observation_space, next_obs)
        self._dones[idx] = done
        self._player_ids[idx] = obs["current_player_index"]
        self._action_masks[idx] = obs["action_mask"]

    def add(self, obs, action, reward, next_obs, done):
        assert self._size < self._capacity
        self._add_to_idx(self._size, obs, action, reward, next_obs, done)
        self._size += 1
        # else:
        #     index = random.randint(0, self._capacity - 1)
        #     self._add_to_idx(index, obs, action, reward, next_obs, done)

    def compute_values(self):
        winning_player = None
        indices_to_keep = []
        self._values = np.zeros(len(self._obss))
        for i in reversed(range(len(self._obss))):
            if self._dones[i]:
                last_terminal_idx = i
                break

        for i in range(last_terminal_idx, -1, -1):
            if self._dones[i]:
                winning_player = self._player_ids[i]
            else:
                self._values[i] = 1 if self._player_ids[i] == winning_player else -1
                indices_to_keep.append(i)
                # self._values[i] = self._rewards[i] + self._gamma*self._values[i+1]
        self._obss = self._obss[indices_to_keep]
        self._actions = self._actions[indices_to_keep]
        self._rewards = self._rewards[indices_to_keep]
        self._next_obss = self._next_obss[indices_to_keep]
        self._dones = self._dones[indices_to_keep]
        self._player_ids = self._player_ids[indices_to_keep]
        self._action_masks = self._action_masks[indices_to_keep]
        self._values = self._values[indices_to_keep]

        self._size = len(self._obss)

    def sample(self, batch_size):
        batch_indices = np.random.randint(low=0, high=self._size, size=batch_size, dtype=int)
        return (
            self._obss[batch_indices],
            self._actions[batch_indices],
            self._rewards[batch_indices],
            self._next_obss[batch_indices],
            self._dones[batch_indices],
            self._player_ids[batch_indices],
            self._action_masks[batch_indices],
            self._values[batch_indices],
        )