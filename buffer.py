import numpy as np
from gym.spaces import flatten, flatten_space, Discrete, MultiBinary, Dict
import random
import copy

class Buffer:
    def __init__(self, observation_space, action_space, capacity: int | None = None):
        assert isinstance(action_space, Discrete), "other action spaces are not supported due to the reset method assuming a dimension of 1" 
        assert isinstance(observation_space["action_mask"], MultiBinary)
        assert isinstance(observation_space, Dict)
        self._observation_space = observation_space
        self._action_space = action_space
        self._capacity = capacity
        self._size = 0
        if self._capacity is not None:
            self.reset(self._capacity)

    def get_big_buffer(self):
        return BigBuffer(self._observation_space, self._action_space)

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
    def probs(self):
        return self._probs[:self.size]

    def reset(self, capacity):
        self._obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._actions = np.zeros((capacity, 1))
        self._rewards = np.zeros((capacity, 1))
        self._next_obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._dones = np.zeros((capacity, 1))
        self._player_ids = np.zeros((capacity, 1))
        self._action_masks = np.zeros((capacity, self._observation_space["action_mask"].n))
        self._values = np.zeros((capacity, 1))
        self._probs = np.zeros((capacity, self._action_space.n))
        self._episode_starts = {}
        self._episode_ends = {}
        self._episode_lengths = {}
        self._episode_number = 0
        self._new_episode = True

    def add(self, obs, action, reward, next_obs, done, probs):
        assert self._size < self._capacity
        index = self._size
        self[index] = (
            flatten(self._observation_space, obs),
            action,
            reward,
            flatten(self._observation_space, next_obs),
            done,
            obs["current_player_index"],
            obs["action_mask"],
            0,
            probs
        )
        if self._new_episode:
            self._episode_starts[self._episode_number] = self.size
            self._new_episode = False
            self._episode_ends[self._episode_number] = None
            self._episode_lengths[self._episode_number] = 0

        self._episode_ends[self._episode_number] = self.size
        self._episode_lengths[self._episode_number] += 1
        
        self._size += 1

        if done:
            self._episode_number += 1
            self._new_episode = True

    def compute_values(self):
        # winning_player = None
        # self._size = len(self._obss)

        # self._values = np.zeros((self._size,1))
        # for i in reversed(range(self._size)):
        #     if self._dones[i]:
        #         last_terminal_idx = i
        #         break

        # for i in range(last_terminal_idx, -1, -1):
        #     if self._dones[i]:
        #         winning_player = self._player_ids[i]
        #     else:
        #         self._values[i] = 1 if self._player_ids[i] == winning_player else -1
        #         # self._values[i] = self._rewards[i] + self._gamma*self._values[i+1]
        for episode, length in self._episode_lengths.items():
            start_index = self._episode_starts[episode]
            end_index = self._episode_ends[episode]
            if self._dones[end_index]:
                winner = self._player_ids[end_index]
                for i in range(end_index, start_index - 1, -1):
                    if i == end_index:
                        self._values[i] = 0
                    else:
                        self._values[i] = 1 if self._player_ids[i] == winner else -1

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
        self._probs[indices] = items[8]

    def sample(self, batch_size, exclude_terminal_states=False):
        if exclude_terminal_states:
            batch_indices = np.random.choice(np.arange(self.size)[~self.dones[:,0].astype(bool)], size=batch_size, replace=False)
        else:
            batch_indices = np.random.randint(low=0, high=self._size, size=batch_size, dtype=int)
        return self[batch_indices]

    def get_random_batches(self, batch_size, exclude_terminal_states=False):
        assert batch_size <= self._size
        n_batches = int(self._size / batch_size)
        if exclude_terminal_states:
            indices = np.arange(self.size)[~self.dones[:,0].astype(bool)]
        else:
            indices = np.arange(self.size)
        np.random.shuffle(indices)
        batches = np.array_split(indices, n_batches)
        return [self[_indices] for _indices in batches]

    def get_episode(self, episode_number: int | None = None):
        # done_indices, _ = np.where(self.dones == 1)
        # episode_start_indices = np.insert(done_indices + 1, 0, 0)[:-1]
        # episode_end_indices = done_indices+1

        # if episode_number is None:
        #     episode_number = np.random.randint(len(done_indices))
        
        return self[self._episode_starts[episode_number]: self._episode_ends[episode_number] + 1]

    def keep_indices(self, indices):
        self._obss = self._obss[indices]
        self._actions = self._actions[indices]
        self._rewards = self._rewards[indices]
        self._next_obss = self._next_obss[indices]
        self._dones = self._dones[indices]
        self._player_ids = self._player_ids[indices]
        self._action_masks = self._action_masks[indices]
        self._values = self._values[indices]
        self._probs = self._probs[indices]
        self._size = len(self._obss)

    def split_buffer(self, split, shuffle=True):
        indices = np.arange(self._size)

        if shuffle:
            np.random.shuffle(indices)
        split_index = int(len(indices)*(1-split))
        buffer_1_indices = indices[:split_index]
        buffer_2_indices = indices[split_index:]

        buffer_2 = copy.deepcopy(self)
        buffer_2.keep_indices(buffer_2_indices)
    
        buffer_1 = copy.deepcopy(self)
        buffer_1.keep_indices(buffer_1_indices)

        return buffer_1, buffer_2

class BigBuffer(Buffer):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space, capacity=None)
    
    def combine_buffers(self, buffers: list[Buffer]):
        self._size = sum([buffer.size for buffer in buffers])
        self._capacity = self._size
        self.reset(capacity=self._capacity)

        bigbuffer_start_index = 0
        bigbuffer_end_index = -1
        episode_number = 0

        for buffer in buffers:
            for episode, episode_length in buffer._episode_lengths.items():
                bigbuffer_start_index = bigbuffer_end_index + 1
                bigbuffer_end_index += episode_length
                self[bigbuffer_start_index:bigbuffer_end_index+1] = buffer.get_episode(episode)
                
                
                self._episode_starts[episode_number] = bigbuffer_start_index
                self._episode_ends[episode_number] = bigbuffer_end_index
                self._episode_lengths[episode_number] = episode_length
                episode_number += 1
            del buffer