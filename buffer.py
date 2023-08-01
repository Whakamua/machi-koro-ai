import numpy as np
from gym.spaces import flatten, flatten_space, Discrete, MultiBinary, Dict
import random
import copy

class Buffer:
    def __init__(self, gamma, observation_space, action_space, capacity: int | None = None):
        assert isinstance(action_space, Discrete), "other action spaces are not supported due to the reset method assuming a dimension of 1" 
        assert isinstance(observation_space["action_mask"], MultiBinary)
        assert isinstance(observation_space, Dict)
        self._gamma = gamma
        self._observation_space = observation_space
        self._action_space = action_space
        self._capacity = capacity
        self._size = 0
        if self._capacity is not None:
            self.reset(self._capacity)

    def get_big_buffer(self):
        return BigBuffer(self._gamma, self._observation_space, self._action_space)

    @property
    def isfull(self):
        return self._size == self._capacity
    
    @property
    def size(self):
        return self._size

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
        self._is_post_processed = False
        self._obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._actions = np.zeros((capacity, 1))
        self._rewards = np.zeros((capacity, 1))
        self._next_obss = np.zeros((capacity, flatten_space(self._observation_space).shape[0]))
        self._dones = np.zeros((capacity, 1))
        self._player_ids = np.zeros((capacity, 1))
        self._action_masks = np.zeros((capacity, self._observation_space["action_mask"].n))
        self._values = np.zeros((capacity, 1))
        self._probs = np.zeros((capacity, self._action_space.n))

    def add(self, obs, action, reward, next_obs, done, probs):
        self._is_post_processed = False
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
        self._size += 1
        # else:
        #     index = random.randint(0, self._capacity - 1)
        #     self._add_to_idx(index, obs, action, reward, next_obs, done)

    def post_process(self):
        winning_player = None
        self._size = len(self._obss)

        self._values = np.zeros((self._size,1))
        self._winning_player = np.zeros((self._size,1))
        for i in reversed(range(self._size)):
            if self._dones[i]:
                last_terminal_idx = i
                break

        for i in range(last_terminal_idx, -1, -1):
            if self._dones[i]:
                winning_player = self._player_ids[i]
            else:
                self._values[i] = 1 if self._player_ids[i] == winning_player else -1
                self._winning_player[i] = winning_player
                # self._values[i] = self._rewards[i] + self._gamma*self._values[i+1]


        self._non_terminal_indices = np.arange(self._size)[~self._dones[:,0].astype(bool)]
        self._is_post_processed = True

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
            np.random.choice(self._non_terminal_indices, size=batch_size, replace=False)
        else:
            batch_indices = np.random.randint(low=0, high=self._size, size=batch_size, dtype=int)
        return self[batch_indices]
    
    def get_random_batches(self, batch_size, exclude_terminal_states=False):
        assert batch_size <= self._size
        n_batches = int(self._size / batch_size)
        if exclude_terminal_states:
            indices = self._non_terminal_indices
        else:
            indices = np.arange(self._size)
        np.random.shuffle(indices)
        batches = np.array_split(indices, n_batches)
        return [self[_indices] for _indices in batches]

    def get_episode(self, episode_number: int | None = None):
        done_indices, _ = np.where(self.dones == 1)
        episode_start_indices = np.insert(done_indices + 1, 0, 0)[:-1]
        episode_end_indices = done_indices+1

        if episode_number is None:
            episode_number = np.random.randint(len(done_indices))
        
        return self[episode_start_indices[episode_number]: episode_end_indices[episode_number]]
        
    
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

    def split_buffer(self, split, exclude_terminal_states=False, shuffle=True):
        assert self._is_post_processed, "postprocess the buffer before splitting!"

        if exclude_terminal_states:
            indices = self._non_terminal_indices
        else:
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
    def __init__(self, gamma, observation_space, action_space):
        super().__init__(gamma, observation_space, action_space, capacity=None)
    
    def combine_buffers(self, buffers: list[Buffer]):
        self._size = sum([buffer.size for buffer in buffers])
        self._capacity = self._size
        self.reset(capacity=self._capacity)

        start_index = 0
        end_index = 0
        for buffer in buffers:
            start_index = end_index
            end_index += buffer.size
            self[start_index:end_index] = buffer[:]

            del buffer

        self._non_terminal_indices = np.arange(self._size)[~self._dones[:,0].astype(bool)]