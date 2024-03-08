import pytest
from buffer import Buffer, BigBuffer
from gym.spaces import flatten, flatten_space, Discrete, MultiBinary, Dict
from collections import OrderedDict
import numpy as np
import copy

@pytest.fixture
def observation_space():
    return MultiBinary(6)

@pytest.fixture
def action_space():
    return Discrete(5)

@pytest.fixture
def buffer(observation_space, action_space):
    return Buffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=10
    )

def test_exclude_terminal_states(buffer, observation_space, action_space):
    init_dones = np.array([0,0,1,0,0,1,0,1]).astype(bool)
    for done in init_dones:
        buffer.add(
            obs = np.array([1, 0, 0, 0, 0, 1]),
            action = action_space.sample(),
            reward = 0,
            next_obs = np.array([1, 0, 0, 0, 0, 1]),
            done = done,
            probs = np.ones(action_space.n)/action_space.n,
            current_player_index=1,
            action_mask=np.array([1, 0, 0, 0, 0]),
            )
    obss,_,_,_,_,_,_,_,_ = buffer.sample(batch_size=len(init_dones))
    assert len(obss) == len(init_dones)

    batches = buffer.get_random_batches(batch_size=len(init_dones)/2)
    assert len(batches) == 2
    assert len(batches[0][0]) == len(init_dones)/2
    assert len(batches[1][0]) == len(init_dones)/2

    obss,_,_,_,_,_,_,_,_  = buffer.sample(batch_size=sum(init_dones == 0), exclude_terminal_states=True)
    assert len(obss) == sum(init_dones == 0)

    batches = buffer.get_random_batches(batch_size=np.ceil(sum(init_dones == 0)/2), exclude_terminal_states=True)
    assert len(batches) == 2
    assert len(batches[0][0]) + len(batches[1][0]) == sum(init_dones == 0)

def test_buffer(buffer, observation_space, action_space):
    count = 0
    for i in range(2):
        if i == 0:
            player_first_4_turns = 0
            player_last_turn = 0
        elif i == 1:
            player_first_4_turns = 0
            player_last_turn = 1

        for j in range(4):
            buffer.add(
                obs = np.array([count, 0, 0, 0, 0, player_first_4_turns]),
                action = action_space.sample(),
                reward = 0,
                next_obs = np.array([1, 0, 0, 0, 0, player_first_4_turns if j < 3 else player_last_turn]),
                done = False,
                probs = np.ones(action_space.n)/action_space.n,
                current_player_index=player_first_4_turns if j < 3 else player_last_turn,
                action_mask=np.array([1, 0, 0, 0, 0]),
                )
            count+=1
        buffer.add(
            obs = np.array([count, 0, 0, 0, 0, player_last_turn]),
            action = action_space.sample(),
            reward = 1,
            next_obs = np.array([1, 0, 0, 0, 0, player_last_turn]),
            done = True,
            probs = np.ones(action_space.n)/action_space.n,
            current_player_index=player_last_turn,
            action_mask=np.array([1, 0, 0, 0, 0]),
            )
        count+=1
        
    buffer.compute_values()
    buffer.sample(batch_size=5)
    assert np.array_equal(buffer.values, np.array([[1.], [1.], [1.], [1.], [0.], [-1.], [-1.], [-1.], [1.], [0.]]))
    assert np.array_equal(buffer.rewards, np.array([[0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [1.]]))

    split = 0.2
    buffer_1, buffer_2 = buffer.split_buffer(split, shuffle=False)
    assert np.array_equal(buffer_1.obss[:,0], np.arange(int(buffer.size * (1-split))))
    assert np.array_equal(buffer_2.obss[:,0], np.arange(int(buffer.size * (1-split)), buffer.size, 1))
    
    (
        obss,
        actions,
        rewards,
        next_obss,
        dones,
        player_ids,
        action_masks,
        values,
        probs,
    ) = buffer.get_episode(0)

    assert np.array_equal(obss[:,0], np.arange(5))


def test_combine_buffer(buffer, action_space):

    for i in range(10):
        buffer.add(
            obs = np.array([1, 0, 0, 0, 0, 1]),
            action = action_space.sample(),
            reward = int((i-1) % 3 == 0),
            next_obs = np.array([1, 0, 0, 0, 0, 1]),
            done = (i-1) % 3 == 0,
            probs = np.ones(action_space.n)/action_space.n,
            current_player_index=1,
            action_mask=np.array([1, 0, 0, 0, 0])
        )
    buffer2 = copy.deepcopy(buffer)
    bigbuffer = buffer.get_big_buffer()
    bigbuffer.combine_buffers([buffer, buffer2])
    bigbuffer.compute_values()
    assert np.array_equal(bigbuffer.values, np.array([
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0]
    ]))

def test_split_buffer(buffer, action_space):
    for i in range(10):
        buffer.add(
            obs = np.array([i, 0, 0, 0, 0, 1]),
            action = action_space.sample(),
            reward = int((i-1) % 3 == 0),
            next_obs = np.array([1, 0, 0, 0, 0, 1]),
            done = (i-1) % 3 == 0,
            probs = np.ones(action_space.n)/action_space.n,
            current_player_index=1,
            action_mask=np.array([1, 0, 0, 0, 0])
        )
    
    buffer1, buffer2 = buffer.split_buffer_by_episode(0.2)

    assert len(buffer1[:]) == len(buffer2[:])

    for i in range(len(buffer1[:])):
        assert buffer1[:][i].shape[0] + buffer2[:][i].shape[0] == buffer[:][i].shape[0]
    
    buffer_episode_lengths = list(buffer._episode_lengths.values())
    for episode_len in buffer1._episode_lengths.values():
        assert episode_len in buffer_episode_lengths
        del buffer_episode_lengths[buffer_episode_lengths.index(episode_len)]
    for episode_len in buffer2._episode_lengths.values():
        assert episode_len in buffer_episode_lengths
        del buffer_episode_lengths[buffer_episode_lengths.index(episode_len)]
    
    assert not buffer_episode_lengths
