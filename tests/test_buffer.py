import pytest
from buffer import Buffer, BigBuffer
from gym.spaces import flatten, flatten_space, Discrete, MultiBinary, Dict
from collections import OrderedDict
import numpy as np

@pytest.fixture
def observation_space():
    return Dict(
        {
        "current_player_index": Discrete(2),
        "action_mask": MultiBinary(5)
        }
    )

@pytest.fixture
def action_space():
    return Discrete(5)

@pytest.fixture
def buffer(observation_space, action_space):
    return Buffer(
        gamma=1,
        observation_space=observation_space,
        action_space=action_space,
        capacity=10
    )

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
                obs = OrderedDict([('action_mask', np.array([count, 0, 0, 0, 0])), ('current_player_index', player_first_4_turns)]),
                action = action_space.sample(),
                reward = 0,
                next_obs = OrderedDict([('action_mask', np.array([1, 0, 0, 0, 0])), ('current_player_index', player_first_4_turns if j < 3 else player_last_turn)]),
                done = False,
                probs = np.ones(action_space.n)/action_space.n,
                )
            count+=1
        buffer.add(
            obs = OrderedDict([('action_mask', np.array([count, 0, 0, 0, 0])), ('current_player_index', player_last_turn)]),
            action = action_space.sample(),
            reward = 1,
            next_obs = OrderedDict([('action_mask', np.array([1, 0, 0, 0, 0])), ('current_player_index', player_last_turn)]),
            done = True,
            probs = np.ones(action_space.n)/action_space.n,
            )
        count+=1
        
    buffer.post_process()
    buffer.sample(batch_size=5)
    assert np.array_equal(buffer.values, np.array([[1.], [1.], [1.], [1.], [0.], [-1.], [-1.], [-1.], [-1.], [0.]]))
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

    