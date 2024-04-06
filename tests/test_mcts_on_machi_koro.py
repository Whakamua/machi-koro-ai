import pytest
from mcts_agent import MCTSAgent
from env import GymMachiKoro
import numpy as np
import copy

class dummyPvnet():
    def __init__(self, env):
        self.env = env

    def predict(self, observation):
        self.env.set_state(observation)
        action_mask = self.env.action_mask()
        return action_mask / action_mask.sum(), 0

@pytest.fixture
def env():
    return GymMachiKoro(n_players=2)

@pytest.fixture
def pvnet(env):
    return dummyPvnet(copy.deepcopy(env))

@pytest.fixture
def mctsagent(env, pvnet):
    mctsagent = MCTSAgent(env=copy.deepcopy(env), num_mcts_sims=10, c_puct=2, pvnet=pvnet)
    return mctsagent

def test_mcts_find_optimal_actions_with_random_policy_net(env, mctsagent):
    # obs, info = env.reset()
    # state = info["state"]
    for card_name, card_info in env._env._card_info.items():
        if card_name == "Harbor":
            continue
        if card_info["type"] == "Landmarks":
            env._env.add_card("player 0", card_name)
    
    obs = env.observation()
    mctsagent.reset(obs)

    action, probs = mctsagent.compute_action(obs)
    assert action == env._action_str_to_idx["1 dice"]

    observation, reward, done, _, info = env.step(action)
    mctsagent.reset(observation)
    mctsagent.mcts.num_mcts_sims = 100
    action, probs = mctsagent.compute_action(observation)
    try:
        assert env._action_idx_to_str[action] == "Harbor"
    except:
        breakpoint()
    
    observation, reward, done, _, info = env.step(action)
    assert done