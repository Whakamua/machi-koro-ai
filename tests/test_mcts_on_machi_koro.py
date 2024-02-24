import pytest
from mcts_agent import MCTSAgent
from env import MachiKoro, GymMachiKoro
import numpy as np

class dummyPvnet():
    def __init__(self):
        pass

    def predict(self, observation):
        return observation["action_mask"] / observation["action_mask"].sum(), 0

@pytest.fixture
def env():
    return GymMachiKoro(MachiKoro(n_players=2))

@pytest.fixture
def pvnet(env):
    return dummyPvnet()

@pytest.fixture
def mctsagent(env, pvnet):
    mctsagent = MCTSAgent(env=env, num_mcts_sims=10, c_puct=2)
    mctsagent.mcts.pvnet = pvnet
    return mctsagent

def test_mcts_find_optimal_actions_with_random_policy_net(env, mctsagent):
    obs, info = env.reset()
    state = info["state"]
    for card, info in env._env._card_info.items():
        if card == "Harbor":
            continue
        if info["type"] == "Landmarks":
            state.player_info["player 0"]._coins = env._env._card_info[card]["cost"]
            state.player_info["player 0"].buy_card(card)
    
    state.player_info["player 0"]._coins = env._env._card_info["Harbor"]["cost"]
    env.set_state(state)
    mctsagent.reset(state)

    action, probs = mctsagent.compute_action(env.observation(), state)
    assert action == env._action_str_to_idx["1 dice"]

    observation, reward, done, _, info = env.step(action)
    mctsagent.reset(info["state"])
    mctsagent.mcts.num_mcts_sims = 100
    action, probs = mctsagent.compute_action(env.observation(), info["state"])
    assert action == env._action_str_to_idx["Harbor"]
    observation, reward, done, _, info = env.step(action)
    assert done