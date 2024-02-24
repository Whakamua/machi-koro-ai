from mcts_agent import MCTSAgent
import pytest
import gym
import numpy as np
import random
import copy
from collections import OrderedDict

@pytest.fixture
def prior():
    return np.array([0.5, 0.5])

class dummyPvnet():
    def __init__(self, prior):
        self.prior = prior

    def predict(self, observation):
        return self.prior, 0

@pytest.fixture
def pvnet(prior):
    return dummyPvnet(prior=prior)

class ToyEnv:
    def __init__(self, stochastic):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            OrderedDict([("state", gym.spaces.Text(6)), ("action_mask", gym.spaces.MultiDiscrete(2))])
        )
        self._current_player = False

        self.stochastic = stochastic

    @property
    def current_player(self):
        return int(self._current_player)

    def get_state(self):
        return {
            "state": copy.deepcopy(self.state),
            "_current_player": copy.deepcopy(self._current_player)
        }
    
    def set_state(self, state):
        self.state = copy.deepcopy(state["state"])
        self._current_player = copy.deepcopy(state["_current_player"])

    def info(self):
        return {"state": self.get_state()}

    def observation(self):
        return OrderedDict([("state", self.state), ("action_mask", np.ones(2).astype(int))])

    def reset(self, state: dict | None = None):
        if state is not None:
            self.set_state(state)
        else:
            self.state = "0"
        return self.observation(), self.info()
    
    def step(self, action):
        if self.stochastic and self.state == "001" and action == 1 and random.random() > 0.3:
            self.state += str(action*2)
        else:
            self.state += str(action)

        if self.state == "001100":
            reward = 2
        elif self.state == "001010":
            reward = 1
        else:
            reward = 0
        
        if len(self.state) == 6:
            done = True
        else:
            # self._current_player = not self._current_player
            done = False


        return self.observation(), reward, done, False, self.info()

@pytest.fixture
def env():
    return ToyEnv(stochastic=False)

@pytest.fixture
def stochastic_env():
    return ToyEnv(stochastic=True)

def test_mcts_agent_gets_optimal_policy(env, pvnet):
    agent = MCTSAgent(env, 100, c_puct=2, pvnet=pvnet, dirichlet_to_root_node=False)

    obs, info = env.reset()
    agent.reset(info["state"])

    actions = []
    while True:
        action, probs = agent.compute_action(obs, info["state"])
        obs, reward, done, _, info = env.step(action)
        actions.append(action)
        if done:
            break

    try:
        assert actions == [0, 1, 1, 0, 0]
    except:
        breakpoint()
    
    assert reward == 2

def test_mcts_agent_gets_optimal_policy_in_stochastic_env(stochastic_env, pvnet):
    agent = MCTSAgent(stochastic_env, 100, c_puct=2, pvnet=pvnet, dirichlet_to_root_node=False)

    obs, info = stochastic_env.reset()
    agent.reset(info["state"])

    actions = []
    while True:
        
        action, probs = agent.compute_action(obs, info["state"])
        obs, reward, done, _, info = stochastic_env.step(action)
        actions.append(action)
        if done:
            break
    try:
        assert actions == [0, 1, 0, 1, 0]
    except:
        breakpoint()
    
    assert reward == 1