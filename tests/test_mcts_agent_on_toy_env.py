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

        self.stochastic = stochastic
        self.reset()
    
    def action_mask(self):
        np.ones(2).astype(int)

    @property
    def current_player(self):
        return False

    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state

    def info(self):
        return {}

    def observation(self):
        return copy.deepcopy(self.state)

    def reset(self, state: dict | None = None):
        if state is not None:
            self.set_state(state)
        else:
            self.state = np.array([0])
        return self.observation(), self.info()
    
    def step(self, action):
        if self.stochastic and np.array_equal(self.state, np.array([0,0,1])) and action == 1 and random.random() > 0.3:
            self.state = np.append(self.state, action*2)
        else:
            self.state = np.append(self.state, action)

        if np.array_equal(self.state, np.array([0,0,1,1,0,0])):
            reward = 2
        elif np.array_equal(self.state, np.array([0,0,1,0,1,0])):
            reward = 1
        else:
            reward = 0

        if len(self.state) == 6:
            done = True
        else:
            done = False


        return self.observation(), reward, done, False, self.info()

@pytest.fixture
def env():
    return ToyEnv(stochastic=False)

@pytest.fixture
def stochastic_env():
    return ToyEnv(stochastic=True)

def test_mcts_agent_gets_optimal_policy(env, pvnet):
    agent = MCTSAgent(copy.deepcopy(env), 100, c_puct=2, pvnet=pvnet, dirichlet_to_root_node=False)

    obs, info = env.reset()
    agent.reset(obs)

    actions = []
    while True:
        action, probs = agent.compute_action(obs)
        obs, reward, done, _, info = env.step(action)
        actions.append(action)
        if done:
            break

    assert actions == [0, 1, 1, 0, 0]
    
    assert reward == 2

def test_mcts_agent_gets_optimal_policy_in_stochastic_env(stochastic_env, pvnet):
    stochastic_env_copy = copy.deepcopy(stochastic_env)
    pvnet_copy = copy.deepcopy(pvnet)
    agent = MCTSAgent(copy.deepcopy(stochastic_env_copy), 1000, c_puct=2, pvnet=pvnet_copy, dirichlet_to_root_node=False)

    obs, info = stochastic_env_copy.reset()
    agent.reset(obs)

    actions = []
    while True:
        
        action, probs = agent.compute_action(obs)
        node = agent.mcts.root
        while True:
            if node.parent is not None:
                node = node.parent
            else:
                break

        obs, reward, done, _, info = stochastic_env_copy.step(action)
        actions.append(action)
        if done:
            break
    assert actions == [0, 1, 0, 1, 0]

    assert reward == 1