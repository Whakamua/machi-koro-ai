import pytest
from mcts import MCTS, Node
import numpy as np
import copy

@pytest.fixture
def player_per_step():
    return [0,1,1,0,1,0,0,0]

class dummyEnv():
    def __init__(self, player_per_step):
        self.player_per_step = player_per_step

    def get_state(self):
        return {"count": copy.deepcopy(self.count)}
    
    def set_state(self, state):
        self.count = copy.deepcopy(state["count"])

    @property
    def current_player(self):
        return self.player_per_step[self.count]
    
    def reset(self, state):
        self.count = 0
        return {"action_mask": np.ones(3)}, {}
    
    def step(self, action):
        self.count+=1
        end_of_episode = self.count == len(self.player_per_step)-1
        return {"action_mask": np.ones(3)}, end_of_episode, end_of_episode, False, {}

@pytest.fixture
def prior():
    return np.array([1.0, 0.0, 0.0])

class dummyPvnet():
    def __init__(self, prior):
        self.prior = prior

    def predict(self, observation):
        return self.prior, 0

@pytest.fixture
def pvnet(prior):
    return dummyPvnet(prior=prior)


@pytest.fixture
def env(player_per_step):
    return dummyEnv(player_per_step=player_per_step)

@pytest.fixture
def mcts(env, pvnet):
    return MCTS(env=env, pvnet=pvnet, num_mcts_sims=10, c_puct=2)

def test_node_backprop(prior, player_per_step):
    
    n0 = Node(parent=None, parent_action=None, c_puct=2)
    nodes = []
    for i, player in enumerate(player_per_step):
        new_node = n0.find_leaf_node()
        new_node.expand_node(prior=prior, reward=i==len(player_per_step)-1, value_estimate=0, env_state=[1,2,3], player=player, is_terminal=i==len(player_per_step)-1) # is_terminal=i==len(player_per_step)-1 is to make the last node a terminal node
        if i < len(player_per_step) - 1:
            new_node.backprop()
        nodes.append(new_node)

    for i, node in enumerate(reversed(nodes)):
        assert np.array_equal(node._Psa, prior)
        assert np.array_equal(node._Qsa_accumulated, np.zeros_like(prior))
        assert np.array_equal(node._Nsa, np.array([i, 0, 0]))
        assert node._accumulated_value == 0
        assert node._N == i + 1
    
    nodes[-1].backprop()

    for i, node in enumerate(nodes):
        if i != len(nodes) - 1:
            assert node.accumulated_value == 1 if player_per_step[i] == player_per_step[-1] else -1

    # finding the terminal node
    leaf_node = nodes[0].find_leaf_node()

    assert leaf_node == nodes[-1]

def test_mcts_search(mcts):
    mcts.reset()
    for i in range(10):
        mcts.search()
        assert mcts.root.N == i+2

    # make sure the leaf node has been found
    node = mcts.root
    for i in range(11, 0,-1):
        assert node.N == i
        if node.is_terminal:
            break
        node = node.children[node.best_action]

    assert mcts.root.find_leaf_node().is_terminal

def test_node_puct():
    node_parent = Node(c_puct=1)
    node_parent.expand_node(prior=np.array([0.5, 0.5, 0.0]), action_mask=np.ones(3), reward=0, value_estimate=0, env_state=[1,2,3], player=0, is_terminal=False)
    node = Node(parent=node_parent, parent_action=0)
    node.expand_node(prior=np.array([0.5, 0.5, 0.0]), action_mask=np.ones(3), reward=0, value_estimate=0, env_state=[1,2,3], player=0, is_terminal=False)

    node._Qsa_accumulated = np.array([8, 16, 32])
    node._Nsa = np.array([4, 4, 0])
    node._N = 125
    assert np.array_equal(node.PUCTsa, np.array([4.5, 6.5, 0]))
    assert node.find_leaf_node() == node.children[1]

