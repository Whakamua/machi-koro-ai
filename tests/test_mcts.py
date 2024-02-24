import pytest
from mcts import MCTS, Node
import numpy as np
import copy
import gym

@pytest.fixture
def player_per_step():
    return [0,1,1,0,1,0,0,0]

class dummyEnv():
    def __init__(self, player_per_step):
        self.player_per_step = player_per_step
        self.observation_space = gym.spaces.Dict(
            {
                "count": gym.spaces.Box(low=0, high=np.inf),
                "action_mask": gym.spaces.MultiBinary(3)
            }
        )

    def get_state(self):
        return {"count": copy.deepcopy(self.count)}
    
    def set_state(self, state):
        self.count = copy.deepcopy(state["count"])
    
    def info(self):
        return {"state": self.get_state()}

    @property
    def current_player(self):
        return self.player_per_step[self.count]
    
    def reset(self, state = None):
        self.count = 0
        return {"count": self.count, "action_mask": np.ones(3)}, self.info()
    
    def step(self, action):
        self.count+=1
        end_of_episode = self.count == len(self.player_per_step)-1
        return {"count": self.count, "action_mask": np.ones(3)}, int(end_of_episode), end_of_episode, False, self.info()

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

def test_node_backprop(prior, player_per_step, mcts, env):
    env.reset()
    mcts.reset()
    n0 = mcts.root
    nodes = []
    for i, player in enumerate(player_per_step[1:]): # step 0 is already used for the root node.
        new_node = mcts.find_leaf_node(n0)
        new_node.expand_node(prior=prior, value_estimate=0)
        new_node.backprop()
        nodes.append(new_node)

    for i, node in enumerate(reversed(nodes)):
        if node.parent is None: # initial node it's prior has dirichlet noise
            assert not np.array_equal(node._Psa, prior)
            
        else:
            assert np.array_equal(node._Psa, prior)
        assert np.array_equal(node._Qsa_accumulated, np.array([0 if i==0 else 1-2*player_per_step[-(i+1)], 0, 0]))
        
            
        
        assert np.array_equal(node._Nsa, np.array([i, 0, 0]))
        assert node._accumulated_value == 0 if i==0 else 1-2*player_per_step[-(i+1)]
        assert node._N == i + 1

    # finding the terminal node
    leaf_node = mcts.find_leaf_node(nodes[0])

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
        node = list(node.children[node.best_action].values())[0]
    assert mcts.find_leaf_node(mcts.root).is_terminal

def test_node_puct(mcts, env):
    obs, info = env.reset()
    node_parent = Node(
        observation=obs,
        reward=None,
        done=False,
        player=env.current_player,
        parent=None,
        parent_action=None,
        env_state=info["state"],
        c_puct=1
        )
    node_parent.expand_node(prior=np.array([0.5, 0.5, 0.0]), value_estimate=0)
    node = Node(observation=obs, reward=0, done=False, player=0, parent=node_parent, parent_action=0, env_state=info["state"])
    node.expand_node(prior=np.array([0.5, 0.5, 0.0]), value_estimate=0)

    node._Qsa_accumulated = np.array([8, 16, 32])
    node._Nsa = np.array([4, 4, 0])
    node._N = 125
    assert np.array_equal(node.PUCTsa, np.array([4.5, 6.5, 0]))
    assert mcts.find_leaf_node(node) == list(node.children[1].values())[0]

def test_mcts_afterstates(mcts):
    mcts.reset()
    mcts.find_leaf_node(mcts.root)
    mcts.root._env_state["count"] = 1 # setting the env state of the root node to be different so that another after state would be reached we find_leaf_node is called again
    mcts.find_leaf_node(mcts.root)
    assert len(mcts.root.children[0]) == 2

def test_mcts_compute_probs(mcts, env):
    obs, info = env.reset()
    probs = mcts.compute_probs(None, info["state"])
    assert np.array_equal(mcts.root.Nsa, np.array([10,0,0]))
    assert np.array_equal(probs, np.array([1,0,0]))

def test_mcts_reuse_generated_tree(mcts, env):
    mcts._dirichlet_for_root_node = False
    obs, info = env.reset()
    probs = mcts.compute_probs(None, info["state"])
    assert np.array_equal(mcts.root.Nsa, np.array([10,0,0]))
    assert np.array_equal(probs, np.array([1,0,0]))

    obs, _, _, _, info = env.step(np.argmax(probs))
    probs = mcts.compute_probs(obs, info["state"])
    assert np.array_equal(mcts.root.Nsa, np.array([19,0,0]))
    assert np.array_equal(probs, np.array([1,0,0]))

def test_mcts_qsasprime(mcts):
    mcts.reset()
    mcts.env.player_per_step = [0,0,0,0,0,0,0,0]
    mcts.root._Psa = np.array([1, 0, 0])
    node = mcts.find_leaf_node(mcts.root)
    node.expand_node(np.array([1,0,0]), 1)
    node.backprop()

    mcts.root._env_state["count"] = 1
    node = mcts.find_leaf_node(mcts.root)
    node.expand_node(np.array([1,0,0]), 2)
    node.backprop()

    mcts.root._Psa = np.array([0, 0, 1])
    node = mcts.find_leaf_node(mcts.root)
    node.expand_node(np.array([1,0,0]), 2)
    node.backprop()

    assert np.array_equal(mcts.root.Qsa, np.array([1.5,0,2]))