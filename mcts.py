# source: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
from __future__ import annotations

import numpy as np
import copy
import gym


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, pvnet, num_mcts_sims, c_puct, dirichlet_for_root_node = True):
        self.env = copy.deepcopy(env)
        self.pvnet = pvnet
        self.num_mcts_sims = num_mcts_sims
        self._c_puct = c_puct
        self._root = None
        self._dirichlet_for_root_node = dirichlet_for_root_node
        print("I lovelovelove you")
    
    @property
    def root(self):
        return self._root
    
    def update_pvnet(self, pvnet):
        self.pvnet = copy.deepcopy(pvnet)

    def add_dirichlet(self, node):
        epsilon = 0.25
        noise = np.random.default_rng().dirichlet((tuple(np.ones(len(node.Psa))*0.3)),1)[0]
        node._Psa = (1-epsilon)*node.Psa + epsilon*noise

    def reset(self, env_state: dict | None = None):
        del self._root
        observation, info = self.env.reset(env_state)
        root_node = Node(
            observation=observation,
            reward=None,
            done=False,
            player=self.env.current_player,
            parent=None,
            parent_action=None,
            env_state=info["state"],
            c_puct=self._c_puct,
        )
        prior, value_estimate = self.pvnet.predict(observation)
        node = self.find_leaf_node(root_node)
        assert node == root_node
        root_node.expand_node(prior=prior, value_estimate=value_estimate)
        self.set_root(root_node)

    def set_root(self, root: Node):
        self._root = root

    def search(self):
        leaf_node = self.find_leaf_node(self.root)
        
        if leaf_node.is_terminal:
            assert leaf_node.player == leaf_node.parent.player
        else:
            prior, value_estimate = self.pvnet.predict(leaf_node.observation)

            leaf_node.expand_node(prior=prior, value_estimate=value_estimate)

        leaf_node.backprop()

    def compute_probs(self, observation, env_state):
        reset_tree = True

        if observation is not None:
            obs_as_str = self.obs_as_str(observation)

            for afterstates in self.root.children.values():
                if obs_as_str in afterstates:
                    reset_tree = False
                    self.set_root(afterstates[obs_as_str])
                    break
        if reset_tree:
            self.reset(env_state=env_state)

        assert self.root.env_state == env_state

        if self._dirichlet_for_root_node:
            self.add_dirichlet(self.root)

        for _ in range(self.num_mcts_sims):
            self.search()
        return self.root.Nsa / self.root.Nsa.sum()

    def obs_as_str(self, obs):
        return gym.spaces.flatten(self.env.observation_space, obs).tobytes()
        

    def find_leaf_node(self, node):
        node._N += 1
        
        if node.parent is not None:
            node.parent._Nsa[node.parent_action] += 1
        if node.is_leaf_node or node.is_terminal:
            return node
        else:
            self.env.set_state(node.env_state)
            obs, reward, done, _, info = self.env.step(node.best_action)
            
            obs_as_str = self.obs_as_str(obs)
            if obs_as_str not in node.children[node.best_action]:
                node.children[node.best_action][obs_as_str] = Node(
                    observation=obs,
                    reward=reward,
                    done=done,
                    player=self.env.current_player,
                    parent=node,
                    parent_action=node.best_action,
                    env_state=info["state"]
                )
            return self.find_leaf_node(node.children[node.best_action][obs_as_str])


class Node:
    def __init__(
            self,
            observation: gym.spaces.Dict,
            reward: float,
            done: bool,
            player: int,
            parent: Node | None = None,
            parent_action: int | None = None,
            env_state: dict | None = None,
            c_puct: int | None = None,
        ):
        self._env_state = env_state
        self._parent = parent
        self._Psa = None
        self._Qsa_accumulated = None
        self._Nsa = None
        self._reward = reward
        self._observation = observation
        self._action_mask = observation["action_mask"].astype(bool)
        self._player = player

        self._c_puct = c_puct or self._parent.c_puct
        self._parent_action = parent_action

        self._children = {}
        self._N = 0
        self._accumulated_value = 0
        self._is_terminal = done
        self._is_expanded = False

    @property
    def observation(self):
        return self._observation

    @property
    def reward(self):
        return self._reward

    @property
    def value(self):
        return self._accumulated_value / self._N

    @property
    def Qsa_accumulated(self):
        return self._Qsa_accumulated

    @property
    def Qsa(self):
        return np.divide(self.Qsa_accumulated, self.Nsa, out=np.zeros_like(self.Nsa.astype(float)), where=self.Nsa!=0)

    @property
    def parent_action(self):
        return self._parent_action

    @property
    def c_puct(self):
        return self._c_puct

    @property
    def is_leaf_node(self):
        return not self._is_expanded

    @property
    def parent(self):
        return self._parent

    @property
    def Psa(self):
        return self._Psa

    @property
    def Nsa(self):
        return self._Nsa
    
    @property
    def N(self):
        return self._N
    
    @property
    def accumulated_value(self):
        return self._accumulated_value
    
    @property
    def env_state(self):
        return self._env_state

    @property
    def is_terminal(self):
        return self._is_terminal

    def expand_node(self, prior, value_estimate):
        self._Psa = prior
        self._Qsa_accumulated = np.zeros_like(prior)
        self._Nsa = np.zeros_like(prior)

        self._value_estimate = value_estimate
        self._children = {i: {} for i in range(len(prior))}
        self._accumulated_value = value_estimate
        self._is_expanded = True

    @property
    def action_mask(self):
        return self._action_mask.astype(int)

    @property
    def value_estimate(self):
        return self._value_estimate

    @property
    def player(self):
        return self._player

    @property
    def children(self):
        return self._children

    @property
    def PUCTsa(self):
        PUCTsa = self.Qsa + self.c_puct * self.Psa * (self.N/ (1+self.Nsa))**0.5
        PUCTsa[~self._action_mask] = -np.inf
        
        return PUCTsa
    
    @property
    def Qsa_estimate(self):
        Qsa_estimate = []
        for action, afterstates in self.children.items():
            value_estimate_sum = 0
            count = 0
            for node in afterstates.values():
                value_estimate_sum += node.value_estimate
                count += 1
            Qsa_estimate.append(value_estimate_sum/count if count != 0 else 0)
        return Qsa_estimate

    @property
    def Qsasprime(self):
        Qsasprime = {}
        for action, afterstates in self.children.items():
            Qsasprime[action] = []
            for node in afterstates.values():
                Qsasprime[action].append(node.value)
        return Qsasprime

    @property
    def best_action(self):
        return np.argmax(self.PUCTsa)

    def _backprop(self, leaf_node_value: float, leaf_node_player):
        if self.parent is None:
            return self
        parent_value_multiplier = (-1 + 2*(leaf_node_player == self.parent.player))

        self.parent._accumulated_value += parent_value_multiplier * leaf_node_value
        self.parent._Qsa_accumulated[self.parent_action] += parent_value_multiplier * leaf_node_value
        self.parent._backprop(leaf_node_value=leaf_node_value, leaf_node_player=leaf_node_player)

    def backprop(self):
        return self._backprop(leaf_node_value=self.reward if self.is_terminal else self.value_estimate, leaf_node_player=self.player)
