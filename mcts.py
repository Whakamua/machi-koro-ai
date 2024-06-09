# source: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
from __future__ import annotations

import numpy as np
import copy
import gym
import time


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, pvnet, num_mcts_sims, c_puct, dirichlet_for_root_node = True, thinking_time: int = None, print_info: bool = False):
        self.env = env
        self.pvnet = pvnet
        self.num_mcts_sims = num_mcts_sims
        self._c_puct = c_puct
        self._root = None
        self._dirichlet_for_root_node = dirichlet_for_root_node
        self._thinking_time = thinking_time
        self._print_info = print_info
        print("I lovelovelove you")
    
    @property
    def root(self):
        return self._root
    
    def update_pvnet(self, state_dict):
        self.pvnet.use_uniform_pvnet = state_dict["uniform_pvnet"]
        self.pvnet.use_custom_policy_edit = state_dict["custom_policy_edit"]
        self.pvnet.use_custom_value = state_dict["custom_value"]
        self.pvnet.load_state_dict(state_dict["pvnet_state_dict"])

    def add_dirichlet(self, node):
        epsilon = 0.25
        noise = np.random.default_rng().dirichlet((tuple(np.ones(len(node.Psa))*0.3)),1)[0]
        node._Psa = (1-epsilon)*node.Psa + epsilon*noise

    def reset(self, env_state: dict | None = None):
        del self._root
        observation, info = self.env.reset(env_state)
        root_node = Node(
            observation=observation,
            action_mask=self.env.action_mask(),
            reward=None,
            done=False,
            player=self.env.current_player,
            parent=None,
            parent_action=None,
            env_state=observation,
            c_puct=self._c_puct,
        )
        prior, value_estimate = self.pvnet.predict(observation)
        node = self.find_leaf_node(root_node)
        assert node == root_node
        root_node.expand_node(prior=prior, value_estimate=value_estimate.item())
        self.set_root(root_node)

    def set_root(self, root: Node):
        self._root = root

    def search(self, node, n: int = None, thinking_time: int = None):
        self.n_sims_executed = 0
        if thinking_time is not None:
            start_time = time.time()
            while True:
                time_spent = time.time() - start_time
                if self._print_info:
                    print(f"Thinking... {np.round(time_spent/thinking_time * 100, 1)}% done.", end="\r")
                if time_spent > thinking_time:
                    break
                self._search(node)
                self.n_sims_executed += 1
        elif n is not None:
            for i in range(n):
                if self._print_info:
                    print(f"Thinking... {np.round(i+1/n * 100, 1)}% done.", end="\r")
                self._search(node)
            self.n_sims_executed = n
        else:
            raise ValueError("Specify number of MCTS simulations or thinking time")

            

    def _search(self, node):
        leaf_node = self.find_leaf_node(node)
            
        if leaf_node.is_terminal:
            assert leaf_node.player == leaf_node.parent.player
        else:
            prior, value_estimate = self.pvnet.predict(leaf_node.observation)

            leaf_node.expand_node(prior=prior, value_estimate=value_estimate.item())

        leaf_node.backprop(root_node=node)


    def compute_probs(self, observation):
        reset_tree = True

        if observation is not None:
            obs_as_str = self.obs_as_str(observation)

            for afterstates in self.root.children.values():
                if obs_as_str in afterstates:
                    reset_tree = False
                    self.set_root(afterstates[obs_as_str])
                    break
        if reset_tree:
            self.reset(observation)

        assert np.array_equal(self.root.env_state, observation)

        if self._dirichlet_for_root_node:
            self.add_dirichlet(self.root)

        self.search(self.root, self.num_mcts_sims, self._thinking_time)
        if self.n_sims_executed > 0:
            return self.root.Nsa / self.root.Nsa.sum()
        else:
            return self.root.PUCTsa


    def obs_as_str(self, obs):
        return obs.tobytes()
        

    def find_leaf_node(self, node):
        node._N += 1
        
        if node.parent is not None:
            node.parent._Nsa[node.parent_action] += 1
        if node.is_leaf_node or node.is_terminal:
            return node
        else:
            self.env.set_state(copy.deepcopy(node.env_state))
            best_action = node.best_action
            obs, reward, done, _, info = self.env.step(best_action)
            
            obs_as_str = self.obs_as_str(obs)
            if obs_as_str not in node.children[best_action]:
                node.children[best_action][obs_as_str] = Node(
                    observation=obs,
                    action_mask=self.env.action_mask(),
                    reward=reward,
                    done=done,
                    player=self.env.current_player,
                    parent=node,
                    parent_action=best_action,
                    env_state=obs
                )
            return self.find_leaf_node(node.children[best_action][obs_as_str])


class Node:
    def __init__(
            self,
            observation: np.ndarray,
            action_mask: np.ndarray,
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
        self._action_mask = action_mask
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
    
    def get_child(self, action_idx, afterstate_idx):
        return list(self.children[action_idx].values())[afterstate_idx]

    @property
    def PUCTsa(self):
        PUCTsa = self.Qsa + self.c_puct * self.Psa * (self.N/ (1+self.Nsa))**0.5
        PUCTsa[self._action_mask == 0] = -np.inf
        
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

    def _backprop(self, leaf_node_value: float, leaf_node_player, root_node):
        if self.parent is None or self == root_node:
            return self
        parent_value_multiplier = (-1 + 2*(leaf_node_player == self.parent.player))

        self.parent._accumulated_value += parent_value_multiplier * leaf_node_value
        self.parent._Qsa_accumulated[self.parent_action] += parent_value_multiplier * leaf_node_value
        self.parent._backprop(
            leaf_node_value=leaf_node_value,
            leaf_node_player=leaf_node_player,
            root_node=root_node
        )

    def backprop(self, root_node):
        return self._backprop(
            leaf_node_value=self.reward if self.is_terminal else self.value_estimate,
            leaf_node_player=self.player,
            root_node=root_node
        )
