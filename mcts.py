# source: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
from __future__ import annotations

import numpy as np
import warnings
import copy


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, pvnet, num_mcts_sims, c_puct):
        self.env = env
        self.pvnet = pvnet
        self.num_mcts_sims = num_mcts_sims
        self._c_puct = c_puct
        self._root = None
        warnings.warn("MCTS is currently not reusing the tree generated in the previous action, see compute_action")
        print("I lovelovelove you")
    
    @property
    def root(self):
        return self._root
    
    def update_pvnet(self, pvnet):
        self.pvnet = copy.deepcopy(pvnet)

    def reset(self, env_state: dict | None = None):
        del self._root
        observation, _ = self.env.reset(env_state)
        root_node = Node(
            c_puct=self._c_puct,
            env_state=self.env.get_state()
        )
        prior, value_estimate = self.pvnet.predict(observation)
        node = root_node.find_leaf_node()
        assert node == root_node
        root_node.expand_node(prior=prior, action_mask=observation["action_mask"], value_estimate=value_estimate, env_state=self.env.get_state(), player=self.env.current_player, reward=0)
        self.set_root(root_node)

    def set_root(self, root: Node):
        self._root = root

    def search(self):
        leaf_node = self.root.find_leaf_node()
        if not leaf_node.is_terminal:
            self.env.set_state(leaf_node.parent.env_state)
            observation, reward, done, _, info = self.env.step(leaf_node.parent_action)
            
            prior, value_estimate = self.pvnet.predict(observation)

            leaf_node.expand_node(prior=prior, action_mask=observation["action_mask"], reward=reward, value_estimate=value_estimate, env_state=self.env.get_state(), player=self.env.current_player, is_terminal=done)

        if leaf_node.is_terminal:
            assert leaf_node.player == leaf_node.parent.player
        leaf_node.backprop()

    def compute_probs(self, observation, env_state):
        self.reset(env_state=env_state)
        assert self.root.env_state == env_state
        for _ in range(self.num_mcts_sims):
            self.search()
        return self.root.Nsa / self.root.Nsa.sum()

class Node:
    def __init__(self, parent: Node | None = None, parent_action: int | None = None, env_state: dict | None = None, c_puct: int | None = None):
        self._env_state = env_state
        self._parent = parent
        self._Psa = None
        self._Qsa_accumulated = None
        self._Nsa = None
        self._reward = None

        self._c_puct = c_puct or self._parent.c_puct
        self._parent_action = parent_action

        self._children = {}
        self._N = 0
        self._accumulated_value = 0
        self._is_terminal = None

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
        return self._children == {}

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

    def expand_node(self, prior, action_mask, reward, value_estimate, env_state, player, is_terminal: bool = False):
        self._env_state = env_state
        if self.parent is None:
            epsilon = 0.25
            noise = np.random.default_rng().dirichlet((tuple(np.ones(len(prior))*0.3)),1)[0]
            prior = (1-epsilon)*prior + epsilon*noise

        self._Psa = prior
        self._reward = reward
        self._Qsa_accumulated = np.zeros_like(prior)
        self._Nsa = np.zeros_like(prior)
        self._is_terminal = is_terminal

        self._value_estimate = value_estimate
        self._children = {i: Node(self, i) for i in range(len(prior))}
        self._accumulated_value = value_estimate
        self._player = player
        self._action_mask = action_mask.astype(bool)

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
    def best_action(self):
        return np.argmax(self.PUCTsa)

    def find_leaf_node(self):
        self._N += 1
        if self.parent is not None:
            self.parent._Nsa[self.parent_action] += 1
        if self.is_leaf_node or self.is_terminal:
            return self
        else:
            return self.children[self.best_action].find_leaf_node()

    def _backprop(self, leaf_node_value: float, leaf_node_player):
        if self.parent is None:
            return self
        parent_value_multiplier = (-1 + 2*(leaf_node_player == self.parent.player))

        self.parent._accumulated_value += parent_value_multiplier * leaf_node_value
        self.parent._Qsa_accumulated[self.parent_action] += parent_value_multiplier * leaf_node_value
        self.parent._backprop(leaf_node_value=leaf_node_value, leaf_node_player=leaf_node_player)

    def backprop(self):
        return self._backprop(leaf_node_value=self.reward if self.is_terminal else self.value_estimate, leaf_node_player=self.player)






    # def getActionProb(self, env_state, temp=1):
    #     """
    #     This function performs numMCTSSims simulations of MCTS starting from
    #     env_state.

    #     Returns:
    #         probs: a policy vector where the probability of the ith action is
    #                proportional to Nsa[(s,a)]**(1./temp)
    #     """
    #     self.root
    #     self.env.set_state(env_state)
    #     obs = self.env.observation()
    #     print(f"{obs=}")

    #     s = self.env.getStringRepresentation(env_state)
    #     print(f"{s=}")
    #     for i in range(self.num_mcts_sims):
    #         self.search(env_state)

    #     self.env.set_state(env_state)
    #     obs = self.env.observation()
    #     print(f"{obs=}")
    #     s = self.env.stringRepresentation
    #     print(f"{s=}")
    #     counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.env.getActionSize())]

    #     if temp == 0:
    #         bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
    #         bestA = np.random.choice(bestAs)
    #         probs = [0] * len(counts)
    #         probs[bestA] = 1
    #         return probs

    #     counts = [x ** (1. / temp) for x in counts]
    #     counts_sum = float(sum(counts))
    #     try:
    #         probs = [x / counts_sum for x in counts]
    #     except:
    #         breakpoint()
        
    #     return probs

    # def search(self, env_state):
    #     """
    #     This function performs one iteration of MCTS. It is recursively called
    #     till a leaf node is found. The action chosen at each node is one that
    #     has the maximum upper confidence bound as in the paper.

    #     Once a leaf node is found, the neural network is called to return an
    #     initial policy P and a value v for the state. This value is propagated
    #     up the search path. In case the leaf node is a terminal state, the
    #     outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
    #     updated.

    #     NOTE: the return values are the negative of the value of the current
    #     state. This is done since v is in [-1,1] and if v is the value of a
    #     state for the current player, then its value is -v for the other player.

    #     Returns:
    #         v: the negative of the value of the current env_state
    #     """
        
    #     s = self.env.getStringRepresentation(env_state)

    #     if s not in self.Es:
    #         self.Es[s] = self.env.getIsTerminal(env_state)
    #     if self.Es[s] != 0:
    #         # terminal node
    #         return -self.Es[s]

    #     print("action_maskz: ", self.env.action_mask)
    #     if s not in self.Ps:
    #         # leaf node
    #         self.Ps[s], v = self.pvnet.predict(self.env.getObservation(env_state))
    #         valids = self.env.getValidMoves(env_state)
    #         print(f"{valids=}")
    #         print("action_mask1: ", self.env.action_mask)
    #         self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
    #         print("action_maska: ", self.env.action_mask)
    #         sum_Ps_s = np.sum(self.Ps[s])
    #         print("action_maskb: ", self.env.action_mask)
    #         if sum_Ps_s > 0:
    #             self.Ps[s] /= sum_Ps_s  # renormalize
    #             print("action_maskc: ", self.env.action_mask)
    #         else:
    #             # if all valid moves were masked make all valid moves equally probable

    #             # NB! All valid moves may be masked if either your pvnet architecture is insufficient or you've get overfitting or something else.
    #             # If you have got dozens or hundreds of these messages you should pay attention to your pvnet and/or training process.   
    #             log.error("All valid moves were masked, doing a workaround.")
    #             self.Ps[s] = self.Ps[s] + valids
    #             self.Ps[s] /= np.sum(self.Ps[s])
    #             print("action_maskd: ", self.env.action_mask)

    #         self.Vs[s] = valids
    #         self.Ns[s] = 0
    #         print("action_maske: ", self.env.action_mask)
    #         return -v
        
    #     print("action_mask2: ", self.env.action_mask)

    #     valids = self.Vs[s]
    #     cur_best = -float('inf')
    #     best_act = -1
    #     print("action_mask3: ", self.env.action_mask)

    #     # pick the action with the highest upper confidence bound
    #     for a in range(self.env.getActionSize()):
    #         if valids[a]:
    #             if (s, a) in self.Qsa:
    #                 u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
    #                         1 + self.Nsa[(s, a)])
    #             else:
    #                 u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

    #             if u > cur_best:
    #                 cur_best = u
    #                 best_act = a
    #     print("action_mask4: ", self.env.action_mask)

    #     a = best_act
    #     if self.env.action_mask[a] == 0:
    #         breakpoint()
    #     obs, reward, done, _, info = self.env.step(a)
    #     next_s = self.env.get_state()

    #     v = self.search(next_s)

    #     if (s, a) in self.Qsa:
    #         self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
    #         self.Nsa[(s, a)] += 1

    #     else:
    #         self.Qsa[(s, a)] = v
    #         self.Nsa[(s, a)] = 1

    #     self.Ns[s] += 1
    #     return -v