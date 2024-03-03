import gym
import yaml
from collections import deque
import random
import copy
import numpy as np
from collections import OrderedDict
from typing import Optional

class MachiKoro:
    def __init__(self, n_players: int):
        self._n_players = n_players

        with open('card_info.yaml') as f:
        # with open('card_info_quick_game.yaml') as f:
            self._card_info = yaml.load(f, Loader=yaml.loader.SafeLoader)

        self._cards_per_activation = {i: [] for i in range(1, 13)}
        for card, info in self._card_info.items():
            if info["type"] != "Landmarks":
                for activation in range(info["activation"][0], info["activation"][1] + 1):
                    self._cards_per_activation[activation].append(card)

        self._landmark_cards_ascending_in_price = [card for card, info in self._card_info.items() if info["type"] == "Landmarks"]
        self._max_landmarks = len(self._landmark_cards_ascending_in_price)
        self._init_establishments = {
            "1-6": [],
            "7+": [],
            "major": [],
        }

        self._cards_per_icon = {}
        for card, info in self._card_info.items():
            if info["icon"] not in self._cards_per_icon:
                self._cards_per_icon[info["icon"]] = []
            self._cards_per_icon[info["icon"]].append(card)

        for card_name, info in self._card_info.items():
            if info["type"] == "Landmarks":
                continue
            elif info["type"] == "Major Establishment":
                self._init_establishments["major"].extend([card_name]*info["n_cards"])
            elif info["activation"][0] <= 6:
                self._init_establishments["1-6"].extend([card_name]*info["n_cards"])
            else:
                self._init_establishments["7+"].extend([card_name]*info["n_cards"])

        self._stage_order = ["diceroll", "build"]
        self._n_stages = len(self._stage_order)
        self._player_order = [f"player {i}" for i in range(self._n_players)]

        self.reset()

    def reset(self):

        # construct state
        self.state = []
        self._state_indices = {}
        self._state_indices["player_info"] = {}
        self.observable_indices = []

        # 1. player_info construct vectors and indices for each player its own state
        ### [n_card_1_p1, n_card_2_p1, ..., n_card_N_pN, n_coins_pN, startup_inv_pN]
        for player in self._player_order:
            self._state_indices["player_info"][player] = {}

            # buildings in city
            self._state_indices["player_info"][player]["cards"] = {}
            for card in self._card_info.keys():
                self._state_indices["player_info"][player]["cards"][card] = len(self.state)
                self.observable_indices.append(len(self.state))
                self.state.append(card in ["Wheat Field", "Bakery"])

            # coins
            self._state_indices["player_info"][player]["coins"] = len(self.state)
            self.observable_indices.append(len(self.state))
            self.state.append(3)
            
            # tech startup investment
            self._state_indices["player_info"][player]["tech_startup_investment"] = len(self.state)
            self.observable_indices.append(len(self.state))
            self.state.append(0)

        
        # 2. state of the deques that lie on the table
        ### [1_6_deque_card_1, 1_6_deque_card_2, ..., _major_deque_card_N]
            
        # the cards will be enumerated, to keep things overseeable there's a name to index mapping.
        self._card_name_to_num = {name: i for i, (name, info) in enumerate(list(self._card_info.items()))}
        self._card_name_to_num["None"] = -1
        self._card_num_to_name = {i: name for name, i in self._card_name_to_num.items()}
    
        self._1_6_deque = []
        self._7_plus_deque = []
        self._major_deque = []

        # create deques unshuffled
        for card_name, card_info in self._card_info.items():
            if card_info["type"] == "Landmarks":
                continue
            elif card_info["type"] == "Major Establishment":
                self._major_deque.extend([self._card_name_to_num[card_name]]*card_info["n_cards"])
            elif card_info["activation"][0] <= 6:
                self._1_6_deque.extend([self._card_name_to_num[card_name]]*card_info["n_cards"])
            else:
                self._7_plus_deque.extend([self._card_name_to_num[card_name]]*card_info["n_cards"])

        # shuffle deques
        random.shuffle(self._1_6_deque)
        random.shuffle(self._7_plus_deque)
        random.shuffle(self._major_deque)

        self._state_indices["deques"] = {}

        self._state_indices["deques"]["1-6"] = {}
        self._state_indices["deques"]["1-6"]["cards"] = list(range(len(self.state), len(self.state)+len(self._1_6_deque)))
        self.observable_indices.extend(list(range(len(self.state), len(self.state)+len(self._1_6_deque))))
        self.state.extend(self._1_6_deque)
        self._state_indices["deques"]["1-6"]["top_card_idx"] = len(self.state)
        self.state.append(len(self._1_6_deque) - 1)

        self._state_indices["deques"]["7+"] = {}
        self._state_indices["deques"]["7+"]["cards"] = list(range(len(self.state), len(self.state)+len(self._7_plus_deque)))
        self.state.extend(self._7_plus_deque)
        self._state_indices["deques"]["7+"]["top_card_idx"] = len(self.state)
        self.state.append(len(self._7_plus_deque) - 1)

        self._state_indices["deques"]["major"] = {}
        self._state_indices["deques"]["major"]["cards"] = list(range(len(self.state), len(self.state)+len(self._major_deque)))
        self.state.extend(self._major_deque)
        self._state_indices["deques"]["major"]["top_card_idx"] = len(self.state)
        self.state.append(len(self._major_deque) - 1)
        
        
        # 3. marketplace
        ### [1_6_pos1_card, 1_6_pos1_count, ..., 7+_pos5_card, 7+_pos5_count, ..., major_pos2_card, major_pos2_count]
        self._state_indices["marketplace"] = {}
        self._state_indices["marketplace"]["1-6"] = {}
        for pos in range(5):
            self._state_indices["marketplace"]["1-6"][f"pos_{pos}"] = {"card": len(self.state), "count": len(self.state)+1}
            self.observable_indices.extend([len(self.state), len(self.state)+1])
            # -1 means no card, 0 means zero count.
            self.state.extend([-1, 0])
        self._state_indices["marketplace"]["7+"] = {}
        for pos in range(5):
            self._state_indices["marketplace"]["7+"][f"pos_{pos}"] = {"card": len(self.state), "count": len(self.state)+1}
            self.observable_indices.extend([len(self.state), len(self.state)+1])
            # -1 means no card, 0 means zero count.
            self.state.extend([-1, 0])
        self._state_indices["marketplace"]["major"] = {}
        for pos in range(2):
            self._state_indices["marketplace"]["major"][f"pos_{pos}"] = {"card": len(self.state), "count": len(self.state)+1}
            self.observable_indices.extend([len(self.state), len(self.state)+1])
            # -1 means no card, 0 means zero count.
            self.state.extend([-1, 0])


        # 4. current_player_index, current_stage_index
        self._state_indices["current_player_index"] = len(self.state)
        self.observable_indices.append(len(self.state))
        self.state.append(0)
        self._state_indices["current_stage_index"] = len(self.state)
        self.observable_indices.append(len(self.state))
        self.state.append(0)
        self.state = np.array(self.state)
        
        self.fill_alleys()


    def state_dict(self, state: Optional[np.array] = None, state_indices: Optional[dict] = None):
        state_dict = {}
        state = state if state is not None else self.state
        if state_indices is None:
            state_indices = self._state_indices
        for k, v in state_indices.items():
            if k == "card":
                state_dict[k] = self._card_num_to_name[state[v]]
            elif isinstance(v, int):
                state_dict[k] = state[v]
            elif isinstance(v, dict):
                state_dict[k] = self.state_dict(state, state_indices[k])
            elif k == "cards":
                state_dict[k] = [self._card_num_to_name[state[card_idx]] for card_idx in v]
            else:
                breakpoint()

        
        return state_dict
        # state = {
        #     "player_info": {},
        #     "marketplace": {},
        #     "current_player_index": self.state[self._state_indices["player_info"]["current_player_index"]],
        #     "current_stage_index": self.state[self._state_indices["player_info"]["current_stage_index"]],
        # }

        # for player, info in self._state_indices["player_info"]:

        # breakpoint()
        # for l1_name, l1 in self._state_indices.items():
        #     state = {}


    @property
    def current_player_index(self):
        return self.state[self._state_indices["current_player_index"]]
    
    @property
    def current_player(self):
        return self._player_order[self.state[self._state_indices["current_player_index"]]]

    @current_player.setter
    def current_player(self, val):
        self.state[self._state_indices["current_player_index"]] = self._player_order.index(val)
    
    @property
    def current_stage_index(self):
        return self.state[self._state_indices["current_stage_index"]]

    @property
    def current_stage(self):
        return self._stage_order[self.current_stage_index]
        

    @property
    def n_players(self):
        return self._n_players

    @property
    def player_order(self):
        return self._player_order

    @property
    def next_players(self):
        if self.current_player_index == self.n_players-1:
            return self._player_order[:self.current_player_index]
        else:
            return self._player_order[self.current_player_index+1:] + self._player_order[:self.current_player_index]

        # return self._player_order[self.current_player_index:] + self._player_order[:self.current_player_index]

    @property
    def next_players_reversed(self):
        return list(reversed(self.next_players))

    def _earn_income(self, diceroll):
        # order in which income is handled
        for card_type in ["Restaurants", "Secondary Industry", "Primary Industry", "Major Establishment"]:
            # next_players_reversed is the current player, followed by the player before the current player etc.
            for player in [self.current_player] + self.next_players_reversed:
                for card in self._cards_per_activation[diceroll]:
                    if self._card_info[card]["type"] == card_type:
                        for _ in range(self.n_of_card_player_owns(player, card)):
                            self._activate_card(player, card)
                        

    def _diceroll_test(self):
        return random.randint(1,6)

    def _diceroll(self, action: int):
        
        if action == "1 dice":
            n_dice = 1
        elif action == "2 dice":
            n_dice = 2
        else:
            raise ValueError("choose either 1 or 2 dice")
        # if n_dice == "3 dice":
        #     n_dice = 3

        # assert 0 < n_dice <= 2 # to exclude Moon Tower logic

        dicerolls = []
        if n_dice == 2:
            assert self.n_of_card_player_owns(self.current_player, "Train Station") >= 1
            
        for _ in range(n_dice):
            diceroll = random.randint(1,6)
            dicerolls.append(diceroll)
        # Not implemented:
        # if diceroll >= 10:
            # Harbor logic
            # ask player to add 2 or not

        # Not implemented
        # if n_dice == 3:
        # Moon Tower logic
            # ask player which dice to remove

        self._earn_income(sum(dicerolls))
        
        if self.n_of_card_player_owns(self.current_player, "Amusement Park") and n_dice > 1 and dicerolls[0] == dicerolls[1]:
            # Amusement Park logic, can throw dice again.
            return True

        if self.player_coins(self.current_player) == 0:
            # City Hall logic, 0 coins will get you 1 coin.
            self.change_coins(self.current_player, 1)
        
        return False
    
    def invest_in_tech_startup(self):
        # Not implemented
        assert self.player_coins(self.current_player) != 0
        self.state[self._state_indices["player_info"][self.current_player]["tech_startup_investment"]] += 1
    
    def _build(self, action: str):
        """
        action: establishment_to_buy
        """
        if action == "Build nothing" and self.n_of_card_player_owns(self.current_player, "Airport"):
            self.change_coins(self.current_player, 10)
    
        if action != "Build nothing":
            self.buy_card(self.current_player, action)
            if self._card_info[action]["type"] != "Landmarks":
                self.remove_card_from_marketplace(action)
            self.add_card(self.current_player, action)


    def remove_card_from_marketplace(self, card):
        for alley in self._state_indices["marketplace"].values():
            for stand in alley.values():
                if self._card_num_to_name[self.state[stand["card"]]] == card:
                    assert self.state[stand["count"]] >= 1, f"card {card} found in marketplace but has 0 count"
                    self.state[stand["count"]] -= 1
                    if self.state[stand["count"]] == 0:
                        self.state[stand["card"]] = -1
                        self.fill_alleys()
                    return
        assert False, f"Could not find card {card} in marketplace"

    def fill_alleys(self):
        for alley_name, alley in self._state_indices["marketplace"].items():
            # if no cards left, continue to next alley
            if self.state[self._state_indices["deques"][alley_name]["top_card_idx"]] == -1:
                continue
            
            while True:
                if not self.alley_needs_card(alley_name) or self.state[self._state_indices["deques"][alley_name]["top_card_idx"]] == -1:
                    break
                self.add_card_to_alley(alley_name)
                

    def add_card_to_alley(self, alley_name):
        card = self.take_from_deque(alley_name)
        for stand in self._state_indices["marketplace"][alley_name].values():
            if card == self.state[stand["card"]]:
                self.state[stand["count"]] +=1
                return
            elif self.state[stand["card"]] == -1:
                self.state[stand["card"]] = card
                self.state[stand["count"]] +=1
                return
        assert False, f"Could not add card to a stand in alley {alley_name}"


    def alley_needs_card(self, alley_name):
        alley = self._state_indices["marketplace"][alley_name]
        for stand in alley.values():
            if self.state[stand["count"]] == 0:
                return True
        return False

    def take_from_deque(self, deque):
        assert self.state[self._state_indices["deques"][deque]["top_card_idx"]] >= 0

        top_card_idx_in_deque = self.state[self._state_indices["deques"][deque]["top_card_idx"]]
        top_card_idx_in_state = self._state_indices["deques"][deque]["cards"][top_card_idx_in_deque]
        self.state[self._state_indices["deques"][deque]["top_card_idx"]] -= 1
        return self.state[top_card_idx_in_state]

    def buy_card(self, player, card):
        assert self.player_coins(player) >= self._card_info[card]["cost"]
        self.change_coins(player, -self._card_info[card]["cost"])
    
    def add_card(self, player, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self.n_of_card_player_owns(player, card) == 0
        self.state[self._state_indices["player_info"][player]["cards"][card]] += 1

    def remove_card(self, player, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self.n_of_card_player_owns(player, card) == 1
        
        assert self.n_of_card_player_owns(player, card) >= 1
        self.state[self._state_indices["player_info"][player]["cards"][card]] -= 1

    def change_coins(self, player, coins):
        self.state[self._state_indices["player_info"][player]["coins"]] += coins
    

    def player_coins(self, player):
        return self.state[self._state_indices["player_info"][player]["coins"]]


    def n_of_card_player_owns(self, player, card):
        return self.state[self._state_indices["player_info"][player]["cards"][card]]


    def n_of_landmarks_player_owns(self, player):
        n_landmarks = 0
        for landmark in self._landmark_cards_ascending_in_price:
            n_landmarks += self.n_of_card_player_owns(player, landmark)
        return n_landmarks
    

    def player_icon_count(self, player, icon):
        n_cards_with_icon = 0
        for card in self._cards_per_icon[icon]:
            n_cards_with_icon += self.n_of_card_player_owns(player, card)
        return n_cards_with_icon
    

    def player_tech_startup_investment(self, player):
        # Not implemented
        return self.state[self._state_indices["player_info"][player]["tech_startup_investment"]]

    def _advance_stage(self):
        self.state[self._state_indices["current_stage_index"]] += 1
        if self.current_stage_index > self._n_stages - 1:
            self.state[self._state_indices["current_stage_index"]] = 0
            self._advance_player()

    def _advance_player(self):
        self.state[self._state_indices["current_player_index"]] += 1
        if self.current_player_index > self._n_players - 1:
            self.state[self._state_indices["current_player_index"]] = 0

    def winner(self):
        for player in [self.current_player] + self.next_players:
            if self.n_of_landmarks_player_owns(player) == self._max_landmarks:
                return True
        return False

    def step(self, action: str) -> bool:
        if self.current_stage == "build":
            self._build(action)
        elif self.current_stage == "diceroll":
            roll_again = self._diceroll(action)

        if self.winner():
            return True
        
        if self.current_stage == "diceroll" and roll_again:
            return False

        self._advance_stage()

        return False

    def _payment(self, payee, reciever, amount):
        payee_money = self.player_coins(payee)
        amount_paid = amount if payee_money - amount >= 0 else payee_money
        self.change_coins(payee, -amount_paid)
        self.change_coins(reciever, amount_paid)
  
    def _activate_card(self, player, card):
        if card == "Wheat Field" or card == "Ranch" or card == "Flower Orchard" or card == "Forest":
            self.change_coins(player, 1)

        elif card == "Mackerel Boat" and self.n_of_card_player_owns(player, "Harbor"):
            self.change_coins(player, 3)
    
        elif card == "Apple Orchard":
            self.change_coins(player, 3)

        elif card == "Tuna Boat" and self.n_of_card_player_owns(player, "Harbor"):
            diceroll = random.randint(1,6) + random.randint(1,6)
            self.change_coins(player, diceroll)
        
        elif card == "General Store" and player == self.current_player and self.n_of_landmarks_player_owns(player) <= 1:
            self.change_coins(player, 2) if self.n_of_card_player_owns(player, "Shopping Mall") == 0 else 3
        
        elif card == "Bakery" and player == self.current_player:
            self.change_coins(player, 1) if self.n_of_card_player_owns(player, "Shopping Mall") == 0 else 2

        elif card == "Demolition Company" and player == self.current_player and self.n_of_landmarks_player_owns(player) >= 1:
            # Not implemented, promt player which landmark to destroy.
            destroyed_landmark = None
            for landmark in self._landmark_cards_ascending_in_price:
                if self.n_of_card_player_owns(player, landmark) != 0:
                    self.remove_card(player, landmark)
                    self.change_coins(player, 8)
                    destroyed_landmark = landmark
                    break
            assert destroyed_landmark is not None


        elif card == "Flower Shop" and player == self.current_player:
            self.change_coins(
                player,
                (
                    self.n_of_card_player_owns(player, "Flower Orchard")
                    if not self.n_of_card_player_owns(self.current_player, "Shopping Mall")
                    else self.n_of_card_player_owns(player, "Flower Orchard")*2
                )
            )

        elif card == "Cheese Factory" and player == self.current_player:
            self.change_coins(player, self.player_icon_count(self.current_player, "Cow") * 3)

        elif card == "Furniture Factory" and player == self.current_player:
            self.change_coins(player, self.player_icon_count(self.current_player, "Gear") * 3)

        # Not implemented
        # elif card == "Moving Company" and player == self.current_player:
            # give non major (icon) to another player and get 4 coins from the bank

        elif card == "Soda Bottling Plant" and player == self.current_player:
            for player in [self.current_player] + self.next_players:
                self.change_coins(self.current_player, self.player_icon_count(player, "Cup"))
        
        elif card == "Fruit and Vegetable Market" and player == self.current_player:
            self.change_coins(player, self.player_icon_count(player, "Grain") * 2)

        elif card == "Sushi Bar" and player != self.current_player and self.n_of_card_player_owns(player, "Harbor"):
            self._payment(self.current_player, player, 3 if not self.n_of_card_player_owns(player, "Shopping Mall") else 4)

        elif card == "Café" or card == "Pizza Joint" and player != self.current_player:
            self._payment(self.current_player, player, 1 if not self.n_of_card_player_owns(player, "Shopping Mall") else 2)
        
        elif card == "French Restaurant" and player != self.current_player and self.n_of_landmarks_player_owns(self.current_player) >= 2:
            # if current player has 2 or more landmarks, transfer 5 coins to the player owning the
            self._payment(self.current_player, player, 5 if not self.n_of_card_player_owns(player, "Shopping Mall") else 6)

        elif card == "Family Restaurant" and player != self.current_player:
            self._payment(self.current_player, player, 2 if not self.n_of_card_player_owns(player, "Shopping Mall") else 3)

        elif card == "Member's Only Club" and player != self.current_player and self.n_of_landmarks_player_owns(self.current_player) >= 3:
            # if current player has 3 or more landmarks, transfer all their coins to the player owning the
            # member's only club.
            self._payment(self.current_player, player, self.player_coins(self.current_player))

        elif card == "Stadium" and player == self.current_player:
            for player in [self.current_player] + self.next_players:
                if player != self.current_player:
                    self._payment(player, self.current_player, 2)
        
        elif card == "Publisher" and player == self.current_player:
            for player in [self.current_player] + self.next_players:
                if player != self.current_player:
                    coins_to_pay = self.player_icon_count(player, "Cup") + self.player_icon_count(player, "Box")
                    self._payment(player, self.current_player, coins_to_pay)

        elif card == "Tax Office" and player == self.current_player:
            for player in [self.current_player] + self.next_players:
                if player != self.current_player and self.player_coins(player) >= 10:
                    coins_to_pay = int(self.player_coins(player) / 2)
                    self._payment(player, self.current_player, coins_to_pay)

        # Not implemented
        # elif card == "Business Center" and player == self.current_player:
            # trade non major (icon) card with anothe player

        elif card == "Tech Startup" and player == self.current_player and self.player_tech_startup_investment(self.current_player) > 0:
            for player in [self.current_player] + self.next_players:
                if player != self.current_player:
                    self._payment(player, self.current_player, self.player_tech_startup_investment(self.current_player))

class GymMachiKoro(gym.Env):
    def __init__(self, n_players: int):
        self._env = MachiKoro(n_players=n_players)

        actions = ["1 dice", "2 dice"]
        [actions.append(action) for action in self.card_info.keys()]
        actions.append("Build nothing")
        self._action_idx_to_str = {idx: name for idx, name in enumerate(actions)}
        self._action_str_to_idx = {name: idx for idx, name in enumerate(actions)}

        self.action_space = gym.spaces.Discrete(len(self._action_idx_to_str))
        self._action_mask = np.zeros(self.action_space.n)

        # self._player_to_idx = {player: idx for idx, player in enumerate(self._env._player_order)}

        self._establishments_to_idx = {
            "1-6": {"None": 0},
            "7+": {"None": 0},
            "major": {"None": 0},
        }

        for card_name, info in self.card_info.items():
            if info["type"] == "Landmarks":
                continue
            elif info["type"] == "Major Establishment":
                self._establishments_to_idx["major"][card_name] = len(self._establishments_to_idx["major"])
            elif info["activation"][0] <= 6:
                self._establishments_to_idx["1-6"][card_name] = len(self._establishments_to_idx["1-6"])
            elif info["activation"][0] >= 7:
                self._establishments_to_idx["7+"][card_name] = len(self._establishments_to_idx["7+"])

        obs_space = OrderedDict()
        self._obs = OrderedDict()
        self._obs_indices = {}
        self._obs_len = 0
        # assert False, "Hier verder!!! Create obs space here, and overwrite in self.observation so to re-use memory." 
        # "In self.observation remove things like self._obs[...] = np.zeros and change to self._obs[...] *= 0"
        def add_player_obs_spaces(player_obs_name):
            for card in self.card_info.keys():
                if self.card_info[card]["type"] == "Landmarks":
                    space_len = 2
                else:
                    max_cards = self.card_info[card]["n_cards"]
                    space_len = 1 + max_cards + (card == "Wheat Field") + (card == "Bakery") # + 1 so that the 1st entry is reserved for 0 cards

                self._obs_indices[f"{player_obs_name}-{card}"] = np.arange(space_len) + self._obs_len
                self._obs_len += space_len

            self._obs_indices[f"{player_obs_name}-coins"] = np.arange(1) + self._obs_len
            self._obs_len += 1
        
        add_player_obs_spaces("current_player")


        for i in range(self._env._n_players-1):
            add_player_obs_spaces(f"player_current_p_{i+1}")

        for alley_name, alley in self._env._state_indices["marketplace"].items():
            for pos, stand in alley.items():
                # encoding which card is in the deck
                space_len = len(self._establishments_to_idx[alley_name])
                self._obs_indices[f"marketplace-{alley_name}-{pos}-card"] = np.arange(space_len) + self._obs_len
                self._obs_len += space_len
                # encoding how many cards are in the deck
                space_len = 5+1 if alley_name == "major" else 6+1 # major have max 5 cards and others have max 6 cards. +1 is to account for and empty deque
                self._obs_indices[f"marketplace-{alley_name}-{pos}-amount"] = np.arange(space_len) + self._obs_len
                self._obs_len += space_len
        self._obs_indices["current_player_index"] = np.arange(self._env._n_players) + self._obs_len
        self._obs_len += self._env._n_players
        self._obs_indices["current_stage_index"] = np.arange(self._env._n_stages) + self._obs_len
        self._obs_len += self._env._n_stages

        self._obs_indices["action_mask"] = np.arange(self.action_space.n) + self._obs_len
        self._obs_len += len(self.action_mask)

        self.observation_space = gym.spaces.MultiBinary(self._obs_len)
        del self._obs_len

        self._obs = np.zeros(self.observation_space.n)

    @property
    def card_info(self):
        return self._env._card_info

    @property
    def player_info(self):
        return self._env.player_info

    @property
    def n_players(self):
        return self._env.n_players

    @property
    def player_order(self):
        return self._env.player_order
    
    @property
    def current_player(self):
        return self._env.current_player
    
    @current_player.setter
    def current_player(self, val):
        self._env.current_player = val

    @property
    def current_player_index(self):
        return self._env.current_player_index
    
    @property
    def current_stage(self):
        return self._env.current_stage

    @property
    def current_stage_index(self):
        return self._env.current_stage_index

    def diceroll(self, n_dice):
        return self._env._diceroll(n_dice)

    def _get_info(self, action_mask: Optional[np.ndarray] = None):
        """
        state and action_mask don't need a deepcopy because they have already been deepcopied
        state in the return statement of self.get_state()
        action_mask when passed as an argument, is taken from the observation, which is deepcopied 
        already.
        """
        return {
            "state": self.get_state(),
            "current_player_index": copy.deepcopy(self.current_player_index),
            "is_stochastic": self._env.current_stage == "diceroll",
            "action_mask": action_mask if action_mask is not None else copy.deepcopy(self.action_mask)
        }

    def reset(self, state: dict | None = None):
        if state is not None:
            self.set_state(state)
        else:  
            self._env.reset()
        obs = self.observation()
        # REMOVE
        self.count = 0
        self._info = self._get_info(action_mask=obs[self._obs_indices["action_mask"]])
        #
        return self.observation(), self._get_info(action_mask=obs[self._obs_indices["action_mask"]])

    def get_state(self):
        # REMOVE
        return self._env.state
        #
        return copy.deepcopy(self._env.state)
    
    def state_dict(self, state: Optional[np.array] = None):
        return copy.deepcopy(self._env.state_dict(state))

    def set_state(self, state):
        # REMOVE
        return
        self._env.state = copy.deepcopy(state)

    def step(self, action, state: dict | None = None):
        # REMOVE
        self.count+=1
        return self._obs, self.count==100, self.count==100, False, self._info
        #
        if state:
            self.set_state(state)
        
        winner = self._env.step(self._action_idx_to_str[action])
        obs = self.observation()
        info = self._get_info(action_mask=obs[self._obs_indices["action_mask"]])
        return obs, int(winner), winner, False, info

    def _diceroll_action_mask(self):
        self._action_mask*=0
        self._action_mask[self._action_str_to_idx["1 dice"]] = 1
        if self._env.n_of_card_player_owns(self._env.current_player, "Train Station"):
            self._action_mask[self._action_str_to_idx["2 dice"]] = 1
        return self._action_mask


    def _build_action_mask(self):
        self._action_mask*=0
        cards_in_marketplace = []

        for alley in self._env._state_indices["marketplace"].values():
            for stand in alley.values():
                if self._env.state[stand["count"]] > 0:
                    card_str = self._env._card_num_to_name[self._env.state[stand["card"]]]
                    cards_in_marketplace.append(card_str)

        for action in range(self.action_space.n):
            action_str = self._action_idx_to_str[action]
            if action_str == "1 dice" or action_str == "2 dice":
                continue
            elif action_str == "Build nothing" or self._env.player_coins(self.current_player) >= self.card_info[action_str]["cost"] and action_str in cards_in_marketplace:
                self._action_mask[action] = 1
            elif self.card_info[action_str]["type"] == "Landmarks" and self._env.player_coins(self.current_player) >= self.card_info[action_str]["cost"] and not self._env.n_of_card_player_owns(self.current_player, action_str):
                self._action_mask[action] = 1

        # action_actually_allowed = True
        # for action, allowed in enumerate(self._action_mask):
        #     if allowed:
        #         a_str = self._action_idx_to_str[action]
        #         if a_str not in ["1 dice", "2 dice", "Build nothing"]:
        #             print(a_str)
        #             for alley_name, alley in self._env.state_dict()["marketplace"].items():
        #                 for pos_name, pos in alley.items():
        #                     if pos["card"] == a_str:
        #                         action_actually_allowed = True
        # breakpoint()
        # if not action_actually_allowed:
        #     breakpoint()
        return self._action_mask

    @property
    def action_mask(self):
        if self._env.current_stage == "build":
            return self._build_action_mask()
        elif self._env.current_stage == "diceroll":
            return self._diceroll_action_mask()

    def sample_action(self):
        action_mask = self.action_mask()
        prob_dist = action_mask/sum(action_mask)
        return np.random.choice(range(self.action_space.n), p=prob_dist)
    
    def _add_player_obs(self, player, player_obs_name):
        for card, amount_idx in self._env._state_indices["player_info"][player]["cards"].items():
            amount = self._env.state[amount_idx]
            self._obs[self._obs_indices[f"{player_obs_name}-{card}"]][amount] = 1
            
        self._obs[self._obs_indices[f"{player_obs_name}-coins"]] = self._env.state[
            self._env._state_indices["player_info"][player]["coins"]
        ]

    def next_players(self):
        return self._env.next_players
 
    def observation(self):
        self._obs *= 0
        # return copy.deepcopy(self.obs)
        # assert False, "Hier verder! Construct observation directly from state using self._env.observable_indices"
        self._add_player_obs(self.current_player, "current_player")

        for i, player in enumerate(self.next_players()):
            self._add_player_obs(player, f"player_current_p_{i+1}")
        
        for alley_name, alley in self._env._state_indices["marketplace"].items():
            for pos, stand in alley.items():
                # encoding which card is in the deck
                if self._env.state[stand["count"]] != 0:
                    card = self._env._card_num_to_name[self._env.state[stand["card"]]]
                    self._obs[self._obs_indices[f"marketplace-{alley_name}-{pos}-card"][self._establishments_to_idx[alley_name][card]]] = 1
                else:
                    self._obs[self._obs_indices[f"marketplace-{alley_name}-{pos}-card"][0]] = 1

                # encoding how many cards are in the deck
                self._obs[self._obs_indices[f"marketplace-{alley_name}-{pos}-amount"][self._env.state[stand["count"]]]] = 1
        self._obs[self._obs_indices["current_player_index"][self._env.current_player_index]] = 1
        self._obs[self._obs_indices["current_stage_index"][self._env.current_player_index]] = 1

        self._obs[self._obs_indices["action_mask"]] = self.action_mask
        return copy.deepcopy(self._obs)
    
    def flattened_obs(self,):
        return gym.spaces.flatten(self.observation_space, self.observation())
    
    # def _player_info_from_obs(self, player, player_obs_name):
    #         info = PlayerInfo(self.card_info)
            
    #         for card, amount in self._env._player_info[player].cards.items():
    #             if self.card_info[card]["type"] == "Landmarks":
    #                 onehot = np.array([not amount, amount]).astype(int)
    #             else:
    #                 max_cards = self.card_info[card]["n_cards"]
    #                 onehot = np.zeros(max_cards + 1 + (card == "Wheat Field") + (card == "Bakery"))
    #                 onehot[amount] = 1
    #             self._obs[f"{player_obs_name}-{card}"] = onehot
    #         self._obs[f"{player_obs_name}-coins"] = self._env._player_info[self.current_player].coins

    def getObservation(self, state):
        self.set_state(state)
        return self.observation()

    # def getStringRepresentation(self, state):
    #     self.set_state(state)
    #     return copy.deepcopy("".join(map(str,gym.spaces.flatten(self.observation_space, self.obs).astype(int))))

    def getActionSize(self):
        return self.action_space.n

    def getIsTerminal(self, state):
        self.set_state(state)
        return self._env.winner()
    
    def getValidMoves(self, state):
        self.set_state(state)
        return self.action_mask