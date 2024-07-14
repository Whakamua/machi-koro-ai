import gym
import yaml
from collections import deque
import random
import copy
import numpy as np
from collections import OrderedDict
from typing import Optional

class MachiKoro2:
    def __init__(
            self,
            n_players: int,
            card_info_path: Optional[str] = None,
            print_info: bool = False,
        ):
        self._print_info = print_info
        self._n_players = n_players
        self._winning_player_index = None

        if card_info_path is None:
            card_info_path = "card_info_machi_koro_2.yaml"
        with open(card_info_path) as f:
            self._card_info = yaml.load(f, Loader=yaml.loader.SafeLoader)

        self._cards_per_activation = {i: [] for i in range(1, 13)}
        for card, info in self._card_info.items():
            if info["type"] != "Landmark":
                for activation in info["activation"]:
                    self._cards_per_activation[activation].append(card)

        self._landmarks = [card for card, info in self._card_info.items() if info["type"] == "Landmarks"]
        self._max_landmarks = 3
        self._init_establishments = {
            "1-6": [],
            "7+": [],
            "landmark": [],
        }

        self._cards_per_icon = {}
        for card, info in self._card_info.items():
            if info["icon"] not in self._cards_per_icon:
                self._cards_per_icon[info["icon"]] = []
            self._cards_per_icon[info["icon"]].append(card)

        for card_name, info in self._card_info.items():
            if info["type"] == "Landmark":
                self._init_establishments["landmark"].append(card_name)
            elif max(info["activation"]) <= 6:
                self._init_establishments["1-6"].extend([card_name]*info["n_cards"])
            else:
                self._init_establishments["7+"].extend([card_name]*info["n_cards"])

        self._stage_order = ["diceroll", "build"]
        self._n_stages = len(self._stage_order)
        self._player_order = [f"player {i}" for i in range(n_players)]

        self.reset()

    def reset(self):
        # construct state
        self.state = []
        self._state_indices = {}
        self._state_indices["player_info"] = {}
        self._state_values = {}
        self._state_values["player_info"] = {}

        # 1. player_info construct vectors and indices for each player its own state
        ### [n_card_1_p1, n_card_2_p1, ..., n_card_N_pN, n_coins_pN, startup_inv_pN]
        for player in self._player_order:
            self._state_indices["player_info"][player] = {}
            self._state_values["player_info"][player] = {}

            # buildings in city
            self._state_indices["player_info"][player]["cards"] = {}
            self._state_values["player_info"][player]["cards"] = {}

            for card_name, card_info in self._card_info.items():
                self._state_indices["player_info"][player]["cards"][card_name] = len(self.state)
                max_cards_player_can_have = card_info["n_cards"]
                self._state_values["player_info"][player]["cards"][card_name] = range(max_cards_player_can_have + 1) # +1 to make it an inclusive range
                self.state.append(0)

            # coins
            self._state_indices["player_info"][player]["coins"] = len(self.state)
            self._state_values["player_info"][player]["coins"] = None
            self.state.append(5)

        # 2. state of the deques that lie on the table
        ### [1_6_deque_card_1, 1_6_deque_card_2, ..., _major_deque_card_N]

        # the cards will be enumerated, to keep things overseeable there's a name to index mapping.
        self._card_name_to_num = {name: i for i, (name, info) in enumerate(list(self._card_info.items()))}
        self._card_name_to_num["None"] = -1
        self._card_num_to_name = {i: name for name, i in self._card_name_to_num.items()}

        self._1_6_deque = []
        self._7_plus_deque = []
        self._landmark_deque = []

        # create deques
        for card_name, card_info in self._card_info.items():
            if card_info["type"] == "Landmark":
                self._landmark_deque.append(self._card_name_to_num[card_name])
            elif max(card_info["activation"]) <= 6:
                self._1_6_deque.extend([self._card_name_to_num[card_name]]*card_info["n_cards"])
            else:
                self._7_plus_deque.extend([self._card_name_to_num[card_name]]*card_info["n_cards"])

        self._state_indices["deques"] = {}

        self._state_indices["deques"]["1-6"] = {}
        self._state_indices["deques"]["1-6"]["cards"] = list(range(len(self.state), len(self.state)+len(self._1_6_deque)))
        self.state.extend(self._1_6_deque)

        self._state_indices["deques"]["7+"] = {}
        self._state_indices["deques"]["7+"]["cards"] = list(range(len(self.state), len(self.state)+len(self._7_plus_deque)))
        self.state.extend(self._7_plus_deque)

        self._state_indices["deques"]["landmark"] = {}
        self._state_indices["deques"]["landmark"]["cards"] = list(range(len(self.state), len(self.state)+len(self._landmark_deque)))
        self.state.extend(self._landmark_deque)

        # 3. marketplace
        ### [1_6_pos1_card, 1_6_pos1_count, ..., 7+_pos5_card, 7+_pos5_count, ..., landmark_pos2_card, landmark_pos2_count]
        self._state_indices["marketplace"] = {}
        self._state_values["marketplace"] = {}

        for alley_name, establishments in self._init_establishments.items():
            self._state_indices["marketplace"][alley_name] = {}
            self._state_values["marketplace"][alley_name] = {}
            unique_card_strs_in_alley = np.unique(np.array(self._init_establishments[alley_name]))
            unique_card_nums_in_alley = [self._card_name_to_num[card_name] for card_name in unique_card_strs_in_alley]
            max_cards_in_stand = max([self._card_info[card_name]["n_cards"] for card_name in unique_card_strs_in_alley])

            for pos in range(5):
                # n_different_cards_in_alley = len(unique_cards_in_alley)
                self._state_indices["marketplace"][alley_name][f"pos_{pos}"] = {
                    "card": len(self.state),
                    "count": len(self.state)+1
                }
                self.state.extend([-1, 0])
                self._state_values["marketplace"][alley_name][f"pos_{pos}"] = {
                    "card": unique_card_nums_in_alley + [-1],
                    "count": range(0, max_cards_in_stand+1),
                }

        # 4. current_player_index, current_stage_index
        self._state_indices["current_player_index"] = len(self.state)
        self._state_values["current_player_index"] = range(0, self.n_players)
        self.state.append(0)
        self._state_indices["current_stage_index"] = len(self.state)
        self._state_values["current_stage_index"] = range(0, self._n_stages)
        self.state.append(1)
        self._state_indices["another_turn"] = len(self.state)
        self._state_values["another_turn"] = range(0, 2)
        self.state.append(0)
        self._state_indices["build_rounds_left"] = len(self.state)
        self._state_values["build_rounds_left"] = range(0, 4)
        self.state.append(3)
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
                raise ValueError(f"key, value: {k}, {v} cannot be parsed")
        return state_dict

    def state_dict_to_array(self, state_dict, state_indices = None, state_array = None):
        if state_array is None:
            state_array = np.zeros_like(self.state)
        if state_indices is None:
            state_indices = self._state_indices

        for k, v in state_indices.items():
            if k == "card":
                state_array[v] = self._card_name_to_num[state_dict[k]]
            elif isinstance(v, int):
                state_array[v] = state_dict[k]
            elif isinstance(v, dict):
                state_array = self.state_dict_to_array(state_dict[k], state_indices[k], state_array)
            elif k == "cards":
                state_array[v] = [self._card_name_to_num[card_name] for card_name in state_dict[k]]
            else:
                raise ValueError(f"key, value: {k}, {v} cannot be parsed")
        return state_array

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

    @property
    def next_players_reversed(self):
        return list(reversed(self.next_players))

    def _earn_income(self, diceroll):
        if isinstance(diceroll, list):
            n_dice = len(diceroll)
            doubles = len(diceroll) == 2 and diceroll[0] == diceroll[1]
            diceroll = sum(diceroll)
        else:
            n_dice = 1
            doubles = False

        # Charterhouse logic
        if self.is_card_owned("Charterhouse"):
            got_any_income = False
            current_player_coins = copy.deepcopy(self.player_coins(self.current_player))
        # order in which income is handled
        for card_type in ["Restaurant (Red)", "Primary Industry (Blue)", "Secondary Industry (Green)", "Major (Purple)", "Landmark"]:
            # next_players_reversed is the current player, followed by the player before the current player etc.
            for player in [self.current_player] + self.next_players_reversed:
                for card in self._cards_per_activation[diceroll]:
                    if self._card_info[card]["type"] == card_type:
                        for _ in range(self.n_of_card_player_owns(player, card)):
                            if self._print_info:
                                print(f"{card}: ", end="")
                            self._activate_card(player, card)

                            # Charterhouse logic
                            if self.is_card_owned("Charterhouse") and not got_any_income:
                                new_current_player_coins = copy.deepcopy(self.player_coins(self.current_player))
                                if new_current_player_coins > current_player_coins:
                                    got_any_income = True
        # Charterhouse logic
        if self.is_card_owned("Charterhouse") and not got_any_income and n_dice == 2:
            if self._print_info:
                print(f"{self.current_player} did not get any income, gets 3 coins from the Charterhouse")
            self.change_coins(self.current_player, 3)
        
        if self.is_card_owned("Tech Startup") and diceroll == 12:
            if self._print_info:
                print(f"{self.current_player} threw 12 and gets 8 coins from the Tech Startup")
            self.change_coins(self.current_player, 8)
        
        if self.is_card_owned("Temple") and doubles:
            if self._print_info:
                print(f"{self.current_player} threw doubles and gets 2 coins from every other player")
            for other_player in self.next_players:
                self._payment(other_player, self.current_player, 2)

        if self.is_card_owned("Amusement Park") and doubles:
            # Amusement Park logic, can take another turn.
            if self._print_info:
                print(f"{self.current_player} can take another turn after this one due to throwing doubles and the Amusement park being in the game.")
            self.another_turn = True
        
        # if self.is_card_owned("Moving Company")  and n_dice > 1 and dicerolls[0] == dicerolls[1]:
        #     # not implemented, give card to player on your right

        if self.player_coins(self.current_player) == 0:
            if self._print_info:
                print(f"{self.current_player} is broke and gets 1 coin.")
            self.change_coins(self.current_player, 1)

    def _diceroll(self, action: int):  
        if action == "1 dice":
            n_dice = 1
        elif action == "2 dice":
            n_dice = 2
        else:
            raise ValueError("choose either 1 or 2 dice")

        dicerolls = []

        for _ in range(n_dice):
            diceroll = random.randint(1,6)
            dicerolls.append(diceroll)
        if self._print_info:
            if len(dicerolls) == 1:
                print(f"{self.current_player} threw {dicerolls[0]} with 1 dice.")
            else:
                print(f"{self.current_player} threw {sum(dicerolls)} ({dicerolls[0]} and {dicerolls[1]}) with 2 dice.")
        self._earn_income(dicerolls)

        

    
    def is_card_owned(self, card):
        for player in self.player_order:
            if self.n_of_card_player_owns(player, card) >= 1:
                return True
        return False
    

    @property
    def winning_player_index(self):
        return self._winning_player_index


    @property
    def another_turn(self):
        return bool(self.state[self._state_indices["another_turn"]])


    @another_turn.setter
    def another_turn(self, another_turn):
        self.state[self._state_indices["another_turn"]] = int(another_turn)


    def _build(self, action: str):
        """
        action: establishment_to_buy
        """

        if action == "Build nothing":
            if self.is_card_owned("Airport"):
                if self._print_info:
                    print(f"{self.current_player} build nothing and got 5 coins from the Airport.")
                self.change_coins(self.current_player, 5)
            else:
                if self._print_info:
                    print(f"{self.current_player} build nothing.")
            return
    
        self.buy_card(self.current_player, action)
        self.remove_card_from_marketplace(action)
        self.add_card(self.current_player, action)

        # one-time side effects of buyingf a card:
        if action == "Museum":
            if self._print_info:
                print(f"By buying the Museum, {self.current_player} gets 3 coins from each other player for each landmark they own.")
            for player in self.next_players:
                n_landmarks = self.player_icon_count(player, "Landmark")
                self._payment(player, self.current_player, 3*n_landmarks)
        elif action == "French Restaurant":
            if self._print_info:
                print(f"By buying the French Restaurant, {self.current_player} gets 2 coins from each other player.")
            for player in self.next_players:
                self._payment(player, self.current_player, 2)
        elif action == "Exhibit Hall":
            if self._print_info:
                print(f"By buying the Exhibit Hall, {self.current_player} gets half of the coins from each other player.")
            for player in self.next_players:
                coins = self.player_coins(player)
                if coins > 10:
                    self._payment(player, self.current_player, coins // 2)
        elif action == "Publisher":
            if self._print_info:
                print(f"By buying the Publisher, {self.current_player} gets 1 coin from each other player for each `Bread` icon card they own.")
            for player in self.next_players:
                n_breads = self.player_icon_count(player, "Bread")
                self._payment(player, self.current_player, n_breads)
        elif action == "TV Station":
            if self._print_info:
                print(f"By buying the TV Station, {self.current_player} gets 1 coin from each other player for each `Cup` icon card they own.")
            for player in self.next_players:
                n_cups = self.player_icon_count(player, "Cup")
                self._payment(player, self.current_player, n_cups)
        elif action == "Radio Tower":
            if self._print_info:
                print(f"By buying the Radio Tower, {self.current_player} gets another turn after this one.")
            self.another_turn = True
        elif action == "Park":
            if self._print_info:
                print(f"By buying the Park, all coins get evenly distributed.")
            player_coins = [self.player_coins(player) for player in self.player_order]
            even_coins = np.ceil(sum(player_coins)/len(player_coins))
            for player in self.player_order:
                self.set_player_coins(player, even_coins)

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
            if (self.state[self._state_indices["deques"][alley_name]["cards"]] == -1).all():
                continue
            
            while True:
                if not self.alley_needs_card(alley_name) or (self.state[self._state_indices["deques"][alley_name]["cards"]] == -1).all():
                    break
                self.add_card_to_alley(alley_name)

    def add_card_to_alley(self, alley_name):
        card = self.take_from_deque(alley_name)
        empty_stand = None
        for stand in self._state_indices["marketplace"][alley_name].values():
            if card == self.state[stand["card"]]:
                self.state[stand["count"]] +=1
                if self._print_info:
                    print(f"{self._card_num_to_name[card]} was added to alley {alley_name}.")
                return
            elif self.state[stand["card"]] == -1:
                empty_stand = stand

        self.state[empty_stand["card"]] = card
        
        self.state[empty_stand["count"]] +=1
        if self._print_info:
            print(f"{self._card_num_to_name[card]} was added to alley {alley_name}.")


    def alley_needs_card(self, alley_name):
        alley = self._state_indices["marketplace"][alley_name]
        for stand in alley.values():
            if self.state[stand["count"]] == 0:
                return True
        return False


    def take_from_deque(self, deque):
        available_card_indices = np.where(self.state[self._state_indices["deques"][deque]["cards"]] != -1)[0]
        chosen_card_index = np.random.choice(available_card_indices)
        chosen_card = self.state[self._state_indices["deques"][deque]["cards"][chosen_card_index]]
        self.state[self._state_indices["deques"][deque]["cards"][chosen_card_index]] = -1

        return chosen_card


    def card_cost(self, player, card):
        if self._card_info[card]["type"] == "Landmark":
            n_landmarks = self.player_icon_count(player, "Landmark")
            cost = self._card_info[card]["cost"][n_landmarks]
        else:
            cost = self._card_info[card]["cost"]
        if self.is_card_owned("Observatory") and card == "Launch Pad":
            cost -= 5
        if self.is_card_owned("Loan Office"):
            cost -= 2
        return cost


    def buy_card(self, player, card):
        cost = self.card_cost(player, card)
        assert self.player_coins(player) >= cost
        self.change_coins(player, -cost)
        if self._print_info:
            print(f"{player} bought {card} for {cost} coins and now has {self.player_coins(player)} coins left.")


    def add_card(self, player, card):
        self.state[self._state_indices["player_info"][player]["cards"][card]] += 1


    def remove_card(self, player, card):
        assert self.n_of_card_player_owns(player, card) >= 1
        self.state[self._state_indices["player_info"][player]["cards"][card]] -= 1

    def change_coins(self, player, coins):
        self.state[self._state_indices["player_info"][player]["coins"]] += coins
    

    def player_coins(self, player):
        return self.state[self._state_indices["player_info"][player]["coins"]]

    def set_player_coins(self, player, coins):
        self.state[self._state_indices["player_info"][player]["coins"]] = coins

    def n_of_card_player_owns(self, player, card):
        return self.state[self._state_indices["player_info"][player]["cards"][card]]


    def player_icon_count(self, player, icon):
        n_cards_with_icon = 0
        for card in self._cards_per_icon[icon]:
            n_cards_with_icon += self.n_of_card_player_owns(player, card)
        return n_cards_with_icon


    @property
    def build_rounds_left(self):
        return self.state[self._state_indices["build_rounds_left"]]
    

    @build_rounds_left.setter
    def build_rounds_left(self, value):
        self.state[self._state_indices["build_rounds_left"]] = value


    def _advance_stage(self):
        if self.build_rounds_left > 0:
            self._advance_player()
            return
        self.state[self._state_indices["current_stage_index"]] += 1
        if self.current_stage_index > self._n_stages - 1:
            self.state[self._state_indices["current_stage_index"]] = 0
            if self.another_turn:
                self.another_turn = False
            else:
                self._advance_player()

    def _advance_player(self):
        self.state[self._state_indices["current_player_index"]] += 1
        if self.current_player_index > self._n_players - 1:
            self.state[self._state_indices["current_player_index"]] = 0
            if self.build_rounds_left > 0:
                self.build_rounds_left -= 1
                if self.build_rounds_left == 0:
                    self.state[self._state_indices["current_stage_index"]] = 0

    def winner(self):
        if self.player_icon_count(self.current_player, "Landmark") == self._max_landmarks:
            self._winning_player_index = copy.deepcopy(self.current_player_index)
            return True
        elif self.n_of_card_player_owns(self.current_player, "Launch Pad") == 1:
            self._winning_player_index = copy.deepcopy(self.current_player_index)
            return True
        else:
            return False

    def step(self, action: str) -> bool:
        if self.current_stage == "build":
            self._build(action)
        elif self.current_stage == "diceroll":
            self._diceroll(action)

        if self.winner():
            return True

        self._advance_stage()

        return False

    def _payment(self, payee, reciever, amount):
        payee_money = self.player_coins(payee)
        amount_paid = amount if payee_money - amount >= 0 else payee_money
        self.change_coins(payee, -amount_paid)
        self.change_coins(reciever, amount_paid)
        if self._print_info:
            print(f"{payee} paid {amount_paid} to {reciever}. {payee} has {self.player_coins(payee)} left. {reciever} now has {self.player_coins(reciever)} coins.")
    

    def activate_restaurant(self, player, transfer):
        self._payment(self.current_player, player, transfer)

    
    def activate_primary_industry(self, player, transfer):
        self.change_coins(player, transfer)

    
    def activate_secondary_industry(self, player, transfer):
        if player == self.current_player:
            self.change_coins(player, transfer)


    def _activate_card(self, player, card):
        transfer = self._card_info[card]["transfer"]

        if self._card_info[card]["icon"] == "Combo":
            combo_icon = self._card_info[card]["combo"]
            transfer *= self.player_icon_count(player, combo_icon)
        elif self._card_info[card]["icon"] == "Grain" and self.is_card_owned("Farmers Market"):
            transfer += 1
        elif self._card_info[card]["icon"] == "Bread" and self.is_card_owned("Shopping Mall"):
            transfer += 1
        elif self._card_info[card]["icon"] == "Cup" and self.is_card_owned("Soda Bottling Plant"):
            transfer += 1
        elif self._card_info[card]["icon"] == "Gear" and self.is_card_owned("Forge"):
            transfer += 1

        if self._card_info[card]["type"] == "Restaurant (Red)":
            if self._print_info:
                print(f"{player} gets {transfer} coins from {card}")
            self.activate_restaurant(player, transfer)
        elif self._card_info[card]["type"] == "Primary Industry (Blue)":
            if self._print_info:
                print(f"{player} gets {transfer} coins from {card}")
            self.activate_primary_industry(player, transfer)
        elif self._card_info[card]["type"] == "Secondary Industry (Green)":
            if self._print_info and player == self.current_player:
                print(f"{player} gets {transfer} coins from {card}")
            self.activate_secondary_industry(player, transfer)
        # elif card == "Business Center":
            # not implemented
        elif card == "Stadium":
            if self._print_info:
                print(f"{player} gets 3 coins from all other players through the Stadium.")
            for other_player in self.next_players:
                self._payment(other_player, player, 3)
        elif card == "Shopping District":
            if self._print_info:
                print(f"{player} gets half of the coins from all other players through the Shopping District.")
            for other_player in self.next_players:
                other_player_coins = self.player_coins(other_player)
                if other_player_coins > 10:
                    self._payment(other_player, player, other_player_coins // 2)

class GymMachiKoro2(gym.Env):
    def __init__(
            self,
            n_players: int,
            card_info_path: Optional[str] = None,
            print_info: bool = False,
        ):
        self._env = MachiKoro2(n_players=n_players, card_info_path=card_info_path, print_info=print_info)

        actions = ["1 dice", "2 dice"]
        [actions.append(action) for action in self.card_info.keys()]
        actions.append("Build nothing")
        self._action_idx_to_str = {idx: name for idx, name in enumerate(actions)}
        self._action_str_to_idx = {name: idx for idx, name in enumerate(actions)}

        self.action_space = gym.spaces.Discrete(len(self._action_idx_to_str))
        self._action_mask = np.zeros(self.action_space.n)

        self.observation_space = gym.spaces.MultiBinary(len(self._env.state))

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

    def _get_info(self):
        return {"winning_player_index": self._env.winning_player_index}

    def reset(self, state: dict | None = None):
        if state is not None:
            self.set_state(state)
        else:
            self._env.reset()
        obs = self.observation()
        return obs, self._get_info()

    def state_dict(self, state: Optional[np.array] = None):
        return copy.deepcopy(self._env.state_dict(state))

    def state_dict_to_array(self, state_dict):
        return self._env.state_dict_to_array(state_dict)

    def set_state(self, obs):
        self._env.state = obs

    def step(self, action, state: dict | None = None):
        if state:
            self.set_state(state)

        assert self.action_mask()[action] == 1, f"Action {self._action_idx_to_str[action]} is not allowed in the current state."

        winner = self._env.step(self._action_idx_to_str[action])
        return self.observation(), int(winner), winner, False, self._get_info()

    def _diceroll_action_mask(self):
        self._action_mask*=0
        self._action_mask[self._action_str_to_idx["1 dice"]] = 1
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

            if action_str == "Loan Office":
                can_build_loan_office = True
                if self._env.player_icon_count(self.current_player, "Landmark") >= 1:
                    can_build_loan_office = False

                for other_player in self.next_players():
                    if self._env.player_icon_count(other_player, "Landmark") == 0:
                        can_build_loan_office = False
                        break
                if not can_build_loan_office:
                    continue

            if action_str == "1 dice" or action_str == "2 dice":
                continue
            elif action_str == "Build nothing":
                self._action_mask[action] = 1
            elif action_str in cards_in_marketplace:
                if self.card_info[action_str]["type"] == "Landmark" and self._env.player_icon_count(self.current_player, "Landmark") == 3:
                    # if player already owns 3 landmarks (meaning the player just won), do not 
                    # allow buying a landmark. This is required because MCTS still requests and 
                    # action_mask when the game is done.
                    continue
                cost = self._env.card_cost(self.current_player, action_str)
                if self._env.player_coins(self.current_player) >= cost:
                    self._action_mask[action] = 1
        return self._action_mask

    def action_mask(self, observation: Optional[np.ndarray] = None):
        if observation is not None:
            current_obs = copy.deepcopy(self.observation())
            self.set_state(observation)
            action_mask = self.action_mask()
            self.set_state(current_obs)
            return action_mask

        if self._env.current_stage == "build":
            return copy.deepcopy(self._build_action_mask())
        elif self._env.current_stage == "diceroll":
            return copy.deepcopy(self._diceroll_action_mask())

    def action_mask_as_dict(self, action_mask):
        return {name: action_mask[idx] for idx, name in self._action_idx_to_str.items()}

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
        return copy.deepcopy(self._env.state)

    @property
    def observation_indices(self):
        return self._env._state_indices

    @property
    def observation_values(self):
        return self._env._state_values