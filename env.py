import gym
import yaml
from collections import deque
import random
import copy
import numpy as np
from collections import OrderedDict

class State():
    def __init__(
            self,
            player_info,
            marketplace,
            current_player_index,
            current_stage_index,
            ):
        self.player_info = player_info
        self.marketplace = marketplace
        self.current_player_index = current_player_index
        self.current_stage_index = current_stage_index

    def __eq__(self, other):

        equal = True
        if self.player_info.keys() != other.player_info.keys():
            print("1")
            equal = False

        for player in self.player_info.keys():
            if self.player_info[player].__dict__ != other.player_info[player].__dict__:
                print("2", player)
                equal = False

        if self.marketplace.__dict__ != other.marketplace.__dict__:
            print("3")
            equal = False

        if self.current_player_index != other.current_player_index:
            print("4")
            equal = False
        
        if self.current_stage_index != other.current_stage_index:
            print("5")
            equal = False

        return equal

    


class PlayerInfo:
    def __init__(self, card_info):
        self._card_info = card_info
        self._cards = {card: 0 if info["type"] != "Landmarks" else False for card, info in self._card_info.items()}
        self._landmarks = [card for card, info in self._card_info.items() if info["type"] == "Landmarks"]
        self._max_landmarks = len(self._landmarks)
        self._coins = 3
        self._tech_startup_investment = 0


        self._icon_count = {}
        for info in self._card_info.values():
            if info["icon"] not in self._icon_count.keys():
                self._icon_count[info["icon"]] = 0

        self._add_card("Wheat Field")
        self._add_card("Bakery")

        self._state_indices = {
            name: i for i, name in enumerate(
                list(self._cards.keys())
                + ["coins"]
                + ["tech_startup_investment"]
            )
        }
        self._state = np.zeros(
            len(self._state_indices)
        ).tolist()

    @property
    def coins(self):
        return self._coins

    def invest_in_tech_startup(self):
        assert self.coins != 0
        self._tech_startup_investment += 1
        self._coins -= 1

    @property
    def tech_startup_investment(self):
        return self._tech_startup_investment

    @property
    def cards(self):
        return self._cards

    def icon_count(self, icon):
        return self._icon_count[icon]

    @property
    def n_landmarks(self):
        n_landmarks = 0
        for landmark in self._landmarks:
            n_landmarks += self._cards[landmark]
        return n_landmarks

    def buy_card(self, card):
        assert self.coins >= self._card_info[card]["cost"]
        self._coins -= self._card_info[card]["cost"]
        self._add_card(card)

    def _add_card(self, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self._cards[card] == False
            self._cards[card] = True
        else:
            self._cards[card] += 1
            self._icon_count[self._card_info[card]["icon"]] += 1

    def remove_card(self, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self._cards[card] == True
            self._cards[card] = False
        else:
            assert self._cards[card] >= 1
            self._cards[card] -= 1
            self._icon_count[self._card_info[card]["icon"]] -= 1

    def as_list(self):
        for card in self._cards.keys():
            self._state[self._state_indices[card]] = self._cards[card]
            
        self._state[self._state_indices["coins"]] = self._coins
        self._state[self._state_indices["tech_startup_investment"]] = self._tech_startup_investment
        return self._state
    
    def from_list(self, _list):
        for card in self._cards.keys():
            self._cards[card] = _list[self._state_indices[card]]
        self._coins = _list[self._state_indices["coins"]]
        self._tech_startup_investment = _list[self._state_indices["tech_startup_investment"]]


class MarketPlace:
    def __init__(self, init_establishments: list[str], card_info):
        self._card_info = card_info
        self._init_establishments = copy.deepcopy(init_establishments)
        
        self._card_to_alley = {}
        for alley_name, init_establisments_for_alley in self._init_establishments.items():
            for card_name in np.unique(init_establisments_for_alley):
                self._card_to_alley[card_name] = alley_name

        [random.shuffle(cards) for cards in self._init_establishments.values()]
        self._deques = {
            name: copy.deepcopy(cards) for name, cards in self._init_establishments.items()
        }

        self._alleys = {"1-6": [deque([]) for _ in range(5)], "7+": [deque([]) for _ in range(5)], "major": [deque([]) for _ in range(2)]}

        self._card_name_to_num = {name: i for i, (name, info) in enumerate(list(self._card_info.items()))}
        self._card_num_to_name = {i: name for name, i in self._card_name_to_num.items()}

        self._state_indices = {}
        self._state_indices["deques"] = {}
        self._deque_max_lengths = {}
        state_len = 0

        self._state_indices["deques"]["1-6"] = list(range(state_len, state_len+len(self._deques["1-6"])))
        state_len += len(self._deques["1-6"])
        # self._deque_max_lengths["1-6"] = len(self._deques["1-6"])

        self._state_indices["deques"]["7+"] = list(range(state_len, state_len+len(self._deques["7+"])))
        state_len += len(self._deques["7+"])
        # self._deque_max_lengths["7+"] = len(self._deques["7+"])

        self._state_indices["deques"]["major"] = list(range(state_len, state_len+len(self._deques["major"])))
        state_len += len(self._deques["major"])
        # self._deque_max_lengths["major"] = len(self._deques["major"])


        self._state_indices["marketplace"] = {}
        self._state_indices["marketplace"]["1-6"] = {}
        for pos in range(5):
            self._state_indices["marketplace"]["1-6"][f"pos_{pos}"] = {"card": state_len, "count": state_len+1}
            state_len += 2

        self._state_indices["marketplace"]["7+"] = {}
        for pos in range(5):
            self._state_indices["marketplace"]["7+"][f"pos_{pos}"] = {"card": state_len, "count": state_len+1}
            state_len += 2

        self._state_indices["marketplace"]["major"] = {}
        for pos in range(2):
            self._state_indices["marketplace"]["major"][f"pos_{pos}"] = {"card": state_len, "count": state_len+1}
            state_len += 2


        self._state = np.zeros(state_len).tolist()

        for alley_name in self._alleys.keys():
            self._fill_alley(alley_name)

    def _fill_alley(self, alley_name: str):
        alley = self._alleys[alley_name]
        while True:
            if len(self._deques[alley_name]) == 0:
                break
            # assert False, "only take card when it can be put in an alley. Currently the card is thrown if there is not spot"
            card_name = self._deques[alley_name][-1]

            for stand in alley:
                if len(stand) == 0:
                    card_in_other_stand = False
                    for other_stand in alley:
                        if len(other_stand) != 0 and card_name == other_stand[-1]:
                            card_in_other_stand = True
                    if not card_in_other_stand:
                        stand.append(self._deques[alley_name].pop(-1))
                        break
                elif stand[-1] == card_name:
                    stand.append(self._deques[alley_name].pop(-1))
                    break

            empty_stands = False
            for stand in alley:
                if len(stand) == 0:
                    empty_stands = True
            if not empty_stands or len(self._deques[alley_name]) == 0:
                break

    def get(self, card_name: str):
        alley_name = self._card_to_alley[card_name]
        for stand in self._alleys[alley_name]:
            if len(stand) != 0 and stand[-1] == card_name:
                card_name = stand.pop()
                if len(stand) == 0:
                    self._fill_alley(alley_name)

                return card_name
        assert False, f"Card {card_name} not in marketplace"

    def as_list(self):
        for deque_name, deque in self._deques.items():
            deque_length = len(deque)
            for i, state_idx in enumerate(self._state_indices["deques"][deque_name]):
                if i >= deque_length:
                    self._state[state_idx] = -1
                else:
                    self._state[state_idx] = self._card_name_to_num[deque[i]]

        for alley_name, alley in self._alleys.items():
            for pos, stand in enumerate(alley):
                if len(stand) == 0:
                    card = -1
                else:
                    card = self._card_name_to_num[stand[0]]
                self._state[self._state_indices["marketplace"][alley_name][f"pos_{pos}"]["card"]] = card
                self._state[self._state_indices["marketplace"][alley_name][f"pos_{pos}"]["count"]] = len(stand)
        return self._state
    
    def from_list(self, _list):
        for deque_name in self._deques.keys():
            self._deques[deque_name] = []
            for state_idx in self._state_indices["deques"][deque_name]:
                if _list[state_idx] == -1:
                    break
                self._deques[deque_name].append(self._card_num_to_name[_list[state_idx]])

        for alley_name, alley in self._alleys:
            for pos in range(len(alley)):
                card = self._state[self._state_indices["marketplace"][alley_name][f"pos_{pos}"]["card"]]
                count = self._state[self._state_indices["marketplace"][alley_name][f"pos_{pos}"]["count"]]
                if card == -1:
                    self._alleys[alley_name][pos] = deque([])
                else:
                    self._alleys[alley_name][pos] = deque([self._card_num_to_name[card]]*count)
            

class MachiKoro:
    def __init__(self, n_players: int):
        self._n_players = n_players

        with open('card_info.yaml') as f:
        # with open('card_info_quick_game.yaml') as f:
            self._card_info = yaml.load(f, Loader=yaml.loader.SafeLoader)

        self._cards_per_activation = {i: [] for i in range(1, 13)}
        for card, info in self._card_info.items():
            if info["activation"]:
                for activation in range(info["activation"][0], info["activation"][1] + 1):
                    self._cards_per_activation[activation].append(card)

        self._landmark_cards_ascending_in_price = [card for card, info in self._card_info.items() if info["type"] == "Landmarks"]
        self._init_establishments = {
            "1-6": [],
            "7+": [],
            "major": [],
        }

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

        self._state_indices = {}
        state_len = 0

        self._state_indices["player_info"] = {}
        for player in self._player_info.keys():
            player_info_state = self._player_info[player].as_list()
            self._state_indices["player_info"][player] = list(range(len(player_info_state)))
            state_len += len(player_info_state)

        marketplace_state = self._marketplace.as_list()
        self._state_indices["marketplace"] = list(range(state_len, state_len+len(marketplace_state)))
        state_len += len(marketplace_state)
        self._state_indices["current_player_index"] = state_len
        state_len += 1
        self._state_indices["current_stage_index"] = state_len
        state_len += 1

        self._state = np.zeros(state_len)

    def get_state_as_list(self):
        for player in self._player_info.keys():
            self._state[self._state_indices["player_info"][player]] = self._player_info[player].as_list()
        self._state[self._state_indices["marketplace"]] = self._marketplace.as_list()
        self._state[self._state_indices["current_player_index"]] = self._current_player_index
        self._state[self._state_indices["current_stage_index"]] = self._current_stage_index
        return list(self._state)

    def set_state_from_list(self, _list):
        _array = np.array(_list)
        for player in self._player_info.keys():
            self._player_info[player].from_list(list(_array[self._state_indices["player_info"][player]]))
        self._marketplace.from_list(list(_array[self._state_indices["marketplace"]]))
        self._current_player_index = _array[self._state_indices["current_player_index"]]
        self._current_stage_index = _array[self._state_indices["current_stage_index"]]

    def reset(self):

        self._marketplace = MarketPlace(self._init_establishments, self._card_info)

        self._player_info = {
            self._player_order[i]: PlayerInfo(self._card_info) for i in range(self._n_players)
        }

        self._current_player_index = 0

        self._current_stage_index = 0

    @property
    def current_player(self):
        return self._player_order[self._current_player_index]
    
    @current_player.setter
    def current_player(self, val):
        self._current_player_index = self._player_order.index(val)
    
    @property
    def current_stage(self):
        return self._stage_order[self._current_stage_index]
    
    @property
    def player_info(self):
        return self._player_info

    @property
    def n_players(self):
        return self._n_players

    @property
    def player_order(self):
        return self._player_order

    def _earn_income(self, diceroll):
        for card_type in ["Restaurants", "Secondary Industry", "Primary Industry", "Major Establishment"]:
            for player in self._player_info.keys():
                for card in self._cards_per_activation[diceroll]:
                    if self._card_info[card]["type"] == card_type:
                        for _ in range(self._player_info[player].cards[card]):
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
            assert self._player_info[self.current_player].cards["Train Station"]
            
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
        
        if self._player_info[self.current_player].cards["Amusement Park"] and n_dice > 1 and dicerolls[0] == dicerolls[1]:
            # Amusement Park logic
            return True

        if self._player_info[self.current_player].coins == 0:
            # City Hall logic
            self._player_info[self.current_player]._coins = 1
        
        return False
    
    def invest_in_tech_startup(self):
        # Not used yet, requires 3rd stage to be added to the stage order.
        assert self._player_info[self.current_player].coins != 0
        self._player_info[self.current_player].invest_in_tech_startup()
    
    def _build(self, action: str):
        """
        action: establishment_to_buy
        returns whether the action resulted in a win or not.
        """
        if action == "Build nothing" and self._player_info[self.current_player].cards["Airport"]:
            self._player_info[self.current_player]._coins += 10

        if action != "Build nothing":
            self._player_info[self.current_player].buy_card(action)
            if self._card_info[action]["type"] != "Landmarks":
                self._marketplace.get(action)

    def _advance_stage(self):
        self._current_stage_index += 1
        if self._current_stage_index > self._n_stages - 1:
            self._current_stage_index = 0
            self._advance_player()

    def _advance_player(self):
        self._current_player_index += 1
        if self._current_player_index > self._n_players - 1:
            self._current_player_index = 0
    
    def winner(self):
        for player in self._player_order:
            if self._player_info[player].n_landmarks == self._player_info[player]._max_landmarks:
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
        payee_money = self._player_info[payee].coins
        amount_paid = amount if payee_money - amount >= 0 else payee_money
        self._player_info[payee]._coins -= amount_paid
        self._player_info[reciever]._coins += amount_paid
  
    def _activate_card(self, player, card):
        if card == "Wheat Field" or card == "Ranch" or card == "Flower Orchard" or card == "Forest":
            self._player_info[player]._coins += 1

        elif card == "Mackerel Boat" and self._player_info[player].cards["Harbor"]:
            self._player_info[player]._coins += 3
    
        elif card == "Apple Orchard":
            self._player_info[player]._coins += 3

        elif card == "Tuna Boat" and self._player_info[player].cards["Harbor"]:
            diceroll = random.randint(1,6) + random.randint(1,6)
            self._player_info[player]._coins += diceroll
        
        elif card == "General Store" and player == self.current_player and self._player_info[player].n_landmarks <= 1:
            self._player_info[player]._coins += 2 if not self._player_info[player].cards["Shopping Mall"] else 3
        
        elif card == "Bakery" and player == self.current_player:
            self._player_info[player]._coins += 1 if not self._player_info[player].cards["Shopping Mall"] else 2

        elif card == "Demolition Company" and player == self.current_player and self._player_info[player].n_landmarks >= 1:
            # Not implemented, promt player which landmark to destroy.
            destroyed_landmark = None
            for landmark in self._landmark_cards_ascending_in_price:
                if self._player_info[player].cards[landmark]:
                    self._player_info[player].remove_card(landmark)
                    self._player_info[player]._coins += 8
                    destroyed_landmark = landmark
                    break
            assert destroyed_landmark is not None

        elif card == "Flower Shop" and player == self.current_player:
            self._player_info[player]._coins += self._player_info[player].cards["Flower Orchard"] if not self._player_info[self.current_player].cards["Shopping Mall"] else self._player_info[player].cards["Flower Orchard"]*2

        elif card == "Cheese Factory" and player == self.current_player:
            self._player_info[player]._coins += self._player_info[self.current_player].icon_count("Cow") * 3

        elif card == "Furniture Factory" and player == self.current_player:
            self._player_info[player]._coins += self._player_info[self.current_player].icon_count("Gear") * 3

        # Not implemented
        # elif card == "Moving Company" and player == self.current_player:
            # give non major (icon) to another player and get 4 coins from the bank

        elif card == "Soda Bottling Plant" and player == self.current_player:
            for player_info in self._player_info.values():
                self._player_info[self.current_player]._coins += player_info.icon_count("Cup")
        
        elif card == "Fruit and Vegetable Market" and player == self.current_player:
            self._player_info[player]._coins += self._player_info[player].icon_count("Grain") * 2

        elif card == "Sushi Bar" and player != self.current_player and self._player_info[player].cards["Harbor"]:
            self._payment(self.current_player, player, 3 if not self._player_info[player].cards["Shopping Mall"] else 4)

        elif card == "Café" or card == "Pizza Joint" and player != self.current_player:
            self._payment(self.current_player, player, 1 if not self._player_info[player].cards["Shopping Mall"] else 2)
        
        elif card == "French Restaurant" and player != self.current_player and self._player_info[self.current_player].n_landmarks >= 2:
            # if current player has 2 or more landmarks, transfer 5 coins to the player owning the
            self._payment(self.current_player, player, 5 if not self._player_info[player].cards["Shopping Mall"] else 6)

        elif card == "Family Restaurant" and player != self.current_player:
            self._payment(self.current_player, player, 2 if not self._player_info[player].cards["Shopping Mall"] else 3)

        elif card == "Member's Only Club" and player != self.current_player and self._player_info[self.current_player].n_landmarks >= 3:
            # if current player has 3 or more landmarks, transfer all their coins to the player owning the
            # member's only club.
            self._payment(self.current_player, player, self._player_info[self.current_player].coins)

        elif card == "Stadium" and player == self.current_player:
            for player in self._player_info.keys():
                if player != self.current_player:
                    self._payment(player, self.current_player, 2)
        
        elif card == "Publisher" and player == self.current_player:
            for player in self._player_info.keys():
                if player != self.current_player:
                    coins_to_pay = self._player_info[player].icon_count("Cup") + self._player_info[player].icon_count("Box")
                    self._payment(player, self.current_player, coins_to_pay)

        elif card == "Tax Office" and player == self.current_player:
            for player in self._player_info.keys():
                if player != self.current_player and self._player_info[player].coins >= 10:
                    coins_to_pay = int(self._player_info[player].coins / 2)
                    self._payment(player, self.current_player, coins_to_pay)

        # Not implemented
        # elif card == "Business Center" and player == self.current_player:
            # trade non major (icon) card with another player

        elif card == "Tech Startup" and player == self.current_player and self._player_info[self.current_player].tech_startup_investment > 0:
            for player in self._player_info.keys():
                if player != self.current_player:
                    self._payment(player, self.current_player, self._player_info[self.current_player].tech_startup_investment)

class GymMachiKoro(gym.Env):
    def __init__(self, env: MachiKoro):
        self._env = env

        actions = ["1 dice", "2 dice"]
        [actions.append(action) for action in self.card_info.keys()]
        actions.append("Build nothing")
        self._action_idx_to_str = {idx: name for idx, name in enumerate(actions)}
        self._action_str_to_idx = {name: idx for idx, name in enumerate(actions)}

        self.action_space = gym.spaces.Discrete(len(self._action_idx_to_str))

        self._player_to_idx = {player: idx for idx, player in enumerate(self._env._player_order)}

        self._establishments_to_idx = {
            "1-6": {},
            "7+": {},
            "major": {},
        }

        for card_name, info in self.card_info.items():
            if info["type"] == "Landmarks":
                continue
            elif info["type"] == "Major Establishment" and card_name not in self._establishments_to_idx["major"]:
                self._establishments_to_idx["major"][card_name] = len(self._establishments_to_idx["major"])
            elif info["activation"][0] <= 6 and card_name not in self._establishments_to_idx["1-6"]:
                self._establishments_to_idx["1-6"][card_name] = len(self._establishments_to_idx["1-6"])
            elif card_name not in self._establishments_to_idx["7+"]:
                self._establishments_to_idx["7+"][card_name] = len(self._establishments_to_idx["7+"])

        obs_space = OrderedDict()

        def add_player_obs_spaces(player):
            for card in self.card_info.keys():
                if self.card_info[card]["type"] == "Landmarks":
                    obs_space[f"{player}-{card}"] = gym.spaces.MultiBinary(2)
                else:
                    max_cards = self.card_info[card]["n_cards"]
                    obs_space[f"{player}-{card}"] = gym.spaces.MultiBinary(1 + max_cards + (card == "Wheat Field") + (card == "Bakery")) # + 1 so that the 1st entry is reserved for 0 cards
            obs_space[f"{player}-coins"] = gym.spaces.Box(low=0, high=np.inf)
        
        add_player_obs_spaces("current_player")

        for i in range(len(self._env._player_info)-1):
            add_player_obs_spaces(f"player_current_p_{i+1}")

        for alley_name, alley in self._env._marketplace._alleys.items():
            for i in range(len(alley)):
                # encoding which card is in the deck
                obs_space[f"marketplace-{alley_name}-{i}-card"] = gym.spaces.MultiBinary(1 + len(self._establishments_to_idx[alley_name])) # +1 so that the 1st entry is reserved for an empty deque

                # encoding how many cards are in the deck
                obs_space[f"marketplace-{alley_name}-{i}-amount"] = gym.spaces.MultiBinary(5+1 if alley_name == "major" else 6+1) # major have max 5 cards and others have max 6 cards. +1 is to account for and empty deque
        obs_space["current_player_index"] = gym.spaces.Discrete(self.n_players)
        obs_space["current_stage_index"] = gym.spaces.Discrete(self._env._n_stages)
        obs_space["action_mask"] = gym.spaces.MultiBinary(self.action_space.n)
        self.observation_space = gym.spaces.Dict(obs_space)
        self._obs = OrderedDict()

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
        return self._env._current_player_index
    
    @property
    def current_stage(self):
        return self._env.current_stage

    @property
    def current_stage_index(self):
        return self._env._current_stage_index

    def diceroll(self, n_dice):
        return self._env._diceroll(n_dice)

    def _get_info(self):
        return {
            "state": self.get_state(),
            "player_index": self.current_player_index,
            "is_stochastic": self._env.current_stage == "diceroll"
        }

    def reset(self, state: dict | None = None):
        if state is not None:
            self.set_state(state)
        else:  
            self._env.reset()
        return self.observation(), self._get_info()

    def get_state_list(self):
        return copy.deepcopy(self._env.get_state_as_list())

    def set_state_list(self):
        pass
        # return copy.deepcopy(self._env.set_state_from_list())
    
    def get_state(self):
        self.get_state_list()
        return State(
            player_info = copy.deepcopy(self._env._player_info),
            marketplace = copy.deepcopy(self._env._marketplace),
            current_player_index = copy.deepcopy(self._env._current_player_index),
            current_stage_index = copy.deepcopy(self._env._current_stage_index)
        )

    def set_state(self, state: State):
        self.set_state_list()
        self._env._player_info = copy.deepcopy(state.player_info)
        self._env._marketplace = copy.deepcopy(state.marketplace)
        self._env._current_player_index = copy.deepcopy(state.current_player_index)
        self._env._current_stage_index = copy.deepcopy(state.current_stage_index)

    def step(self, action, state: dict | None = None):
        if state:
            self.set_state(state)
        
        winner = self._env.step(self._action_idx_to_str[action])
        obs = self.observation()
        info = self._get_info()
        return obs, int(winner), winner, False, info

    def _diceroll_action_mask(self):
        action_mask = np.zeros(self.action_space.n)
        action_mask[self._action_str_to_idx["1 dice"]] = 1
        if self._env._player_info[self.current_player].cards["Train Station"]:
            action_mask[self._action_str_to_idx["2 dice"]] = 1
        return action_mask


    def _build_action_mask(self):
        action_mask = np.zeros(self.action_space.n)
        cards_in_marketplace = []
        for alley in self._env._marketplace._alleys.values():
            for stand in alley:
                if len(stand) > 0:
                    cards_in_marketplace.append(stand[-1])
        for action in range(self.action_space.n):
            action_str = self._action_idx_to_str[action]
            if action_str == "1 dice" or action_str == "2 dice":
                continue
            elif action_str == "Build nothing" or self._env._player_info[self.current_player].coins >= self.card_info[action_str]["cost"] and action_str in cards_in_marketplace:
                action_mask[action] = 1
            elif self.card_info[action_str]["type"] == "Landmarks" and self._env._player_info[self.current_player].coins >= self.card_info[action_str]["cost"] and not self._env._player_info[self.current_player].cards[action_str]:
                action_mask[action] = 1
        return action_mask

    @property
    def action_mask(self):
        if self._env.current_stage == "build":
            return self._build_action_mask()
        elif self._env.current_stage == "diceroll":
            return self._diceroll_action_mask()

    # def sample_action(self):
    #     action_mask = self.action_mask()
    #     prob_dist = action_mask/sum(action_mask)
    #     return np.random.choice(range(self.action_space.n), p=prob_dist)
    
    def _add_player_obs(self, player, player_obs_name):
        for card, amount in self._env._player_info[player].cards.items():
            if self.card_info[card]["type"] == "Landmarks":
                onehot = np.array([not amount, amount]).astype(int)
            else:
                max_cards = self.card_info[card]["n_cards"]
                onehot = np.zeros(max_cards + 1 + (card == "Wheat Field") + (card == "Bakery"))
                onehot[amount] = 1
            self._obs[f"{player_obs_name}-{card}"] = onehot
        self._obs[f"{player_obs_name}-coins"] = self._env._player_info[player].coins
    
    def _next_players(self):
        if self._env._current_player_index == self.n_players-1:
            return self._env._player_order[:self._env._current_player_index]
        else:
            return self._env._player_order[self._env._current_player_index+1:] + self._env._player_order[:self._env._current_player_index]
 
    def observation(self):

        self._add_player_obs(self.current_player, "current_player")

        for i, player in enumerate(self._next_players()):
            self._add_player_obs(player, f"player_current_p_{i+1}")
        
        for alley_name, alley in self._env._marketplace._alleys.items():
            for i, stand in enumerate(alley):
                # encoding which card is in the deck
                if len(stand) != 0:
                    card = stand[-1]
                    onehot = np.zeros(1 + len(self._establishments_to_idx[alley_name])) # +1 so that the 1st entry is reserved for an empty deque
                    onehot[1 + self._establishments_to_idx[alley_name][card]] = 1 # +1 to account for an empty deck
                else:
                    onehot = np.zeros(1 + len(self._establishments_to_idx[alley_name])) # +1 so that the 1st entry is reserved for an empty deque
                    onehot[0] = 1
                self._obs[f"marketplace-{alley_name}-{i}-card"] = onehot

                # encoding how many cards are in the deck
                onehot = np.zeros(5+1 if alley_name == "major" else 6+1) # major have max 5 cards and others have max 6 cards. +1 is to account for and empty deque
                onehot[len(stand)] = 1
                self._obs[f"marketplace-{alley_name}-{i}-amount"] = onehot
        self._obs["current_player_index"] = self._env._current_player_index
        self._obs["current_stage_index"] = self._env._current_stage_index
        self._obs["action_mask"] = self.action_mask
        return copy.deepcopy(self._obs)
    
    def flattened_obs(self,):
        return gym.spaces.flatten(self.observation_space, self.observation())
    
    def _player_info_from_obs(self, player, player_obs_name):
            info = PlayerInfo(self.card_info)
            
            for card, amount in self._env._player_info[player].cards.items():
                if self.card_info[card]["type"] == "Landmarks":
                    onehot = np.array([not amount, amount]).astype(int)
                else:
                    max_cards = self.card_info[card]["n_cards"]
                    onehot = np.zeros(max_cards + 1 + (card == "Wheat Field") + (card == "Bakery"))
                    onehot[amount] = 1
                self._obs[f"{player_obs_name}-{card}"] = onehot
            self._obs[f"{player_obs_name}-coins"] = self._env._player_info[self.current_player].coins
    
    def getObservation(self, state):
        self.set_state(state)
        return self.observation()

    def getStringRepresentation(self, state):
        self.set_state(state)
        return copy.deepcopy("".join(map(str,gym.spaces.flatten(self.observation_space, self._obs).astype(int))))

    def getActionSize(self):
        return self.action_space.n

    def getIsTerminal(self, state):
        self.set_state(state)
        return self._env.winner()
    
    def getValidMoves(self, state):
        self.set_state(state)
        return self.action_mask