import gym
import yaml
from collections import deque
import random
import copy
import numpy as np

class PlayerInfo:
    def __init__(self, card_info):
        # self._icons = {info["icon"]: Tracker() for info in card_info.values()}
        # self._cards = {card: Card(card, info["cost"], info["activation"], self._icons[info["icon"]], Tracker()) for card, info in card_info.items()}
        self._card_info = card_info
        self._cards = {card: 0 if info["type"] != "Landmarks" else False for card, info in self._card_info.items()}
        self._cards_per_activation = {i: [] for i in range(1, 13)}
        for card, info in self._card_info.items():
            if info["activation"]:
                for activation in range(info["activation"][0], info["activation"][1]):
                    self._cards_per_activation[activation].append(card)
        self.coins = 3
        self._landmarks = 1
        self._tech_startup_investment = 0


        self._icon_count = {}
        for info in self._card_info.values():
            if info["icon"] not in self._icon_count.keys():
                self._icon_count[info["icon"]] = 0

        self._add_card("Wheat Field")
        self._add_card("Bakery")

    @property
    def cards(self):
        return self._cards

    def icon_count(self, icon):
        return self._icon_count[icon]
    
    @property
    def landmarks(self):
        return self._landmarks
    
    def activated_cards(self, diceroll):
        activated_cards = []
        for card in self._cards_per_activation[diceroll]:
            for _ in range(self._cards[card]):
                activated_cards.append(card)
        return activated_cards
    
    def buy_card(self, card):
        assert self._coins >= self._card_info[card]["cost"]
        self._coins -= self._card_info[card]["cost"]
        self._add_card(card)

    def _add_card(self, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self._cards[card] == False
            self._cards[card] = True
            self._landmarks += 1
        else:
            self._cards[card] += 1
            self._icon_count[self._card_info[card]["icon"]] += 1

    def remove_card(self, card):
        if self._card_info[card]["type"] == "Landmarks":
            assert self._cards[card] == True
            self._cards[card] = False
            self._landmarks -= 1
        else:
            assert self._cards[card] >= 1
            self._cards[card] -= 1
            self._icon_count[self._card_info[card]["icon"]] -= 1

class MarketPlace:
    def __init__(self, init_establishments: list[str]):
        self._init_establishments = copy.deepcopy(init_establishments)
        
        self._card_to_alley = {}
        for alley_name, init_establisments_for_alley in self._init_establishments.items():
            for card_name in np.unique(init_establisments_for_alley):
                self._card_to_alley[card_name] = alley_name

        self._marketplace = {"1-6": [deque([]) for _ in range(5)], "7+": [deque([]) for _ in range(5)], "major": [deque([]) for _ in range(2)]}

        self.reset()

    def _fill_alley(self, alley_name: str):
        alley = self._marketplace[alley_name]
        while True:
            if len(self._deques[alley_name]) == 0:
                break

            card_name = self._deques[alley_name].pop()
            
            for stand in alley:
                if len(stand) == 0:
                    card_in_other_stand = False
                    for _stand in alley:
                        if len(_stand) != 0 and card_name == _stand[-1]:
                            card_in_other_stand = True
                    if not card_in_other_stand:
                        stand.append(card_name)
                        break
                elif len(stand) < 5 and stand[-1] == card_name:
                    stand.append(card_name)
                    break

            empty_stands = False
            for stand in alley:
                if len(stand) == 0:
                    empty_stands = True
            if not empty_stands or len(self._deques[alley_name]) == 0:
                break

    def reset(self,):
        [random.shuffle(cards) for cards in self._init_establishments.values()]
        self._deques = {
            name: deque(cards) for name, cards in self._init_establishments.items()
        }

        for alley_name in self._marketplace.keys():
            self._fill_alley(alley_name)

    def get(self, card_name: str):
        alley_name = self._card_to_alley[card_name]
        for stand in self._marketplace[alley_name]:
            if len(stand) != 0 and stand[-1] == card_name:
                card_name = stand.pop()
                if len(stand) == 0:
                    self._fill_alley(alley_name)

                return card_name

    @property
    def obs(self,):
        return self._marketplace

            

class MachiKoro(gym.Env):
    def __init__(self, n_players: int):
        self._n_players = n_players

        with open('src/card_info.yaml') as f:
            self._card_info = yaml.load(f, Loader=yaml.loader.SafeLoader)

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

        self.reset()

    def reset(self):
        self._coins = {
            "1": 42,
            "5": 24,
            "10": 12,
        }

        self._marketplace = MarketPlace(self._init_establishments)

        self._player_info = {
            f"player {i}": PlayerInfo(self._card_info) for i in range(self._n_players)
        }

        self._player_order = [player for player in self._player_info.keys()]
        random.shuffle(self._player_order)
        self._current_player = self._player_order[0]

    def _dice_roll(self, n_dice: int):
        diceroll = 0
        if n_dice == 2:
            assert "Train Station" in self._player_info[self._current_player]["cards"]
        for _ in range(n_dice):
            diceroll += random.randint(1,6)

        for player in self._player_info.keys():
            for card in self._player_info[player].activated_cards(diceroll):
                self._activate_card(player, card)
        

    def step(self, action: str):
        """action: establishment_to_buy"""
        pass

    def _payment(self, payee, reciever, amount):
        payee_money = self._player_info[payee].coins
        amount_paid = amount if payee_money - amount >= 0 else payee_money
        self._player_info[payee].coins -= amount_paid
        self._player_info[reciever].coins += amount_paid
  
    def _activate_card(self, player, card, info: str = None):
        if card == "Wheat Field" or card == "Ranch" or card == "Flower Orchard" or card == "Forest":
            self._player_info[player].coins += 1

        if card == "Mackerel Boat" and self._player_info[player].cards["Harbor"]:
            self._player_info[player].coins += 3
    
        if card == "Apple Orchard":
            self._player_info[player].coins += 3

        if card == "Tuna Boat" and self._player_info[player].cards["Harbor"]:
            diceroll = random.randint(1,6) + random.randint(1,6)
            self._player_info[player].coins += diceroll
        
        if card == "General Store" and player == self._current_player and self._player_info[player].landmarks < 3:
            # less than 2 landmarks but because city hall doesn't count less than 3 is used
            self._player_info[player].coins += 2
        
        if card == "Bakery" and player == self._current_player:
            self._player_info[player].coins += 1

        if card == "Demolition Company" and self._player_info[player].landmarks > 1:
            # if a landmark can be destroyed, city hall doesn't count so check is for when landmarks ar larger than 1.
            self._player_info[player].remove_card(info)
            self._player_info[player].coins += 8
        
        if card == "Flower Shop" and player == self._current_player:
            self._player_info[player].coins += self._player_info[player].cards["Flower Orchard"]

        if card == "Cheese Factory" and player == self._current_player:
            self._player_info[player].coins += self._player_info[player].icon_count("Cow") * 3

        if card == "Furniture Factory" and player == self._current_player:
            self._player_info[player].coins += self._player_info[player].icon_count("Gear") * 3

        if card == "Soda Bottling Plant" and player == self._current_player:
            for player_info in self._player_info.values():
                self._player_info[self._current_player].coins += player_info.icon_count("Cup")
        
        if card == "Fruit and Vegetable Market" and player == self._current_player:
            self._player_info[player].coins += self._player_info[player].icon_count("Grain") * 2

        if card == "Sushi Bar" and player != self._current_player and self._player_info[player].cards["Harbor"]:
            self._payment(self._current_player, player, 3)

        if card == "Café" or card == "Pizza Joint" and player != self._current_player:
            self._payment(self._current_player, player, 1)
        
        if card == "French Restaurant" and player != self._current_player and self._player_info[self._current_player].landmarks > 2:
            # if current player has 2 or more landmarks, transfer 5 coins to the player owning the
            # french restaurant. City hall doesn't count so check is for when landmarks ar larger than 2.
            self._payment(self._current_player, player, 5)

        if card == "Family Restaurant" and player != self._current_player:
            self._payment(self._current_player, player, 2)

        if card == "Member's Only Club" and player != self._current_player and self._player_info[self._current_player].landmarks > 3:
            # if current player has 3 or more landmarks, transfer all their coins to the player owning the
            # member's only club. City hall doesn't count so check is for when landmarks ar larger than 3.
            self._payment(self._current_player, player, self._player_info[self._current_player].coins)

        if card == "Stadium" and player == self._current_player:
            for player in self._player_info.keys():
                if player != self._current_player:
                    self._payment(player, self._current_player, 2)
        
        if card == "Publisher" and player == self._current_player:
            for player in self._player_info.keys():
                if player != self._current_player:
                    coins_to_pay = self._player_info[player].icon_count("Cup") + self._player_info[player].icon_count("Bread")
                    self._payment(player, self._current_player, coins_to_pay)


if __name__ == "__main__":
    env = MachiKoro(4)
    breakpoint()