import gym
import yaml
from collections import deque
import random
import copy
import numpy as np

class MarketPlace:
    def __init__(self, init_establishments):
        self._init_establishments = copy.deepcopy(init_establishments)
        
        self._card_to_alley = {}
        for alley_name, init_establisments_for_alley in self._init_establishments.items():
            for card_name in np.unique(init_establisments_for_alley):
                self._card_to_alley[card_name] = alley_name

        self._marketplace = {"1-6": [deque([]) for _ in range(5)], "7+": [deque([]) for _ in range(5)], "major": [deque([]) for _ in range(2)]}

        self.reset()

    def _fill_alley(self, alley_name):
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

    def _fill_stand(self, stand, alley_name):
        if len(self._deques[alley_name]) > 0:
            card_name = self._deques[alley_name].pop()

    def get(self, card_name):
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

        with open('card_info.yaml') as f:
            self._card_info = yaml.load(f, Loader=yaml.loader.SafeLoader)

        self._activations = {i: [] for i in range(1,13)}

        # create a dictionary of dice roll as keys and a list of activated establishments as values
        for card_name, info in self._card_info.items():
            if info["type"] != "Landmarks":
                for i in range(info["activation"][0], info["activation"][1] + 1):
                    if info["activation"][1] > 12:
                        self._activations[12].append(card_name)
                        break
                    else:
                        self._activations[i].append(card_name)

        self._init_establishments = {
            "1-6": [],
            "7+": [],
            "major": [],
        }

        for card_name, info in self._card_info.items():
            if info["type"] == "Landmarks":
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
        breakpoint()
        




if __name__ == "__main__":
    env = MachiKoro(4)
    breakpoint()