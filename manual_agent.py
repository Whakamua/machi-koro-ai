import gym
import numpy as np
from torch import nn
import torch
import torch.functional as F
from pprint import pprint
from tabulate import tabulate
import sys
import random
# random.seed(4)
# np.random.seed(8)
# torch.random.manual_seed(42)

class ManualAgent:
    def __init__(
            self,
            env,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

    def reset(self, state):
        self.env.reset(state=state)


    def print_state_nicely(self, state_dict):
        board = {alley_name: {} for alley_name in state_dict["deques"].keys()}
        board = []

        for alley_name, info in state_dict["deques"].items():
            row = [alley_name]
            cards_in_deque = sum([1 for card in info["cards"] if card != 'None'])
            row.append(cards_in_deque)
            for pos in state_dict["marketplace"][alley_name].values():
                row.append(pos['count'])
                row.append(pos['card'])

            board.append(row)
        
        headers = ["Alley", "Deque"]
        for pos in range(5):
            headers.append("N")
            headers.append("Card")
        print(tabulate(board, headers=headers))


        for player, info in state_dict["player_info"].items():
            print(f"\n{player} | coins {info['coins']} | cards ->")
            player_cards = []
            for card, count in info["cards"].items():
                if count > 0:
                    cards = [card]*count
                    if len(cards) > 1:
                        cards = "\n".join([card]*count)
                    else:
                        cards = cards[0]

                    player_cards.append(cards)
            print(tabulate([player_cards]))


    def compute_action(self, observation):
        self.env.set_state(observation)
        self.print_state_nicely(self.env.state_dict())
        action_mask = self.env.action_mask()
        action_choices = "\n".join([f"{idx+1}: {action}" for idx, action in self.env._action_idx_to_str.items() if action_mask[idx] == 1])
        allowed_actions = [idx for idx in self.env._action_idx_to_str.keys() if action_mask[idx] == 1]
        while True:
            try:
                action = input(f"{self.env.current_player} - choose 1 of the following actions or type `exit` to exit the game:\n{action_choices}\n")
                if action == "exit":
                    assert False
                action = int(action)-1
                assert action in allowed_actions, f"Please select a legal action, you chose {action}"
                break
            except:
                if action == "exit":
                    exit()
                pass
        return action, None