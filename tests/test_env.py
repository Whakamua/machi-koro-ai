import pytest
from env import MachiKoro, GymMachiKoro
from collections import deque
import random
import numpy as np
import pprint
from random_agent import RandomAgent
import gym
import copy

@pytest.fixture
def env():
    return MachiKoro(n_players=4)

@pytest.fixture
def gymenv(env):
    return GymMachiKoro(env)

@pytest.fixture
def random_agent(gymenv):
    return RandomAgent(gymenv.observation_space, gymenv.action_space)

def get_not_current_player(player_order, current_player):
    assert len(player_order) > 1
    assert len(player_order) == len(set(player_order))

    for player in player_order:
        if current_player != player:
            return player

def test_n_landmarks(env):
    for player, info in env._player_info.items():
        n_landmarks = 0
        assert info.n_landmarks == n_landmarks
        for landmark in info._landmarks:
            info._add_card(landmark)
            n_landmarks += 1
            try:
                assert info.n_landmarks == n_landmarks
            except:
                breakpoint()
            

def test_env_init(env):
    for player, info in env._player_info.items():
        assert info.cards == {
            'Wheat Field': 1, 'Ranch': 0, 'Flower Orchard': 0, 'Forest': 0, 'Mackerel Boat': 0, 
            'Apple Orchard': 0, 'Tuna Boat': 0, 'General Store': 0, 'Bakery': 1, 
            'Demolition Company': 0, 'Flower Shop': 0, 'Cheese Factory': 0, 'Furniture Factory': 0,
            'Soda Bottling Plant': 0, 'Fruit and Vegetable Market': 0, 
            'Sushi Bar': 0, 'Café': 0, 'French Restaurant': 0, 'Pizza Joint': 0, 
            'Family Restaurant': 0, "Member's Only Club": 0, 'Stadium': 0, 'Publisher': 0, 
            'Tax Office': 0, 'Tech Startup': 0, 
            'Harbor': False, 'Train Station': False, 'Shopping Mall': False, 
            'Amusement Park': False, 'Moon Tower': False, 'Airport': False
        }
        assert info.coins == 3
        assert info.n_landmarks == 0
    
def test_city_hall(env):
    for player_info in env._player_info.values():
        player_info._coins = 0

    for i in range(env._n_players):
        env._current_player_index = i
        env._diceroll("1 dice")
        assert env._player_info[env.current_player].coins >= 1

def test_get_one_coin_cards(env):
    for card in ["Wheat Field", "Ranch", "Flower Orchard", "Forest"]:
        for player in env._player_info.keys():
            coins = env._player_info[player].coins
            env._activate_card(player, card)
            assert coins + 1 == env._player_info[player].coins

def test_mackerel_boat(env):
    for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Mackerel Boat")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")

        env._activate_card(player, "Mackerel Boat")
        assert coins + 3 == env._player_info[player].coins

def test_apple_orchard(env):
    for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Apple Orchard")
        assert coins + 3 == env._player_info[player].coins

def test_tuna_boat(env):
     for player in env._player_info.keys():
        coins = env._player_info[player].coins
        env._activate_card(player, "Tuna Boat")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")

        for _ in range(100):
            coins = env._player_info[player].coins
            env._activate_card(player, "Tuna Boat")
            assert 2 <= env._player_info[player].coins - coins <= 12

def test_general_store(env):
    for player in env._player_info.keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins == env._player_info[player].coins

        env.current_player = player
        
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Moon Tower")
        coins = env._player_info[player].coins
        env._activate_card(player, "General Store")
        assert coins == env._player_info[player].coins

def test_bakery(env):
    for player in env._player_info.keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Bakery")
        assert coins == env._player_info[player].coins

        env.current_player = player
        
        coins = env._player_info[player].coins
        env._activate_card(player, "Bakery")
        assert coins + 1 == env._player_info[player].coins

def test_demolition_company(env):
    for player in env._player_info.keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins == env._player_info[player].coins

        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Harbor")
        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False

        env._player_info[player]._add_card("Harbor")
        env._player_info[player]._add_card("Train Station")
        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False
        assert env._player_info[player].cards["Train Station"] == True

        coins = env._player_info[player].coins
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env._player_info[player].coins

        assert env._player_info[player].cards["Harbor"] == False
        assert env._player_info[player].cards["Train Station"] == False

def test_flower_shop(env):
    for player in env._player_info.keys():
        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Flower Orchard")
        env._player_info[player]._add_card("Flower Orchard")
        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins + 2 == env._player_info[player].coins

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Flower Shop")
        assert coins == env._player_info[player].coins

def test_cheese_factory(env):
    for player in env._player_info.keys():
        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Ranch")
        env._player_info[player]._add_card("Ranch")

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins + 2 * 3 == env._player_info[player].coins

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Cheese Factory")
        assert coins == env._player_info[player].coins

def test_furniture_factory(env):
    for player in env._player_info.keys():
        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins == env._player_info[player].coins

        env._player_info[player]._add_card("Forest")
        env._player_info[player]._add_card("Forest")

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins + 2 * 3 == env._player_info[player].coins

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Furniture Factory")
        assert coins == env._player_info[player].coins

def test_soda_bottling_plant(env):

    n_players = 0
    for player in env._player_info.keys():
        n_players+=1
        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Soda Bottling Plant")
        assert coins == env._player_info[player].coins

    for player in env._player_info.keys():
        env._player_info[player]._add_card("Sushi Bar")
        env._player_info[player]._add_card("Café")
        
    for player in env._player_info.keys():

        coins = env._player_info[player].coins
        env._activate_card(player, "Soda Bottling Plant")
        if player == env.current_player:
            assert coins + n_players*2 == env._player_info[player].coins

def test_fruit_and_vegetable_market(env):
    for player in env._player_info.keys():
        env.current_player = player

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 == env._player_info[player].coins

        env._player_info[player]._add_card("Flower Orchard")
        env._player_info[player]._add_card("Apple Orchard")

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 * 3 == env._player_info[player].coins

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env._player_info[player].coins
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins == env._player_info[player].coins

def test_sushi_bar(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Sushi Bar")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins


    env._player_info[env.current_player]._coins = 20

    for _player in env._player_info.keys():
        env._player_info[_player]._add_card("Harbor")

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Sushi Bar")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + 3 == env._player_info[_player].coins
            assert coins_current_player - 3 == env._player_info[env.current_player].coins


def test_cafe(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Café")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + 1 == env._player_info[_player].coins
            assert coins_current_player - 1 == env._player_info[env.current_player].coins

def test_french_restaurant(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "French Restaurant")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins


    env._player_info[env.current_player]._add_card("Harbor")
    env._player_info[env.current_player]._add_card("Airport")

    env._player_info[env.current_player]._coins = 20

    for _player in env._player_info.keys():
        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "French Restaurant")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + 5 == env._player_info[_player].coins # french restaurant requires to transfer 5 coins but players in this test have only 3 coins.
            assert coins_current_player - 5 == env._player_info[env.current_player].coins

def test_pizza_joint(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Pizza Joint")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + 1 == env._player_info[_player].coins
            assert coins_current_player - 1 == env._player_info[env.current_player].coins

def test_family_restaurant(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    env._player_info[env.current_player]._coins = 10

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Family Restaurant")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + 2 == env._player_info[_player].coins
            assert coins_current_player - 2 == env._player_info[env.current_player].coins

def test_members_only_club(env):
    player = list(env._player_info.keys())[0]
    env.current_player = player

    for _player in env._player_info.keys():

        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Member's Only Club")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins


    env._player_info[env.current_player]._add_card("Harbor")
    env._player_info[env.current_player]._add_card("Airport")
    env._player_info[env.current_player]._add_card("Train Station")

    for _player in env._player_info.keys():
        coins_player = env._player_info[_player].coins
        coins_current_player = env._player_info[env.current_player].coins

        env._activate_card(_player, "Member's Only Club")
        if _player == env.current_player:
            assert coins_player == env._player_info[_player].coins
            assert coins_current_player == env._player_info[env.current_player].coins
        else:
            assert coins_player + coins_current_player == env._player_info[_player].coins
            assert env._player_info[env.current_player].coins == 0

def test_stadium(env):
    coins = {}
    n_players = 0
    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins

    env._activate_card(env.current_player, "Stadium")

    for player in env._player_info.keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins

def test_publisher(env):
    coins = {}
    n_players = 0
    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins
        env._player_info[player]._add_card("Café")


    env._activate_card(env.current_player, "Publisher")

    for player in env._player_info.keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins # coins subtracted due to (default) bakery and Café.

def test_tax_office(env):
    env.current_player = list(env._player_info.keys())[0]
    rich_player = list(env._player_info.keys())[1]
    env._player_info[rich_player]._coins = 23

    coins = {}
    for player in env._player_info.keys():
        coins[player] = env._player_info[player].coins

    env._activate_card(env.current_player, "Tax Office")
    for player in env._player_info.keys():
        if player == env.current_player:
            assert coins[player] + 11 == env._player_info[player].coins
        elif player == rich_player:
            assert coins[player] - 11 == env._player_info[player].coins
        else:
            assert coins[player] == env._player_info[player].coins

def test_tech_startup(env):
    coins = {}
    n_players = 0

    env._player_info[env.current_player]._add_card("Tech Startup")
    env.invest_in_tech_startup()
    env.invest_in_tech_startup()

    for player in env._player_info.keys():
        n_players += 1
        coins[player] = env._player_info[player].coins
    


    env._activate_card(env.current_player, "Tech Startup")
    for player in env._player_info.keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env._player_info[player].coins
        else:
            assert coins[player] - 2 == env._player_info[player].coins

def test_shopping_mall_box_icon(env):
    for card, info in env._card_info.items():
        if card == "General Store":
            payment = 2
        elif card == "Bakery":
            payment = 1
        elif card == "Flower Shop":
            env._player_info[env.current_player]._add_card("Flower Orchard")
            payment = 1
        else:
            break
        
        coins = env._player_info[env.current_player].coins
        env._activate_card(env.current_player, card)
        assert coins + payment == env._player_info[env.current_player].coins
        env._player_info[env.current_player]._add_card("Shopping Mall")
        coins = env._player_info[env.current_player].coins
        env._activate_card(env.current_player, card)
        assert coins + payment*2 == env._player_info[env.current_player].coins

def test_shopping_mall_cup_icon(env):
    env.current_player = env._player_order[0]
    second_player = env._player_order[1]

    env._player_info[second_player]._add_card("Harbor")

    for card, info in env._card_info.items():
        for player in env._player_info.keys():
            env._player_info[player]._coins = 10

        if card == "Café" or card == "Pizza Joint":
            payment = 1
        elif card == "French Restaurant":
            env._player_info[env.current_player]._add_card("Harbor")
            env._player_info[env.current_player]._add_card("Airport")
            payment = 5
        elif card == "Family Restaurant":
            payment = 2
        else:
            break
        
        coins_second_player = env._player_info[second_player].coins
        coins = env._player_info[env.current_player].coins
        env._activate_card(second_player, card)
        assert coins_second_player + payment == env._player_info[second_player].coins
        assert coins - payment == env._player_info[env.current_player].coins

        for player in env._player_info.keys():
            env._player_info[player]._coins = 10

        env._player_info[env.current_player]._add_card("Shopping Mall")
        
        coins_second_player = env._player_info[second_player].coins
        coins = env._player_info[env.current_player].coins
        env._activate_card(second_player, card)
        assert coins_second_player + (payment+1) == env._player_info[second_player].coins
        assert coins - (payment+1) == env._player_info[env.current_player].coins

def test_amusement_park(env):
    # so that diceroll will be [1, 1], [1, 3]
    random.seed(2)
    env._player_info[env.current_player]._add_card("Amusement Park")
    env._player_info[env.current_player]._add_card("Train Station")

    for _ in range(5):
        env._player_info[env.current_player]._add_card("Flower Orchard")


    coins = env._player_info[env.current_player].coins
    env.step("2 dice") # throws [1, 1] so can throw again
    env.step("2 dice") # throws [1, 3]

    # 1 for the bakery and 5 for each flower orchard
    assert coins + 1 + 5 == env._player_info[env.current_player].coins

def test_airport(env):
    player = env.current_player
    coins = env._player_info[player].coins
    env._advance_stage()
    env.step("Build nothing")
    assert coins == env._player_info[player].coins

    player = env.current_player
    env._player_info[player]._add_card("Airport")
    coins = env._player_info[player].coins
    env._advance_stage()
    env.step("Build nothing")
    assert coins + 10 == env._player_info[player].coins

def test_earn_income_order(env):
    env._player_order = [f"player {i}" for i in range(1,4)]
    env.current_player = env._player_order[0]
    second_player = env._player_order[1]

    # Test Secondary Industry after Restaurants
    env._player_info[second_player]._add_card("Café")
    env._player_info[env.current_player]._coins = 0
    env._player_info[second_player]._coins = 3

    env._earn_income(3)

    assert env._player_info[env.current_player].coins == 1 # from bakery
    assert env._player_info[second_player].coins == 3 # still 3 because Café could get money from current player because it had 0 coins
    
    # Test Primary Industry after Restaurants
    env._player_info[second_player]._add_card("Sushi Bar")
    env._player_info[env.current_player]._coins = 0
    env._player_info[second_player]._coins = 3

    env._earn_income(1)
    assert env._player_info[env.current_player].coins == 1 # from bakery
    assert env._player_info[second_player].coins == 4 # only 1 from Wheat Field (default card) because Sushi Bar could get money from current player because it had 0 coins

    # Test Major Establishments after Secondary Industry
    env._player_info[second_player]._add_card("Mackerel Boat")
    env._player_info[second_player]._add_card("Harbor")
    env._player_info[env.current_player]._add_card("Tax Office")
    env._player_info[second_player]._coins = 9
    env._player_info[env.current_player]._coins = 0

    env._earn_income(8)

    assert env._player_info[second_player].coins == 6 # got 3 from Mackerel Boat, then lost half through tax office of second player
    assert env._player_info[env.current_player].coins == 6 # got 12 / 2 from second player through tax office

def test_step(env):
    env._current_player_index = 0

    for i in range(len(env._player_order)):
        env._marketplace._state["1-6"][0] = deque(["Ranch"])
        env.step("1 dice")
        env.step("Ranch")
        if i == len(env._player_order) - 1:
            assert env.current_player == env._player_order[0]
        else:
            assert env.current_player == env._player_order[i + 1]

def test_marketplace(env):
    n_cards = 0
    for info in env._card_info.values():
        if info["type"] != "Landmarks":
            n_cards += info["n_cards"]

    
    alleys = list(env._marketplace._state.keys())

    for _ in range(2):
        n_gotten_cards = 0
        for _ in range(n_cards):
            for alley in alleys:
                for stand in env._marketplace._state[alley]:
                    if len(stand) > 0:
                        card_to_get = stand[-1]
                        gotten_card = env._marketplace.get(stand[-1])
                        assert gotten_card == card_to_get
                        n_gotten_cards +=1
        
        for alley in alleys:
            for stand in env._marketplace._state[alley]:
                assert len(stand) == 0
        
        assert n_gotten_cards == n_cards
        env.reset()
    
def test_gym_env_obs(gymenv):
    gymenv.reset()

    gymenv._env._advance_stage()

    obs, _, _, _, _ = gymenv.step(gymenv._action_str_to_idx["Build nothing"])
    obs_key_list = list(obs.keys())

    for i, (obs_key, obs_value) in enumerate(gymenv.observation_space.items()):
        assert obs_key == obs_key_list[i]
        if isinstance(obs_value, gym.spaces.Box) or isinstance(obs_value, gym.spaces.Discrete):
            assert isinstance(obs[obs_key_list[i]], int)
        else:
            assert obs_value.n == len(obs[obs_key_list[i]])
            assert sum(obs[obs_key_list[i]]) == 1

def test_winner_random_policy(gymenv, random_agent):
    obs, _ = gymenv.reset()
    for i in range(1000):
        action, prob_dist = random_agent.compute_action(obs)
        obs, reward, done, truncated, info = gymenv.step(action)
        
        if done:
            break
    assert done

from unittest.mock import MagicMock

def test_roll_dice_always_returns_6(gymenv):
    random.randint = MagicMock(return_value=2)
    # makes sure that player 0 gets 1 coin in the first turn

    init_state = gymenv.get_state()
    init_state_deepcopy = copy.deepcopy(init_state)
    assert init_state.player_info["player 0"].coins == 3

    obsa, reward, done, _, info = gymenv.step(0)
    obsa_deepcopy = copy.deepcopy(obsa)
    state_tp1a = info["state"]
    assert init_state.player_info["player 0"].coins == 3
    assert state_tp1a.player_info["player 0"].coins == 4

    assert init_state != state_tp1a
    assert init_state != gymenv.get_state()
    assert info["state"] == gymenv.get_state()
    assert init_state == init_state_deepcopy

    gymenv.set_state(init_state)
    assert init_state.player_info["player 0"].coins == 3

    obsb, reward, done, _, info = gymenv.step(0)
    obsb_deepcopy = copy.deepcopy(obsb)
    state_tp1b = info["state"]
    assert init_state.player_info["player 0"].coins == 3
    assert state_tp1b.player_info["player 0"].coins == 4

    assert state_tp1a == state_tp1b

    for key in obsa_deepcopy.keys():
        assert np.array_equal(obsa_deepcopy[key], obsb_deepcopy[key])
    
