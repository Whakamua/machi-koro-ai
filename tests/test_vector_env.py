import pytest
from env_vector_state import MachiKoro, GymMachiKoro
from collections import deque
import random
import numpy as np
import pprint
from random_agent import RandomAgent
import gym
import copy
from unittest.mock import patch

@pytest.fixture
def n_players():
    return 4

@pytest.fixture
def env(n_players):
    return MachiKoro(n_players=n_players)

@pytest.fixture
def gymenv(n_players):
    return GymMachiKoro(n_players=n_players)

@pytest.fixture
def random_agent(gymenv):
    return RandomAgent(gymenv)

def get_not_current_player(player_order, current_player):
    assert len(player_order) > 1
    assert len(player_order) == len(set(player_order))

    for player in player_order:
        if current_player != player:
            return player

def test_n_landmarks(env):
    for player, info in env.state_dict()["player_info"].items():
        n_landmarks = 0
        assert env.n_of_landmarks_player_owns(player) == n_landmarks
        for landmark in env._landmark_cards_ascending_in_price:
            env.add_card(player, landmark)
            n_landmarks += 1
            assert env.n_of_landmarks_player_owns(player) == n_landmarks
            

def test_env_init(env):
    for player, info in env.state_dict()["player_info"].items():
        assert info["cards"] == {
            'Wheat Field': 1, 'Ranch': 0, 'Flower Orchard': 0, 'Forest': 0, 'Mackerel Boat': 0, 
            'Apple Orchard': 0, 'Tuna Boat': 0, 'General Store': 0, 'Bakery': 1, 
            'Demolition Company': 0, 'Flower Shop': 0, 'Cheese Factory': 0, 'Furniture Factory': 0,
            'Soda Bottling Plant': 0, 'Fruit and Vegetable Market': 0, 
            'Sushi Bar': 0, 'Café': 0, 'French Restaurant': 0, 'Pizza Joint': 0, 
            'Family Restaurant': 0, "Member's Only Club": 0, 'Stadium': 0, 'Publisher': 0, 
            'Tax Office': 0, 'Tech Startup': 0, 
            'Harbor': 0, 'Train Station': 0, 'Shopping Mall': 0, 
            'Amusement Park': False, 'Moon Tower': 0, 'Airport': 0
        }
        assert env.player_coins(player) == 3
        assert env.n_of_landmarks_player_owns(player) == 0
    
def test_city_hall(env):
    for player, player_info in env.state_dict()["player_info"].items():
        env.player_coins(player) == 0

    for player in env.player_order:
        env._current_player_index = player
        env._diceroll("1 dice")
        assert env.player_coins(player) >= 1

def test_get_one_coin_cards(env):
    for card in ["Wheat Field", "Ranch", "Flower Orchard", "Forest"]:
        for player in env.state_dict()["player_info"].keys():
            coins = env.player_coins(player)
            env._activate_card(player, card)
            assert coins + 1 == env.player_coins(player)

def test_mackerel_boat(env):
    for player in env.state_dict()["player_info"].keys():
        coins = env.player_coins(player)
        env._activate_card(player, "Mackerel Boat")
        assert coins == env.player_coins(player)

        env.add_card(player, "Harbor")

        env._activate_card(player, "Mackerel Boat")
        assert coins + 3 == env.player_coins(player)

def test_apple_orchard(env):
    for player in env.state_dict()["player_info"].keys():
        coins = env.player_coins(player)
        env._activate_card(player, "Apple Orchard")
        assert coins + 3 == env.player_coins(player)

def test_tuna_boat(env):
     for player in env.state_dict()["player_info"].keys():
        coins = env.player_coins(player)
        env._activate_card(player, "Tuna Boat")
        assert coins == env.player_coins(player)

        env.add_card(player, "Harbor")

        for _ in range(100):
            coins = env.player_coins(player)
            env._activate_card(player, "Tuna Boat")
            assert 2 <= env.player_coins(player) - coins <= 12

def test_general_store(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "General Store")
        assert coins == env.player_coins(player)

        env.current_player = player
        
        coins = env.player_coins(player)
        env._activate_card(player, "General Store")
        assert coins + 2 == env.player_coins(player)

        env.add_card(player, "Harbor")
        coins = env.player_coins(player)
        env._activate_card(player, "General Store")
        assert coins + 2 == env.player_coins(player)

        env.add_card(player, "Moon Tower")
        coins = env.player_coins(player)
        env._activate_card(player, "General Store")
        assert coins == env.player_coins(player)

def test_bakery(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Bakery")
        assert coins == env.player_coins(player)

        env.current_player = player
        
        coins = env.player_coins(player)
        env._activate_card(player, "Bakery")
        assert coins + 1 == env.player_coins(player)

def test_demolition_company(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Demolition Company")
        assert coins == env.player_coins(player)

        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Demolition Company")
        assert coins == env.player_coins(player)

        env.add_card(player, "Harbor")
        coins = env.player_coins(player)
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env.player_coins(player)

        assert env.state_dict()["player_info"][player]["cards"]["Harbor"] == 0

        env.add_card(player, "Harbor")
        env.add_card(player, "Train Station")
        coins = env.player_coins(player)
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env.player_coins(player)

        assert env.state_dict()["player_info"][player]["cards"]["Harbor"] == 0
        assert env.state_dict()["player_info"][player]["cards"]["Train Station"] == 1

        coins = env.player_coins(player)
        env._activate_card(player, "Demolition Company")
        assert coins + 8 == env.player_coins(player)

        assert env.state_dict()["player_info"][player]["cards"]["Harbor"] == 0
        assert env.state_dict()["player_info"][player]["cards"]["Train Station"] == 0

def test_flower_shop(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Flower Shop")
        assert coins == env.player_coins(player)

        env.add_card(player, "Flower Orchard")
        env.add_card(player, "Flower Orchard")
        coins = env.player_coins(player)
        env._activate_card(player, "Flower Shop")
        assert coins + 2 == env.player_coins(player)

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Flower Shop")
        assert coins == env.player_coins(player)

def test_cheese_factory(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Cheese Factory")
        assert coins == env.player_coins(player)

        env.add_card(player, "Ranch")
        env.add_card(player, "Ranch")

        coins = env.player_coins(player)
        env._activate_card(player, "Cheese Factory")
        assert coins + 2 * 3 == env.player_coins(player)

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Cheese Factory")
        assert coins == env.player_coins(player)

def test_furniture_factory(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Furniture Factory")
        assert coins == env.player_coins(player)

        env.add_card(player, "Forest")
        env.add_card(player, "Forest")

        coins = env.player_coins(player)
        env._activate_card(player, "Furniture Factory")
        assert coins + 2 * 3 == env.player_coins(player)

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Furniture Factory")
        assert coins == env.player_coins(player)

def test_soda_bottling_plant(env):

    n_players = 0
    for player in env.state_dict()["player_info"].keys():
        n_players+=1
        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Soda Bottling Plant")
        assert coins == env.player_coins(player)

    for player in env.state_dict()["player_info"].keys():
        env.add_card(player, "Sushi Bar")
        env.add_card(player, "Café")
        
    for player in env.state_dict()["player_info"].keys():

        coins = env.player_coins(player)
        env._activate_card(player, "Soda Bottling Plant")
        if player == env.current_player:
            assert coins + n_players*2 == env.player_coins(player)

def test_fruit_and_vegetable_market(env):
    for player in env.state_dict()["player_info"].keys():
        env.current_player = player

        coins = env.player_coins(player)
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 == env.player_coins(player)

        env.add_card(player, "Flower Orchard")
        env.add_card(player, "Apple Orchard")

        coins = env.player_coins(player)
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins + 2 * 3 == env.player_coins(player)

        env.current_player = get_not_current_player(env.player_order, player)

        coins = env.player_coins(player)
        env._activate_card(player, "Fruit and Vegetable Market")
        assert coins == env.player_coins(player)

def test_sushi_bar(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Sushi Bar")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]


    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 20

    for _player in env.state_dict()["player_info"].keys():
        env.add_card(_player, "Harbor")

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Sushi Bar")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + 3 == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player - 3 == env.state_dict()["player_info"][env.current_player]["coins"]


def test_cafe(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Café")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + 1 == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player - 1 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_french_restaurant(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "French Restaurant")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]


    env.add_card(env.current_player, "Harbor")
    env.add_card(env.current_player, "Airport")

    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 20

    for _player in env.state_dict()["player_info"].keys():
        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "French Restaurant")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + 5 == env.state_dict()["player_info"][_player]["coins"] # french restaurant requires to transfer 5 coins but players in this test have only 3 coins.
            assert coins_current_player - 5 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_pizza_joint(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Pizza Joint")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + 1 == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player - 1 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_family_restaurant(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 10

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Family Restaurant")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + 2 == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player - 2 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_members_only_club(env):
    player = list(env.state_dict()["player_info"].keys())[0]
    env.current_player = player

    for _player in env.state_dict()["player_info"].keys():

        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Member's Only Club")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]


    env.add_card(env.current_player, "Harbor")
    env.add_card(env.current_player, "Airport")
    env.add_card(env.current_player, "Train Station")

    for _player in env.state_dict()["player_info"].keys():
        coins_player = env.state_dict()["player_info"][_player]["coins"]
        coins_current_player = env.state_dict()["player_info"][env.current_player]["coins"]

        env._activate_card(_player, "Member's Only Club")
        if _player == env.current_player:
            assert coins_player == env.state_dict()["player_info"][_player]["coins"]
            assert coins_current_player == env.state_dict()["player_info"][env.current_player]["coins"]
        else:
            assert coins_player + coins_current_player == env.state_dict()["player_info"][_player]["coins"]
            assert env.state_dict()["player_info"][env.current_player]["coins"] == 0

def test_stadium(env):
    coins = {}
    n_players = 0
    for player in env.state_dict()["player_info"].keys():
        n_players += 1
        coins[player] = env.player_coins(player)

    env._activate_card(env.current_player, "Stadium")

    for player in env.state_dict()["player_info"].keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env.player_coins(player)
        else:
            assert coins[player] - 2 == env.player_coins(player)

def test_publisher(env):
    coins = {}
    n_players = 0
    for player in env.state_dict()["player_info"].keys():
        n_players += 1
        coins[player] = env.player_coins(player)
        env.add_card(player, "Café")


    env._activate_card(env.current_player, "Publisher")

    for player in env.state_dict()["player_info"].keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env.player_coins(player)
        else:
            assert coins[player] - 2 == env.player_coins(player) # coins subtracted due to (default) bakery and Café.

def test_tax_office(env):
    env.current_player = list(env.state_dict()["player_info"].keys())[0]
    rich_player = list(env.state_dict()["player_info"].keys())[1]
    env.state[env._state_indices["player_info"][rich_player]["coins"]] = 23

    coins = {}
    for player in env.state_dict()["player_info"].keys():
        coins[player] = env.player_coins(player)

    env._activate_card(env.current_player, "Tax Office")
    for player in env.state_dict()["player_info"].keys():
        if player == env.current_player:
            assert coins[player] + 11 == env.player_coins(player)
        elif player == rich_player:
            assert coins[player] - 11 == env.player_coins(player)
        else:
            assert coins[player] == env.player_coins(player)

def test_tech_startup(env):
    coins = {}
    n_players = 0

    env.add_card(env.current_player, "Tech Startup")
    env.invest_in_tech_startup()
    env.invest_in_tech_startup()

    for player in env.state_dict()["player_info"].keys():
        n_players += 1
        coins[player] = env.player_coins(player)
    


    env._activate_card(env.current_player, "Tech Startup")
    for player in env.state_dict()["player_info"].keys():
        if player == env.current_player:
            assert coins[player] + (n_players-1)*2 == env.player_coins(player)
        else:
            assert coins[player] - 2 == env.player_coins(player)

def test_shopping_mall_box_icon(env):
    for card, info in env._card_info.items():
        if card == "General Store":
            payment = 2
        elif card == "Bakery":
            payment = 1
        elif card == "Flower Shop":
            env.add_card(env.current_player, "Flower Orchard")
            payment = 1
        else:
            break
        
        coins = env.state_dict()["player_info"][env.current_player]["coins"]
        env._activate_card(env.current_player, card)
        assert coins + payment == env.state_dict()["player_info"][env.current_player]["coins"]
        env.add_card(env.current_player, "Shopping Mall")
        coins = env.state_dict()["player_info"][env.current_player]["coins"]
        env._activate_card(env.current_player, card)
        assert coins + payment*2 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_shopping_mall_cup_icon(env):
    env.current_player = env._player_order[0]
    second_player = env._player_order[1]

    env.add_card(second_player, "Harbor")

    for card, info in env._card_info.items():
        for player in env.state_dict()["player_info"].keys():
            env.state[env._state_indices["player_info"][player]["coins"]] = 10

        if card == "Café" or card == "Pizza Joint":
            payment = 1
        elif card == "French Restaurant":
            env.add_card(env.current_player, "Harbor")
            env.add_card(env.current_player, "Airport")
            payment = 5
        elif card == "Family Restaurant":
            payment = 2
        else:
            break
        
        coins_second_player = env.state_dict()["player_info"][second_player].coins
        coins = env.state_dict()["player_info"][env.current_player]["coins"]
        env._activate_card(second_player, card)
        assert coins_second_player + payment == env.state_dict()["player_info"][second_player].coins
        assert coins - payment == env.state_dict()["player_info"][env.current_player]["coins"]

        for player in env.state_dict()["player_info"].keys():
            env.state[env._state_indices["player_info"][player]["coins"]] = 10

        env.add_card(env.current_player, "Shopping Mall")
        
        coins_second_player = env.state_dict()["player_info"][second_player].coins
        coins = env.state_dict()["player_info"][env.current_player]["coins"]
        env._activate_card(second_player, card)
        assert coins_second_player + (payment+1) == env.state_dict()["player_info"][second_player].coins
        assert coins - (payment+1) == env.state_dict()["player_info"][env.current_player]["coins"]


def test_amusement_park(env):
    with patch('random.randint', side_effect=[1,1,1,3]) as mock_random:
        # so that diceroll will be [1, 1], [1, 3]
        env.add_card(env.current_player, "Amusement Park")
        env.add_card(env.current_player, "Train Station")

        for _ in range(5):
            env.add_card(env.current_player, "Flower Orchard")


        coins = env.state_dict()["player_info"][env.current_player]["coins"]
        env.step("2 dice") # throws [1, 1] so can throw again
        env.step("2 dice") # throws [1, 3]

        # 1 for the bakery and 5 for each flower orchard
        assert coins + 1 + 5 == env.state_dict()["player_info"][env.current_player]["coins"]

def test_airport(env):
    player = env.current_player
    coins = env.player_coins(player)
    env._advance_stage()
    env.step("Build nothing")
    assert coins == env.player_coins(player)

    player = env.current_player
    env.add_card(player, "Airport")
    coins = env.player_coins(player)
    env._advance_stage()
    env.step("Build nothing")
    assert coins + 10 == env.player_coins(player)

def test_earn_income_order(env):
    env._player_order = [f"player {i}" for i in range(1,4)]
    env.current_player = env._player_order[0]
    second_player = env._player_order[1]

    # Test Secondary Industry after Restaurants
    env.add_card(second_player, "Café")
    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 0
    env.state[env._state_indices["player_info"][second_player]["coins"]] = 3

    env._earn_income(3)

    assert env.state_dict()["player_info"][env.current_player]["coins"] == 1 # from bakery
    assert env.state_dict()["player_info"][second_player]["coins"] == 3 # still 3 because Café could get money from current player because it had 0 coins
    
    # Test Primary Industry after Restaurants
    env.add_card(second_player, "Sushi Bar")
    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 0
    env.state[env._state_indices["player_info"][second_player]["coins"]] = 3

    env._earn_income(1)
    assert env.state_dict()["player_info"][env.current_player]["coins"] == 1 # from bakery
    assert env.state_dict()["player_info"][second_player]["coins"] == 4 # only 1 from Wheat Field (default card) because Sushi Bar could get money from current player because it had 0 coins

    # Test Major Establishments after Secondary Industry
    env.add_card(second_player, "Mackerel Boat")
    env.add_card(second_player, "Harbor")
    env.add_card(env.current_player, "Tax Office")
    env.state[env._state_indices["player_info"][second_player]["coins"]] = 9
    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 0

    env._earn_income(8)

    assert env.state_dict()["player_info"][second_player]["coins"] == 6 # got 3 from Mackerel Boat, then lost half through tax office of second player
    assert env.state_dict()["player_info"][env.current_player]["coins"] == 6 # got 12 / 2 from second player through tax office

def test_step(env):
    env._current_player_index = 0

    for i in range(len(env._player_order)):
        env.state[env._state_indices["marketplace"]["1-6"]["pos_0"]["card"]] = env._card_name_to_num["Ranch"]
        env.step("1 dice")
        env.step("Ranch")
        if i == len(env._player_order) - 1:
            assert env.current_player == env._player_order[0]
        else:
            assert env.current_player == env._player_order[i + 1]

def test_marketplace(env):
    n_cards = 0
    for card_name, info in env._card_info.items():
        if info["type"] != "Landmarks":
            if card_name in ["Wheat Field", "Bakery"]:
                n_cards += info["n_cards"] - env.n_players
            else:
                n_cards += info["n_cards"]


    for j in range(2):
        n_gotten_cards = 0
        for i in range(n_cards):
            for alley_name, alley in env._state_indices["marketplace"].items():
                for stand_name, stand in alley.items():
                    n_cards_in_stand = copy.deepcopy(env.state[stand["count"]])
                    if n_cards_in_stand > 0:
                        card_to_get = env._card_num_to_name[env.state[stand["card"]]]
                        env.remove_card_from_marketplace(card_to_get)
                        if n_cards_in_stand > 1:
                            assert env._card_num_to_name[env.state[stand["card"]]] == card_to_get
                            assert env.state[stand["count"]] == n_cards_in_stand - 1   
                        n_gotten_cards +=1

        
        for alley_name, alley in env._state_indices["marketplace"].items():
            for stand_name, stand in alley.items():
                assert env.state[stand["count"]] == 0
        assert n_gotten_cards == n_cards
        env.reset()
    
def test_gym_env_obs(gymenv):
    obs, _ = gymenv.reset()
    gymenv._env._advance_stage()
    obs1, _, _, _, _ = gymenv.step(gymenv._action_str_to_idx["Build nothing"])

    gymenv.set_state(copy.deepcopy(obs))
    gymenv._env._advance_stage()
    obs2, _, _, _, _ = gymenv.step(gymenv._action_str_to_idx["Build nothing"])
    
    assert np.array_equal(obs1, obs2)


def test_winner_random_policy(gymenv, random_agent):
    obs, info = gymenv.reset()
    for i in range(1000):
        action, prob_dist = random_agent.compute_action(obs)
        obs, reward, done, truncated, info = gymenv.step(action)
        
        if done:
            break
    assert done


def test_roll_dice_always_returns_6(gymenv):
    # makes sure that player 0 gets 1 coin in the first turn
    with patch('random.randint', return_value=2) as mock_random:
        init_state = gymenv.observation()
        init_state_deepcopy = copy.deepcopy(init_state)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 3

        obsa, reward, done, _, info = gymenv.step(0)
        obsa_deepcopy = copy.deepcopy(obsa)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 3
        assert gymenv.state_dict(obsa)["player_info"]["player 0"]["coins"] == 4

        assert not np.array_equal(init_state, obsa)
        assert not np.array_equal(init_state, gymenv.observation())
        assert np.array_equal(obsa, gymenv.observation())
        assert np.array_equal(init_state, init_state_deepcopy)

        # assert init_state != gymenv.state_dict()
        # assert np.array_equal(info["state"], gymenv.state_dict())
        # assert np.array_equal(init_state, init_state_deepcopy)

        gymenv.set_state(copy.deepcopy(init_state))
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 3

        obsb, reward, done, _, info = gymenv.step(0)
        obsb_deepcopy = copy.deepcopy(obsb)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 3
        assert gymenv.state_dict(obsb)["player_info"]["player 0"]["coins"] == 4

        assert np.array_equal(obsa, obsb)

        assert np.array_equal(obsa_deepcopy, obsb_deepcopy)
    

def test_state_dict_and_back(gymenv):
    state_dict = gymenv.state_dict()
    state_array = gymenv.state_dict_to_array(state_dict)
    assert np.array_equal(state_array, gymenv._env.state)
    gymenv.set_state(state_array)
    assert state_dict == gymenv.state_dict()