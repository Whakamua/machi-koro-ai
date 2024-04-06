import pytest
from env_machi_koro_2 import MachiKoro2, GymMachiKoro2
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
    return MachiKoro2(n_players=n_players)

@pytest.fixture
def gymenv(n_players):
    return GymMachiKoro2(n_players=n_players)

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
        assert env.player_icon_count(player, "Landmark") == n_landmarks
        for landmark in env._landmarks:
            env.add_card(player, landmark)
            n_landmarks += 1
            assert env.player_icon_count(player, "Landmark") == n_landmarks
            

def test_env_init(env):
    for player, info in env.state_dict()["player_info"].items():

        assert info["cards"] == {
            'Sushi Bar': 0, 'Wheat Field': 0, 'Vineyard': 0, 'Bakery': 0, 'Café': 0,
            'Convenience Store': 0, 'Flower Garden': 0, 'Forest': 0, 'Flower Shop': 0,
            'Stadium': 0, 'Corn Field': 0, 'Furniture Factory': 0, 'Hamburger Stand': 0,
            'Shopping District': 0, 'Winery': 0, 'Family Restaurant': 0, 'Apple Orchard': 0,
            'Food Warehouse': 0, 'Mine': 0, 'Amusement Park': 0, 'Airport': 0, 'Museum': 0,
            'Exhibit Hall': 0, 'TV Station': 0, 'Radio Tower': 0, 'Temple': 0, 'Charterhouse': 0,
            'Park': 0, 'Soda Bottling Plant': 0, 'Forge': 0, 'French Restaurant': 0,
            'Observatory': 0, 'Publisher': 0, 'Farmers Market': 0, 'Shopping Mall': 0,
            'Tech Startup': 0, 'Loan Office': 0, 'Launch Pad': 0
        }
        assert env.player_coins(player) == 5
        assert env.player_icon_count(player, "Landmark") == 0
    
def test_city_hall(env):
    for player, player_info in env.state_dict()["player_info"].items():
        env.player_coins(player) == 0

    for player in env.player_order:
        env._current_player_index = player
        env._diceroll("1 dice")
        assert env.player_coins(player) >= 1


def test_restaurants(env):
    restaurants = set(["Sushi Bar", "Café", "Hamburger Stand", "Family Restaurant"])
    covered_restaurants = set()
    for card, info in env._card_info.items():
        if card in restaurants:
            covered_restaurants.add(card)
            for player in env.player_order:
                env.current_player = player
                env.set_player_coins(player, info["transfer"]*len(env.next_players))
                for other_player in env.next_players:
                    env.set_player_coins(other_player, 0)
                    env._activate_card(other_player, card)
                    assert env.player_coins(other_player) == info["transfer"]
                assert env.player_coins(player) == 0
    assert restaurants == covered_restaurants

def test_primary_industries(env):
    primary_industries = set(["Wheat Field", "Vineyard", "Flower Garden", "Forest", "Corn Field", "Apple Orchard", "Mine"])
    covered_primary_industries = set()
    for card, info in env._card_info.items():
        if card in primary_industries:
            covered_primary_industries.add(card)
            for player in env.player_order:
                env.current_player = player
                for _player in env.player_order:
                    env.set_player_coins(_player, 0)
                for _player in env.player_order:
                    env._activate_card(_player, card)
                    assert env.player_coins(_player) == info["transfer"]

    assert primary_industries == covered_primary_industries

def test_secondary_industries(env):
    secondary_industries = set(["Bakery", "Convenience Store", "Flower Shop", "Furniture Factory", "Winery", "Food Warehouse"])
    covered_secondary_industries = set()
    env_state = copy.deepcopy(env.state)
    for card, info in env._card_info.items():
        if card in secondary_industries:
            covered_secondary_industries.add(card)
            for player in env.player_order:
                env.state = copy.deepcopy(env_state)
                env.current_player = player
                for _player in env.player_order:
                    env.set_player_coins(_player, 0)
                    if info["icon"] == "Combo":
                        if info["combo"] == "Flower":
                            env.add_card(_player, "Flower Garden")
                            env.add_card(_player, "Flower Garden")
                        if info["combo"] == "Cup":
                            env.add_card(_player, "Sushi Bar")
                            env.add_card(_player, "Sushi Bar")
                        if info["combo"] == "Gear":
                            env.add_card(_player, "Mine")
                            env.add_card(_player, "Mine")
                        if info["combo"] == "Fruit":
                            env.add_card(_player, "Vineyard")
                            env.add_card(_player, "Vineyard")
                for _player in env.player_order:
                    env._activate_card(_player, card)
                    if _player == player:
                        assert env.player_coins(_player) == info["transfer"] * (1+int(info["icon"] == "Combo"))
                    else:
                        assert env.player_coins(_player) == 0
    assert secondary_industries == covered_secondary_industries


def test_earn_income_order(env):
    env._player_order = [f"player {i}" for i in range(1,4)]
    env.current_player = env._player_order[0]
    second_player = env._player_order[1]

    # Test Secondary Industry after Restaurants
    env.add_card(env.current_player, "Bakery")
    env.add_card(second_player, "Café")
    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 0
    env.state[env._state_indices["player_info"][second_player]["coins"]] = 3

    env._earn_income(3)

    assert env.state_dict()["player_info"][env.current_player]["coins"] == 2 # from bakery
    assert env.state_dict()["player_info"][second_player]["coins"] == 3 # still 3 because Café could not get money from current player because it had 0 coins
    
    # Test Primary Industry after Restaurants
    env.add_card(env.current_player, "Wheat Field")
    env.add_card(second_player, "Wheat Field")
    env.add_card(second_player, "Sushi Bar")
    env.state[env._state_indices["player_info"][env.current_player]["coins"]] = 0
    env.state[env._state_indices["player_info"][second_player]["coins"]] = 3

    env._earn_income(1)
    assert env.state_dict()["player_info"][env.current_player]["coins"] == 1 # from wheat field
    assert env.state_dict()["player_info"][second_player]["coins"] == 4 # only 1 from wheat field, nothing from Sushi Bar because current player had 0 coins

    # Test Major Establishments after Secondary Industry
    env.add_card(second_player, "Corn Field")
    env.add_card(env.current_player, "Corn Field")
    env.add_card(env.current_player, "Stadium")
    for player in env._player_order:
        env.state[env._state_indices["player_info"][player]["coins"]] = 0

    env._earn_income(7)

    assert env.state_dict()["player_info"][second_player]["coins"] == 0 # got 3 from Corn Field, then it through Stadium of current playyer
    assert env.state_dict()["player_info"][env.current_player]["coins"] == 6 # got 3 from Corn Field, then 3 from Stadium

def test_step(env):
    env._current_player_index = 0

    for i in range(len(env._player_order)):
        env.state[env._state_indices["marketplace"]["1-6"]["pos_0"]["card"]] = env._card_name_to_num["Wheat Field"]
        env.step("1 dice")
        env.step("Wheat Field")
        if i == len(env._player_order) - 1:
            assert env.current_player == env._player_order[0]
        else:
            assert env.current_player == env._player_order[i + 1]

def test_marketplace(env):
    n_cards = 0
    for card_name, info in env._card_info.items():
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
    gymenv._env._advance_stage(info={"another_turn": False})
    obs1, _, _, _, _ = gymenv.step(gymenv._action_str_to_idx["Build nothing"])

    gymenv.set_state(copy.deepcopy(obs))
    gymenv._env._advance_stage(info={"another_turn": False})
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
        gymenv._env.add_card("player 0", "Wheat Field")
        init_state = gymenv.observation()
        init_state_deepcopy = copy.deepcopy(init_state)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 5

        obsa, reward, done, _, info = gymenv.step(0)
        obsa_deepcopy = copy.deepcopy(obsa)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 5
        assert gymenv.state_dict(obsa)["player_info"]["player 0"]["coins"] == 6

        assert not np.array_equal(init_state, obsa)
        assert not np.array_equal(init_state, gymenv.observation())
        assert np.array_equal(obsa, gymenv.observation())
        assert np.array_equal(init_state, init_state_deepcopy)

        # assert init_state != gymenv.state_dict()
        # assert np.array_equal(info["state"], gymenv.state_dict())
        # assert np.array_equal(init_state, init_state_deepcopy)

        gymenv.set_state(copy.deepcopy(init_state))
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 5

        obsb, reward, done, _, info = gymenv.step(0)
        obsb_deepcopy = copy.deepcopy(obsb)
        assert gymenv.state_dict(init_state)["player_info"]["player 0"]["coins"] == 5
        assert gymenv.state_dict(obsb)["player_info"]["player 0"]["coins"] == 6

        assert np.array_equal(obsa, obsb)

        assert np.array_equal(obsa_deepcopy, obsb_deepcopy)

def test_stadium(env):
    for player in env.player_order:
        env.set_player_coins(player, 3)
    env.add_card(env.current_player, "Stadium")
    env._activate_card(env.current_player, "Stadium")
    for player in env.player_order:
        if player == env.current_player:
            assert env.player_coins(player) == 3*len(env.player_order)
        else:
            assert env.player_coins(player) == 0


def test_shopping_district(env):
    for player in env.player_order:
        if player == env.current_player:
            env.set_player_coins(player, 10)
        else:
            env.set_player_coins(player, 11)

    env.add_card(env.next_players[0], "Shopping District")
    env._activate_card(env.next_players[0], "Shopping District")
    for player in env.player_order:
        if player == env.current_player:
            assert env.player_coins(player) == 10
        elif player == env.next_players[0]:
            assert env.player_coins(player) == 11 + 5*(len(env.player_order)-2)
        else:
            assert env.player_coins(player) == 6


def test_amusement_park(env):
    # makes sure that player 0 gets 1 coin in the first turn
    with patch('random.randint', return_value=1) as mock_random:
        current_player = copy.deepcopy(env.current_player)
        env.add_card(current_player, "Bakery")
        env.add_card(env.next_players[0], "Amusement Park")
        env.step("2 dice")
        env.step("Build nothing")
        assert env.current_player == current_player
        env.step("2 dice")
        env.step("Build nothing")
        assert env.current_player == current_player


def test_airport(env):
    env.add_card(env.current_player, "Airport")
    for _ in env.player_order:
        env.step("1 dice")
        env.step("Build nothing")
        env.player_coins(env.current_player) == 5


def test_museum(env):
    current_player = copy.deepcopy(env.current_player)
    for player in env.player_order:
        if player == current_player:
            env.set_player_coins(player, 12)
        else:
            env.set_player_coins(player, 6)
            env.add_card(player, "Airport")
            env.add_card(player, "Amusement Park")
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Museum"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    env.step("1 dice")
    env.step("Museum")
    for player in env.player_order:
        if player == current_player:
            assert env.player_coins(player) == 2*3*(len(env.player_order)-1)
        else:
            assert env.player_coins(player) == 0

def test_french_restaurant(env):
    current_player = copy.deepcopy(env.current_player)
    for player in env.player_order:
        if player == current_player:
            env.set_player_coins(player, 10)
        else:
            env.set_player_coins(player, 2)

    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["French Restaurant"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    env.step("1 dice")
    env.step("French Restaurant")
    for player in env.player_order:
        if player == current_player:
            assert env.player_coins(player) == 2*(len(env.player_order)-1)
        else:
            assert env.player_coins(player) == 0


def test_exhibit_hall(env):
    current_player = copy.deepcopy(env.current_player)
    next_player = copy.deepcopy(env.next_players[0])
    for player in env.player_order:
        if player == current_player:
            env.set_player_coins(player, 12)
        elif player == next_player:
            env.set_player_coins(player, 10)
        else:
            env.set_player_coins(player, 11)

    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Exhibit Hall"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    env.step("1 dice")
    env.step("Exhibit Hall")
    for player in env.player_order:
        if player == current_player:
            assert env.player_coins(player) == 5*(len(env.player_order)-2)
        elif player == next_player:
            assert env.player_coins(player) == 10
        else:
            assert env.player_coins(player) == 6


def test_observatory(env):
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Observatory"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    env.set_player_coins(env.current_player, 10)
    env.step("1 dice")
    env.step("Observatory")
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Launch Pad"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    env.set_player_coins(env.current_player, 40)
    env.step("1 dice")
    env.step("Launch Pad")


def test_loan_office(env):
    for player in env.player_order:
        if player != env.current_player:
            env.add_card(player, "Amusement Park")
    
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Loan Office"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    env.set_player_coins(env.current_player, 10)
    env.step("1 dice")
    env.step("Loan Office")

    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Amusement Park"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    env.set_player_coins(env.current_player, 14)
    env.step("1 dice")
    env.step("Amusement Park")


def test_publisher(env):
    current_player = copy.deepcopy(env.current_player)
    for player in env.player_order:
        if player != current_player:
            env.add_card(player, "Bakery")
            env.add_card(player, "Bakery")
            env.set_player_coins(player, 2)
        else:
            env.set_player_coins(player, 10)

    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Publisher"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    env.step("1 dice")
    env.step("Publisher")

    for player in env.player_order:
        if player != current_player:
            assert env.player_coins(player) == 0
        else:
            assert env.player_coins(player) == 2*(len(env.player_order)-1)


def test_tv_station(env):
    current_player = copy.deepcopy(env.current_player)
    for player in env.player_order:
        if player != current_player:
            env.add_card(player, "Café")
            env.add_card(player, "Café")
            env.set_player_coins(player, 2)
        else:
            env.set_player_coins(player, 12)

    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["TV Station"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    env.step("1 dice")
    env.step("TV Station")

    for player in env.player_order:
        if player != current_player:
            assert env.player_coins(player) == 0
        else:
            assert env.player_coins(player) == 2*(len(env.player_order)-1)


def test_farmers_market(env):
    env.add_card(env.current_player, "Wheat Field")
    env.add_card(env.next_players[0], "Farmers Market")
    env._activate_card(env.current_player, "Wheat Field")
    assert env.player_coins(env.current_player) == 5+2


def test_radio_tower(env):
    # makes sure that player 0 gets 1 coin in the first turn
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Radio Tower"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1
    current_player = copy.deepcopy(env.current_player)
    env.set_player_coins(env.current_player, 12)
    env.step("1 dice")
    env.step("Radio Tower")
    assert env.current_player == current_player
    env.step("1 dice")
    env.step("Build nothing")
    assert env.current_player != current_player


def test_shopping_mall(env):
    env.add_card(env.current_player, "Bakery")
    env.add_card(env.next_players[0], "Shopping Mall")
    env._activate_card(env.current_player, "Bakery")
    assert env.player_coins(env.current_player) == 5+3


def test_temple(env):
    # makes sure that player rolls doubles
    with patch('random.randint', return_value=1) as mock_random:
        current_player = copy.deepcopy(env.current_player)
        env.add_card(env.current_player, "Temple")
        env.step("2 dice")
        for player in env.player_order:
            if player == current_player:
                assert env.player_coins(player) == 5 + 2*(len(env.player_order)-1)
            else:
                assert env.player_coins(player) == 3


def test_charter_house(env):
    env.add_card(env.next_players[0], "Charterhouse")
    env.step("2 dice")

    for player in env.player_order:
        if player == env.current_player:
            assert env.player_coins(player) == 8
        else:
            assert env.player_coins(player) == 5


def test_launch_pad(env):
    env.set_player_coins(env.current_player, 45)
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Launch Pad"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    winner = env.step("2 dice")
    assert winner == False
    winner = env.step("Launch Pad")
    assert winner == True


def test_park(env):
    env.set_player_coins(env.current_player, 23)
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["card"]] = env._card_name_to_num["Park"]
    env.state[env._state_indices["marketplace"]["landmark"]["pos_0"]["count"]] = 1

    env.step("1 dice")
    env.step("Park")

    for player in env.player_order:
        assert env.player_coins(player) == np.ceil((11 + 5*(len(env.player_order)-1))/len(env.player_order))


def test_soda_bottling_plant(env):
    env.add_card(env.current_player, "Café")
    env.add_card(env.next_players[0], "Soda Bottling Plant")
    env._activate_card(env.next_players[0], "Café")
    assert env.player_coins(env.current_player) == 5-3


def test_forge(env):
    env.add_card(env.current_player, "Mine")
    env.add_card(env.next_players[0], "Forge")
    env._activate_card(env.current_player, "Mine")
    assert env.player_coins(env.current_player) == 5+7


def test_tech_startup(env):
    # makes sure that player rolls 12
    with patch('random.randint', return_value=6) as mock_random:
        env.add_card(env.next_players[0], "Tech Startup")
        env.step("2 dice")
        assert env.player_coins(env.current_player) == 5 + 8


def test_state_dict_and_back(gymenv):
    state_dict = gymenv.state_dict()
    state_array = gymenv.state_dict_to_array(state_dict)
    assert np.array_equal(state_array, gymenv._env.state)
    gymenv.set_state(state_array)
    assert state_dict == gymenv.state_dict()