# ---- Regular Packages ----
import numpy as np
from tqdm import tqdm

# ---- Custom Packages ----
from ..game import GameRL
from ..rl_agents import RLAgent

# --------------------------------------------------------
# Functions
# --------------------------------------------------------

def test_agents(max_dice, 
                players_dict, 
                n_simulation, 
                reward_dict, 
                tqdm_display = True, 
                verbose = False):
    
    """Make players in players_dict play n_simulation games.

    Returns:
        _type_: _description_
    """

    player_number = len(players_dict)
    rankings = np.zeros((n_simulation, player_number))
    rewards_per_game = np.zeros((n_simulation, player_number))
    # ---- Simulating games ----
    for i in tqdm(range(n_simulation), disable = not tqdm_display):
        # Initialising game.
        game = GameRL(player_number = player_number,
                           max_dice = max_dice)

        if verbose:
            print("---- New game ----")
        # Playing the game.
        while not game.game_over:
            
            n_dice, last_bet, turn_player, turn_player_hand = game.get_turn_info()
            if verbose:
                print(f"player {turn_player} plays with hand {turn_player_hand} ")

            # Seperating deterministic functions VS RL agents
            player_object = players_dict[turn_player]
            if isinstance(player_object, RLAgent): # player_object is an Agent class instance.

                state = game.get_state()
                action = player_object.select_action(last_bet = last_bet, 
                                                     total_dice = n_dice, 
                                                     state = state)
                #print(f"RL agent plays {action} in state {state}")

            else: # player_object is a deterministic function.
                action = player_object(last_bet = last_bet,
                                       total_dice = n_dice,
                                       player_hand = turn_player_hand,
                                       verbose = verbose)
            if verbose:
                print(action)
            # Taking action.
            bet_outcome = game.make_a_bet(action, 
                                          verbose = verbose)

            # Saving reward
            rewards_per_game[i, turn_player - 1] += reward_dict[bet_outcome]

        # Awarding game winner
        winner = game.active_players[0]
        rewards_per_game[i, winner - 1] += reward_dict[3]
        # Saving ranking
        rankings[i, :] = game._ranking

    average_reward_per_game = rewards_per_game.mean(axis=0)
    
    return rankings, average_reward_per_game


def get_player_score(ranking_array, player):
    """Returns the average place of player in ranking_array.

    Args:
        ranking_array (np.array): Array with i-row being the ranking of players 
        for the i-th game.
        player (int): Index of player we compute the score of.

    Returns:
        float: player's average place.
    """

    n_simulation, player_number = ranking_array.shape
    score = sum([(i+1)* np.count_nonzero(ranking_array[:, i] == player) 
                 for i in range(player_number)]) / n_simulation
    return score