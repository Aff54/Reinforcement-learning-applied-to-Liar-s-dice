# ---- Regular Packages ----
import numpy as np
from tqdm import tqdm
from collections import namedtuple

# ---- Custom Packages ----
from ..game import GameRL

# --------------------------------------------------------
# Functions for simulating games
# --------------------------------------------------------

def test_agents(max_dice, 
                players_dict, 
                n_simulation, 
                reward_dict, 
                RLAgent_class,
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
            if isinstance(player_object, RLAgent_class): # player_object is an Agent class instance.

                state = game.get_state()
                action = player_object.select_action(last_bet = last_bet, 
                                                     total_dice = n_dice, 
                                                     state = state)
                #print(f"RL agent plays {action} in state {state}")

            else: # player_object is a deterministic function.
                action = player_object(last_bet = tuple(last_bet),
                                       total_dice = n_dice,
                                       player_hand = tuple(turn_player_hand),
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


def simulate_games(max_dice, 
                players_dict, 
                n_simulation, 
                RLAgent_class,
                tqdm_display = False,
                verbose = False):
    
    player_number = len(players_dict)
    rankings = np.zeros((n_simulation, player_number))
    transition_format = ('player_id', 'last_bet', 'player_hand', 'action', 'outcome', 'is_out')
    Transition = namedtuple('Transition', transition_format)
    histories = []
    game_index = 0

    # ---- Simulating games ----
    for i in tqdm(range(n_simulation), disable = not tqdm_display):

        game_history = []
        game_index += 1

        # Initialising game.
        game = GameRL(player_number = player_number,
                           max_dice = max_dice)
        
        if verbose:
            print(f" -- Game n°{game_index}")

        # Playing the game.
        while not game.game_over:
            
            n_dice, last_bet, turn_player, turn_player_hand = game.get_turn_info()

            # Seperating deterministic functions VS RL agents
            player_object = players_dict[turn_player]
            if isinstance(player_object, RLAgent_class): # player_object is an Agent class instance.

                state = game.get_state()
                action = player_object.select_action(last_bet = last_bet, 
                                                     total_dice = n_dice, 
                                                     state = state)
                if verbose:
                    print(f"RL agent plays {action} in state {state}")

            else: # player_object is a deterministic function.
                action = player_object(last_bet = last_bet,
                                       total_dice = n_dice,
                                       player_hand = turn_player_hand,
                                       verbose = verbose)
                
            # Taking action.
            bet_outcome = game.make_a_bet(action, 
                                          verbose = verbose)
            
            # Current bet is a challenge
            if bet_outcome != 2:


                current_player_is_out = False
                former_transition =  game_history.pop()

                # Current player called liar and was right
                if bet_outcome == 0:
                    last_player_outcome = -1

                    # Checking if last player just went out of the game.
                    if former_transition.player_id not in game.active_players:
                        last_player_is_out = True
                    else:
                        last_player_is_out = False

                # Current player called liar or exact and was wrong
                elif bet_outcome == -1:
                    last_player_outcome = 4
                    last_player_is_out = False

                    # Checking if current player just went out of the game.
                    if turn_player not in game.active_players:
                        current_player_is_out = True
                    
                
                elif bet_outcome == 1:
                    last_player_outcome = 5
                    last_player_is_out = False

                
                updated_transition = Transition(former_transition.player_id, 
                                                former_transition.last_bet, 
                                                former_transition.player_hand,
                                                former_transition.action,
                                                last_player_outcome,
                                                last_player_is_out)
                game_history.append(updated_transition)

            # Action was not a challenge
            else:
                current_player_is_out = False

            current_transition = Transition(turn_player, 
                                    last_bet, 
                                    turn_player_hand, 
                                    action,
                                    bet_outcome,
                                    current_player_is_out)

            game_history.append(current_transition) 


        # Saving results
        histories.append(game_history)
        rankings[i, :] = game._ranking

    return rankings, histories





# --------------------------------------------------------
# Functions for computing score metrics
# --------------------------------------------------------

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