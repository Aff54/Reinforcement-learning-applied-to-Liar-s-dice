# ---- Regular Packag ----
import random

# ---- Custom packages ----
from .probability_analysis import bet_conditional_probability, bet_exact_probability
from .action_management import get_possible_actions


# --------------------------------------------------------
# Agent functions
# --------------------------------------------------------


# ---- Simple agents definition ----
def agent_random(last_bet, 
                 total_dice,
                 player_hand = [],
                 verbose = False):
    """ Return a random pick among available actions.

    Args:
        last_bet (list): last bet to be played on.
        total_dice (int): Number of total dice in current game. 
        player_hand (list, optional): Hand of player. Defaults to []. Not used
        but kept so that the function as the same input as other agent functions.
        verbose (bool, optional): If True: prints what the agent chooses. 
        Defaults to False.

    Returns:
        _type_: bet chosen by the agent. Format: [quantity, value].
    """
    
    possible_actions = get_possible_actions(last_bet, total_dice)
    action = random.choice(possible_actions)

    if verbose:
        if action == [-1, -1]:
            print("agent calls liar")
        elif action == [0, 0]:
            print("agent calls exact")
        else:
            print(f"agent outbids with: {action}")

    return action


def agent_max_probability(last_bet,
                          total_dice,
                          player_hand,
                          verbose = False):
    """ Agent that returns the action with the highest probability of being 
    correct.

    Args:
        last_bet (list): last bet to be played on.
        total_dice (int): Number of total dice in current game. 
        player_hand (_type_): Hand of player.
        verbose (bool, optional): If True: prints what the agent chooses. 
        Defaults to False.

    Returns:
        _type_: bet chosen by the agent. Format: [quantity, value].
    """
    
    possible_actions = get_possible_actions(last_bet, total_dice)
    quantity, value = last_bet

    # -- Computing probabilities --
    # Probabilities of current bet outcomes.
    if value != 0: 
        liar_probability = 1 - bet_conditional_probability(last_bet,
                                                           total_dice,
                                                           player_hand)
        exact_probability = bet_exact_probability(last_bet,
                                                  total_dice,
                                                  player_hand)
        challenge_last_bet_probabilities = [liar_probability,
                                            exact_probability]
        # Probabilities for each possible outbid.
        bet_probabilities = [
            bet_conditional_probability(bet, total_dice, player_hand) 
            for bet in possible_actions[2:]
        ]
    else:
        challenge_last_bet_probabilities = []
        # Probabilities for each possible outbid.
        bet_probabilities = [
            bet_conditional_probability(bet, total_dice, player_hand) 
            for bet in possible_actions
        ]
    
    # Combine all probabilities
    possible_actions_probabilities = challenge_last_bet_probabilities + bet_probabilities

    max_probability = max(possible_actions_probabilities)
    action = [action 
              for action, probability 
              in zip(possible_actions, possible_actions_probabilities) 
              if probability == max_probability][-1]

    if verbose:
        if action == [-1, -1]:
            print("agent calls liar")
        elif action == [0, 0]:
            print("agent calls exact")
        else:
            print(f"agent outbids with: {action}")
        
    return action


def agent_min_probability(last_bet,
                          total_dice,
                          player_hand,
                          threshold = 0.5,
                          verbose = False):
    """Agent that tries to play the action with the lowest probability of being
       true while also being above given threshold.

    Args:
        last_bet (list): last bet to be played on.
        total_dice (int): Number of total dice in current game. 
        player_hand (_type_): Hand of player.
        threshold (float, optional): Threshold for minimal probability. 
        Defaults to 0.5.
        verbose (bool, optional): If True: prints what the agent chooses. 
        Defaults to False.

    Returns:
        _type_: bet chosen by the agent. Format: [quantity, value].
    """
    
    possible_actions = get_possible_actions(last_bet, total_dice)
    quantity, value = last_bet

    # -- Computing probabilities --
    # Probabilities of current bet outcomes.
    if value != 0: # Checks if it is the first bet of the game.
        liar_probability = 1 - bet_conditional_probability(last_bet,
                                                           total_dice,
                                                           player_hand)
        exact_probability = bet_exact_probability(last_bet,
                                                  total_dice,
                                                  player_hand)
        challenge_last_bet_probabilities = [liar_probability, 
                                            exact_probability]
        # Probabilities for each possible outbid.
        bet_probabilities = [
            bet_conditional_probability(bet, total_dice, player_hand) 
            for bet in possible_actions[2:]
        ]
    else:
        challenge_last_bet_probabilities = []
        # Probabilities for each possible outbid.
        bet_probabilities = [
            bet_conditional_probability(bet, total_dice, player_hand) 
            for bet in possible_actions
        ]
    
    # Combine all probabilities
    possible_actions_probabilities = challenge_last_bet_probabilities + bet_probabilities
    
    probabilities_above_threshold = [probability 
                                     for probability 
                                     in possible_actions_probabilities
                                     if probability > threshold]

    if probabilities_above_threshold:
        min_probability = min(probabilities_above_threshold)
        action = [action 
                  for action, probability 
                  in zip(possible_actions, 
                         possible_actions_probabilities) 
                  if probability == min_probability][-1]
    else:
        max_probability = max(possible_actions_probabilities)
        action = [action 
                  for action, probability 
                  in zip(possible_actions, 
                         possible_actions_probabilities) 
                  if probability == max_probability][-1]

    if verbose:
        if action == [-1, -1]:
            print("agent calls liar")
        elif action == [0, 0]:
            print("agent calls exact")
        else:
            print(f"agent outbids with: {action}")
        
    return action