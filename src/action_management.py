# ---- Regular Packages ----
from math import ceil
import numpy as np


# --------------------------------------------------------
# get_possible_actions function
# --------------------------------------------------------

def get_possible_actions(last_bet, total_dice):
    # return every legal action from last bet
    quantity, value = last_bet
    liar = [-1, -1]
    exact = [0, 0]
    challenge_last_bet = [liar, exact]

    # Case of bet = [1, 0] corresponding to initialization of a round.
    if value == 0:
        challenge_last_bet = []

    # Bet is on wilds
    if value == 1:
        upper_values = [[i, j] 
                        for i 
                        in range(2*quantity+1, total_dice+1) 
                        for j in range(2, 7)]
        upper_quantities = [[q, 1] 
                            for q 
                            in range(quantity + 1, total_dice + 1)]
        possible_actions = challenge_last_bet + upper_quantities + upper_values 

    # Bet is on non wilds values. max(value, 1) is here in case value = 0 (first bet).
    else:
        upper_values = [[quantity, v] 
                        for v 
                        in range(max(value, 1) + 1, 7)]
        upper_quantities = [[q, v] 
                            for q 
                            in range(quantity + 1, total_dice + 1) 
                            for v in range(2, 7)]
        wilds = [[i, 1] 
                   for i 
                   in range(int(ceil(quantity/2)), total_dice+1)]
        possible_actions = challenge_last_bet + wilds + upper_values + upper_quantities

    return possible_actions


# --------------------------------------------------------
# Masking functions
# --------------------------------------------------------

# /!\ Note : This function doesn't use get_possible_actions 
# but re-build action lists in order not to iterate over the action list.
# This implementation should be faster than using get_possible_actions.

def get_legal_actions_indices(last_bet, total_dice, action_dict):
    # Returns the indices of legal actions.
    
    quantity, value = last_bet
    liar_index = action_dict[tuple([-1, -1])]
    exact_index = action_dict[tuple([0, 0])]
    challenge_last_bet_indices = [liar_index, exact_index]

    # Case of bet = [1, 0] corresponding to initialization of a round.
    if value == 0:
        challenge_last_bet_indices = []

    # Bet is on wilds
    if value == 1:
        upper_values_indices = [action_dict[tuple([i, j])] for i in range(2*quantity+1, total_dice+1) for j in range(2, 7)]
        upper_quantities_indices = [action_dict[tuple([q, 1])] for q in range(quantity + 1, total_dice + 1)]
        possible_actions_indices = challenge_last_bet_indices + upper_quantities_indices + upper_values_indices 

    # Bet is on non wilds values. max(value, 1) is here in case value = 0 (first bet).
    else:
        upper_values_indices = [action_dict[tuple([quantity, v])] for v in range(max(value, 1) + 1, 7)]
        upper_quantities_indices = [action_dict[tuple([q, v])] for q in range(quantity + 1, total_dice + 1) for v in range(2, 7)]
        wilds_indices = [action_dict[tuple([i, 1])] for i in range(int(ceil(quantity/2)), total_dice+1)]
        possible_actions_indices = challenge_last_bet_indices + wilds_indices + upper_values_indices + upper_quantities_indices

    return possible_actions_indices

def get_legal_actions_mask(last_bet, total_dice, action_dict):

    possible_actions_indices = get_legal_actions_indices(last_bet, total_dice, action_dict)
    mask = np.zeros(len(action_dict), dtype=bool)
    mask[possible_actions_indices] = True
    return mask