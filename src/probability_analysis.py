# --------------------------------------------------------
# packages
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from math import ceil

# --------------------------------------------------------
# functions
# --------------------------------------------------------

# ---- Probability calculation. ----
def bet_probability(bet,
                    total_dice):
    # Return the probability of a bet being true.
    quantity, value = bet
    # Exceptions.
    if not (1 <= value <= 6):
        raise ValueError("Value must be in {1..6}.")
    if total_dice <= 0:
        raise ValueError("Total_dice must be strictly positive.")
        
    # Probability of drawing given value.
    p = 1/6 if value == 1 else 1/3
    return binom(total_dice, p).sf(quantity-1)

def bet_conditional_probability(bet,
                                total_dice,
                                hand=[]):
    # Return the probability of a bet being true knowing a player's hand.

    # Expections.
    if len(hand) > total_dice:
        raise ValueError("Hand length cannot exceed or equal total_dice.")
    if not all([1 <= value <= 6 for value in hand]):
        raise ValueError("Values in input hand must be in {1...6}.")
        
    quantity, value = bet
    known_count = hand.count(value)
    if value > 1:
        known_count += hand.count(1)
    updated_bet = [quantity - known_count, value]
    updated_total_dice = total_dice - len(hand)
    return bet_probability(updated_bet, updated_total_dice)


def bet_exact_probability(bet, 
                          total_dice,
                          player_hand = []):
    # Return the probability of a bet being exact (exact quantity on the value).
    
    quantity, value = bet

    # Exceptions.
    if total_dice <= 0:
        raise ValueError("Total_dice must be strictly positive.")
    if len(player_hand) >= total_dice:
        raise ValueError("Hand length cannot exceed or equal total_dice.")
    if not (1 <= value <= 6):
        raise ValueError("Value must be in {1..6}.")
    if quantity <= 0:
        raise ValueError("Quantity must be strictly postivie.")
    if not all([1 <= value <= 6 for value in player_hand]):
        raise ValueError("Values in input hand must be in {1...6}.")

    # Probability of drawing given value.
    p = 1/6 if value == 1 else 1/3
    
    # If we know current player's hand.
    if player_hand:
        known_count = player_hand.count(value)
        # Counting pacoses.
        if value > 1:
            known_count += player_hand.count(1)
        # Updating parameters of the binomial distribution.
        updated_quantity = quantity - known_count
        updated_total_dice = total_dice - len(player_hand)
        
        if updated_quantity < 0:
            return np.float64(0.0)

        return binom(updated_total_dice, p).pmf(updated_quantity)
            
    # If we don't know current player's hand
    else:
        return binom(total_dice, p).pmf(quantity)