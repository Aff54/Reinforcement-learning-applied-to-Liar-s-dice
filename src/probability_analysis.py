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
        # Counting wilds.
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
    

# --------------------------------------------------------
# Visual functions
# --------------------------------------------------------

def get_probability_matrix(total_dice,
                           player_hand = []):
    # return the naive probability of each bet based on the active player's hand
    probability_matrix = np.zeros((total_dice, 6))
    bet_quantities = np.arange(1, total_dice + 1) # possible quantities

    # creating probability matrix
    for j in range(6):
        value = j + 1
        probability_matrix[:, j] = np.array(
            [bet_conditional_probability([q, value], 
                                        total_dice, 
                                        player_hand) 
             for q in bet_quantities]
        )

    return probability_matrix


def plot_probability_matrix(probability_matrix,
                            last_bet = []):

    total_dice = probability_matrix.shape[0]
    bet_quantities = np.arange(1, total_dice + 1) # possible quantities
    copy_matrix = probability_matrix.copy()

    if last_bet:
        quantity, value = last_bet
        # cutting illegal bets
        if value != 1:
            for i in range(quantity-1):
                copy_matrix[i, 1:] = np.nan
            for j in range(1, value):
                copy_matrix[quantity-1, j] = np.nan
            for i in range(ceil(quantity / 2)-1):
                copy_matrix[i, 0] = np.nan
        else:
            for i in range(min(2*quantity, total_dice)):
                copy_matrix[i, 1:] = np.nan
            for i in range(quantity):
                copy_matrix[i, 0] = np.nan 

    # -- plotting --
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Use a colormap that supports NaN masking
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")  # gray for illegal bets
    
    # Mask NaN values
    im = ax.imshow(np.ma.masked_invalid(copy_matrix), 
                   aspect='auto', 
                   cmap=cmap, 
                   vmin = 0, 
                   vmax = 1)

    # - axes ticks -
    ax.set_xticks(range(6))
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_yticks(range(len(bet_quantities)))
    ax.set_yticklabels(bet_quantities)
    # Drawing grind arround cells.
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, total_dice, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)  # light separators
    ax.tick_params(which="minor", bottom=False, left=False)  # don't show minor ticks

    # Loop over data dimensions and create text annotations.
    if (last_bet and last_bet != [1,0]):
        for j in range(6):
            for i in np.arange(total_dice):
                if not np.isnan(copy_matrix[i, j]):
                    text = ax.text(j, i, round(copy_matrix[i, j], 3),
                                   ha="center", va="center", color="w")
                else:
                    if not (i == quantity -1 and j == value - 1):
                        text = ax.text(j, i, "illegal",
                                       ha="center", va="center", color="w")
        ax.text(value - 1, 
                quantity - 1, 
                "last bet", 
                ha="center", 
                va="center", 
                color="red")
        
    else:
        for j in range(6):
            for i in np.arange(total_dice):
                text = ax.text(j, i, round(copy_matrix[i, j], 3),
                               ha="center", va="center", color="w")

    ax.set_xlabel("Die value") 
    ax.set_ylabel("Quantity")
    ax.set_title(f"Probability of each bet being true prior to current player hand")
    fig.tight_layout()
    plt.show()


def plot_situation(total_dice,
                   player_hand, 
                   last_bet = []):
    # Plot the probability matrix associated with a given hand and the last bet
    probability_matrix = get_probability_matrix(total_dice, player_hand)
    plot_probability_matrix(probability_matrix, last_bet)