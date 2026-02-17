# --------------------------------------------------------
# packages
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from math import ceil
import unittest

# --------------------------------------------------------
# tested functions
# --------------------------------------------------------

from src.probability_analysis import bet_probability, bet_conditional_probability, bet_exact_probability

# --------------------------------------------------------
# class definition
# --------------------------------------------------------

# ---- Testing bet_probability ----
class Test_bet_probability(unittest.TestCase):
    
    def test_single_die_probabilities(self):
        # Test if probability values are good for a single dice.
        n_dice = 1
        expected_map = {1: 1/6, 2: 1/3, 3: 1/3, 4: 1/3, 5: 1/3, 6: 1/3}
        for value, expected in expected_map.items():
            prob = bet_probability((1, value), n_dice)
            self.assertAlmostEqual(prob, expected, places=8,
                                       msg=f"value={value}: got {prob}, expected {expected}")

    def test_edge_cases(self):
        n_dice = 1

        # -- Values out of range(1, 7). --
        invalid_values = [0, 7]
        for value in invalid_values:
            with self.assertRaises(ValueError) as context:
                bet_probability([1, value], n_dice)
            self.assertIn("Value must be in {1..6}.", str(context.exception),
                          msg=f"Failed for value={value}")

        # -- Wrong n_dice values. --
        invalid_n_dices = [-1, 0]
        for n_dice in invalid_n_dices:
            with self.assertRaises(ValueError) as context:
                bet_probability([1, 1], n_dice)
            self.assertIn("Total_dice must be strictly positive.", str(context.exception),
                          msg=f"Failed for value={n_dice}")

        # -- High and low quantities management. --
        n_dice = 5
        # Case 1: non-strictly positive values
        test_quantities = [-1, 0]
        for quantity in test_quantities:
            probability = bet_probability([quantity, 2], n_dice)
            self.assertEqual(probability, 1)
        # Case 2: quantity > n_dice
        probability = bet_probability([6, 2], n_dice)
        self.assertEqual(probability, 0)


# ---- Testing bet_conditional_probability ----
class Test_bet_conditional_probability(unittest.TestCase):

    def test_without_player_hand(self):
        # Testing that calling the conditional probability without the player's hand is the same as the non conditional probability.
        n_dice = 1
        for v in range(1, 7):
            self.assertEqual(bet_probability([1, v], n_dice), 
                             bet_conditional_probability([1, v], n_dice, []))

        n_dice = 10
        self.assertEqual(bet_probability([7, 1], n_dice), 
                         bet_conditional_probability([7, 1], n_dice, []))
        self.assertEqual(bet_probability([4, 3], n_dice), 
                         bet_conditional_probability([4, 3], n_dice, []))
        self.assertEqual(bet_probability([10, 6], n_dice), 
                         bet_conditional_probability([10, 6], n_dice, []))

    def test_proper_probability_calculation(self):
        # Testing if probability prior to knowing someone's hand are rightly computed.

        # Case 1: no wild in current player's hand.
        n_dice = 11
        last_bet = [4, 3]
        player_hand = [2, 3, 5]
        computed_probability = bet_conditional_probability(bet = last_bet, 
                                                           total_dice = n_dice, 
                                                           hand = player_hand)
        expected_probability = bet_probability(bet = [3, 3], 
                                               total_dice = n_dice - 3)
        self.assertEqual(computed_probability, 
                         expected_probability)

        # Case 2: current player has wilds in hand.
        n_dice = 8
        last_bet = [5, 2]
        player_hand = [1, 1, 3]
        computed_probability = bet_conditional_probability(bet = last_bet, 
                                                   total_dice = n_dice, 
                                                   hand = player_hand)
        expected_probability = bet_probability(bet = [3, 2], 
                                               total_dice = n_dice - 3)
        self.assertEqual(computed_probability, 
                         expected_probability)

        # Case 3: current player's hand satisfies asked bet -> probability 1.
        n_dice = 6
        last_bet = [3, 5]
        player_hand = [4, 1, 5, 5, 5]
        computed_probability = bet_conditional_probability(bet = last_bet, 
                                                           total_dice = n_dice, 
                                                           hand = player_hand)
        self.assertEqual(computed_probability, 1)

        # Case 4: current player's hand makes asked bet impossible -> probability 0.
        n_dice = 4
        last_bet = [2, 1]
        player_hand = [2, 3, 4]
        computed_probability = bet_conditional_probability(bet = last_bet, 
                                                           total_dice = n_dice, 
                                                           hand = player_hand)
        self.assertEqual(computed_probability, 0)


    def test_edge_cases(self):
        
        # Case 1: current player's hand is larger than the total number of dice.
        n_dice = 5
        player_hand = [i for i in range(1, 7)]
        with self.assertRaises(ValueError) as context:
            bet_conditional_probability([1, 3], n_dice, player_hand)
        self.assertIn("Hand length cannot exceed or equal total_dice.", str(context.exception),
                          msg=f"Failed for value={n_dice}")

        # Case 2: invalid values in the input hand.
        n_dice = 5
        player_hand = [0]
        with self.assertRaises(ValueError) as context:
            bet_conditional_probability([1, 3], n_dice, player_hand)
        self.assertIn("Values in input hand must be in {1...6}.", str(context.exception),
                          msg=f"Failed for value={n_dice}")
        

# ---- Testing bet_exact_probability ----
class Test_bet_exact_probability(unittest.TestCase):

    def test_proper_probability_calculation(self):

        # Case 1: no hand is given -> there is no prior knowledge.
        test_cases = [
            # (n_dice, quantity, value, p)
            (12, 7, 2, 1/3),  # non-wild
            (6,  4, 1, 1/6),  # wild
        ]
    
        for n_dice, quantity, value, p in test_cases:
            with self.subTest(n_dice=n_dice, quantity=quantity, value=value):
                bet = [quantity, value]
                calculated_probability = bet_exact_probability(bet, n_dice, [])
                expected_probability = binom(n_dice, p).pmf(quantity)
                self.assertEqual(calculated_probability, expected_probability)

        # Case 2: hands are given -> we have partial knowledge about dice.
        test_cases = [
            # (n_dice, quantity, value, [hand], p)
            (6, 2, 5, [3, 5, 2], 1/3),    # betting on non-wilds
            (9, 7, 1, [1, 4, 2, 1], 1/6), # betting on wilds
            (8, 2, 3, [1, 3, 5], 1/3)     # having wilds in hand
        ]
        for n_dice, quantity, value, player_hand, p in test_cases:
            with self.subTest(n_dice=n_dice, quantity=quantity, player_hand = player_hand,value=value):
                bet = [quantity, value]
                discount = player_hand.count(value) if value == 1 else player_hand.count(value) + player_hand.count(1) 
                calculated_probability = bet_exact_probability(bet, n_dice, player_hand)
                expected_probability = binom(n_dice - len(player_hand), p).pmf(quantity - discount)
                self.assertEqual(calculated_probability, expected_probability)

    def test_edge_cases(self):

        # Case 1: current player's hand is larger than the total number of dice.
        n_dice = 7
        player_hand = [1 for i in range(1, 12)]
        with self.assertRaises(ValueError) as context:
            bet_exact_probability([1, 3], n_dice, player_hand)
        self.assertIn("Hand length cannot exceed or equal total_dice.", str(context.exception),
                          msg=f"Failed for value={n_dice}")

        # Case 2: invalid values in the input hand.
        n_dice = 3
        invalid_values = [-2, 7]
        for value in invalid_values:
            player_hand = [value]
            with self.assertRaises(ValueError) as context:
                bet_exact_probability([1, 3], n_dice, player_hand)
            self.assertIn("Values in input hand must be in {1...6}.", str(context.exception),
                              msg=f"Failed for value={n_dice}")

        # Case 3: invalid total dice value.
        invalid_n_dices = [-1, 0]
        for n_dice in invalid_n_dices:
            with self.assertRaises(ValueError) as context:
                bet_exact_probability([1, 1], n_dice)
            self.assertIn("Total_dice must be strictly positive.", str(context.exception),
                          msg=f"Failed for value={n_dice}")

        # Case 4: invalid bets: non postive quantity and value out of range.
        n_dice = 4
        bet = [-1, 5]
        with self.assertRaises(ValueError) as context:
            bet_exact_probability(bet, n_dice, [])
        self.assertIn("Quantity must be strictly postivie.", str(context.exception),
                          msg=f"Failed for value={n_dice}")

        bet = [2, 7]
        with self.assertRaises(ValueError) as context:
            bet_exact_probability(bet, n_dice, [])
        self.assertIn("Value must be in {1..6}.", str(context.exception),
                          msg=f"Failed for value={n_dice}")
        

if __name__ == "__main__":
    unittest.main(verbosity=2)
