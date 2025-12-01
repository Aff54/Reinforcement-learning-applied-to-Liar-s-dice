import numpy as np
import random
from collections import deque # for ring buffer


# --------------------------------------------------------
# class definition
# --------------------------------------------------------

class Game:
    
    def __init__(self, player_number, max_dice = 5):
        """Perudo game class. The class allows to play a game between a 
        given number of opponents

        Args:
            player_number (int): number of total player at the start of the game.
            max_dice (int, optional): number of dice each player starts with. Defaults to 5.

        Raises:
            ValueError: _description_
        """
        if player_number < 2:
            raise ValueError("player_number must be at least 2")
        
        if max_dice < 1:
            raise ValueError("max_dice must be at least 1")

        # Variables.
        self._active_players = deque(random.sample(range(1, player_number+1), 
                                                   k=player_number))
        self._n_players = player_number
        self._game_over = False
        self._last_bet = [1, 0]
        self._max_dice = max_dice
        self._ranking = []

        # Functions.
        self._init_players(player_number, max_dice = max_dice)
        self._deal_player_hands()

    
    def _init_players(self, player_number, max_dice):
        """Initializes the player number of dice.

        Args:
            player_number (int): number of total player at the start of the game.
            max_dice (_type_): number of dice each player starts with. Defaults to 5.
        """
        self._players = {i: max_dice for i in range(1, player_number + 1)}


    def _deal_player_hands(self):
        """Distributes dice to every player.
        """
        self._player_hands = [
            np.random.randint(1, 7, size=self._players[i]) 
            for i in range(1, self._n_players + 1)
            ]
        self._n_dice = sum(self._players.values())


    def get_possible_actions(self):
        """Return every legal action from last bet.

        Returns:
            list(list): list of every legal bet as [quantity, value]
        """
        quantity, value = self._last_bet
        
        # If the number of total dice has been reached.
        if quantity == self._n_dice:
            if value == 6:
                pacos = [
                    [i, 1]
                    for i in range(int(np.ceil(quantity / 2)), self._n_dice + 1)
                    ]
                return [[-1, -1], [0, 0]] + pacos
            
            elif value == 1:
                return [[-1, -1], [0, 0]]
            
            else:
                pacos = [
                    [i, 1]
                    for i in range(int(np.ceil(quantity / 2)), self._n_dice + 1)
                    ]
                upper_values = [
                    [quantity, j]
                    for j in range(value + 1, 7)
                    ]
                return [[-1, -1], [0, 0]] + pacos + upper_values

        # Last bet was on pacos + the quantity isn't maximal.
        elif value == 1:
            pacos = [
                [i, 1]
                for i in range(quantity + 1, self._n_dice + 1)
                ]
            possible_actions = [[-1, -1], [0, 0]] + pacos

            if quantity <= self._n_dice // 2:
                possible_actions += [
                    [i, j]
                    for i in range(2 * quantity + 1, self._n_dice + 1)
                    for j in range(2, 7)
                    ]
            return possible_actions
        
        # If it is possible to escalate the value.
        else:
            upper_values = [
                [quantity, j] 
                for j in range(max(value + 1, 2), 7)
                ]
            upper_quantities_values = [
                [i, j]
                for i in range(quantity + 1, self._n_dice + 1)
                for j in range(2, 7)
                ]
            possible_actions = upper_values + upper_quantities_values
    
            # Actions involving pacoses.
            pacos = [
                [i, 1]
                for i in range(int(np.ceil(quantity / 2)), self._n_dice + 1)
                ]
            possible_actions = pacos + possible_actions
            # If it is not the first turn (default bet is [1, 0] at the beginning).
            if value > 0:
                possible_actions = [[-1, -1], [0, 0]] + possible_actions
            return possible_actions


    def _check_dice(self, bet, verbose = True):
        """Check if last bet was correct. 

        This function is only triggered when a player calls the last one a liar
        or called "exact" ([0, 0]). In these cases, we will check the last bet.

        Args:
            bet (list): Bet to verify. Format: [quantity, value].

        Returns:
            int: 0 if last player lied => he will lose a die.
                -1 if current player was wrong => current player loses a die.
                1 if current player called "exact" and was right => he earns a die back.
        """
        total_dice = np.concatenate(self._player_hands)
        quantity, value = self._last_bet
        current_player = self._active_players[0]
        last_player = self._active_players[-1]

        count = np.count_nonzero(total_dice == value)
        # Counting pacoses.
        if value > 1:
            count += np.count_nonzero(total_dice == 1)
        
        # Turn player called the previous one liar.
        if bet == [-1, -1]:
            if verbose:
                print(f"player {current_player} called player {last_player} a liar")
            
            # Last player indeed lied.
            if count < quantity:
                return 0
            # Last player was right.
            else:
                return -1

        # Turn player called exact.
        if bet == [0, 0]:
            if verbose:
                print(f"player {current_player} called exact")
            # Last player said the exact quantity.
            if count == quantity:
                return 1
            # Last player didn't say right quantity.
            else:
                return -1
            
        # If _check_dice was called while it shouldn't have had.
        raise ValueError(f"_check_dice called with invalid bet: {bet}")



    def make_a_bet(self, bet, verbose = True):
        """Current player makes a bet.

        The player can call a quantity of dice ([quantity, value]), call last player a "liar" ([-1, -1]) or call "exact" ([0, 0]).

        Args:
            bet (list): Bet done by the player. Format: [quantity, value].
        """
        current_player = self._active_players[0]
        last_player = self._active_players[-1]

        if not self._game_over:
            # Turn player calls "liar" or "exact".
            if bet in [[-1, -1], [0, 0]]:
                result = self._check_dice(bet, verbose = verbose)

                # Last player indeed lied.
                if result == 0:
                    self._players[last_player] += -1
                    self._active_players.rotate(1)
                    if verbose:
                        print(f"player {last_player} lost a dice")

                    if self._players[last_player] == 0:
                        if verbose:
                            print(f"player {last_player} is out")
                        self._remove_current_player(verbose = verbose)

                # Turn player was wrong.
                elif result == -1:
                    self._players[current_player] += -1
                    if verbose:
                        print(f"player {current_player} lost a dice")

                    if self._players[current_player] == 0:
                        if verbose:
                            print(f"player {current_player} is out")
                        self._remove_current_player(verbose = verbose)

                # Turn player called exact and won.
                else:
                    self._players[current_player] = min(
                        self._players[current_player] + 1, self._max_dice
                        )
                    if verbose:
                        print(f"player {current_player} gets a dice back")
                # Starting a new round.
                self._new_round()
                
            else:
                self._last_bet = bet
                self._new_turn()


    def get_state(self):
        """Return state of the game.

        Returns:
            tuple: (turn player, list of possible actions)
        """
        turn_player = self._active_players[0]
        if self._game_over == True:
            return (turn_player, [])

        return turn_player


    def _remove_current_player(self, verbose = True):
        """Remove current player.
        """
        player_index = self._active_players.popleft()
        self._ranking = [player_index] + self._ranking
        if len(self._active_players) <=1:
            if verbose:
                print("game ended")
            self._game_over = True
            self._ranking = [self._active_players[0]] + self._ranking


    def _new_turn(self):
        """Make the game update for next player to play
        """
        self._active_players.rotate(-1)


    def _new_round(self):
        """Update game for a new round
        """
        self._deal_player_hands()
        self._last_bet = [1, 0]
