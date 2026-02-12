# ---- Custom Packages ----
from ..game import GameRL


# ---- Class made streamlit application ----
class GameStreamlit(GameRL):

    def __init__(self, player_number, max_dice = 5):

        super().__init__(player_number, max_dice)

    def make_a_bet(self, bet):
        """Current player makes a bet. 
           Returns the outcome of the bet (used for reinforcement learning).

        The player can call a quantity of dice ([quantity, value]), call last player a "liar" ([-1, -1]) or call "exact" ([0, 0]).

        Args:
            bet (list): Bet done by the player. Format: [quantity, value].

        Returns:
            result (int): 0 if last player lied => he will lose a die.
                         -1 if current player was wrong => current player loses a die.
                          1 if current player called "exact" and was right => he earns a die back.
                          2 if the player outbid.
        """
        current_player = self._active_players[0]
        last_player = self._active_players[-1]

        if not self.game_over:
            # Turn player calls "liar" or "exact".
            if bet in [[-1, -1], [0, 0]]:
                result = self._check_dice(bet)

                # Last player indeed lied.
                if result == 0:
                    self._player_dice_number[last_player] += -1
                    self._active_players.rotate(1)
                    message = f"player {last_player} lost a dice"

                    if self._player_dice_number[last_player] == 0:
                        message = f"player {last_player} is out"
                        self._remove_current_player(verbose = False)

                # Turn player was wrong.
                elif result == -1:
                    self._player_dice_number[current_player] += -1
                    message = f"player {current_player} lost a dice"

                    if self._player_dice_number[current_player] == 0:
                        message = f"player {current_player} is out"
                        self._remove_current_player(verbose = False)

                # Turn player called exact and was right.
                elif result == 1:
                    self._player_dice_number[current_player] = min(
                        self._player_dice_number[current_player] + 1, 
                        self._max_dice
                        )
                    
                    message = f"player {current_player} gets a dice back"
                # Starting a new round.
                self._new_round()
                
            else:
                self._last_bet = bet
                self._new_turn()
                result = 2
                message = None

        return result, message