# ---- Regular Packag ----
import random
import numpy as np
from functools import lru_cache 

# ---- Custom packages ----
from .probability_analysis import bet_conditional_probability, bet_exact_probability
from .action_management import get_possible_actions
from .reinforcement_learning import ReplayMemory


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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


# --------------------------------------------------------
# Agent class
# --------------------------------------------------------

class Agent():

    def __init__(self,
                 agent_function,
                 action_dict,
                 reward_dict,
                 mask_function,
                 transitions_tuple,
                 memory_capacity):
        
        self._decision_function = agent_function
        self._action_dict = action_dict
        self._reward_dict = reward_dict
        self._memory = ReplayMemory(transitions_tuple = transitions_tuple, 
                                             capacity = memory_capacity)
        self._mask_function = mask_function

        self.__name__ = "Agent_" + agent_function.__name__

        # Initialising histories
        self._rest_histories()

    
    def make_a_bet(self,
                   last_bet,
                   total_dice,
                   player_hand,
                   current_state,
                   game_index,
                   verbose = False):

        action = self._decision_function(last_bet = last_bet,
                                         total_dice = total_dice,
                                         player_hand = player_hand,
                                         verbose = verbose)

        if self._state_history:
            
            self._push_transition(last_bet = last_bet, 
                                  total_dice = total_dice, 
                                  second_state = current_state,
                                  game_index = game_index)

        # Saving histories
        self._state_history.append(current_state)
        self._action_history.append(action)

        return action

    
    def receive_outcome(self, 
                        bet_outcome_index, game_index):

        if self._state_history:
            reward = self._reward_dict[bet_outcome_index]
            self._reward_history.append(reward)
    
            # Last state was terminal.
            if bet_outcome_index != 2:
                former_state = self._state_history[-1]
                former_action = self._action_history[-1]
                former_action_index = self._action_dict[tuple(former_action)]
                current_state = None
                legal_actions_mask = np.zeros(len(self._action_dict), dtype=bool)
    
                self._memory.push(former_state, 
                              former_action_index, 
                              current_state, 
                              reward, 
                              legal_actions_mask, 
                              game_index)
    
                self._rest_histories()
            
    
    def _push_transition(self, 
                         last_bet, 
                         total_dice, 
                         second_state,
                         game_index):
        
        """Pushes a transition into the agent's memory.
        
        :param last_bet: last bet current player plays after. 
        format: [quantity, value]
        :param total_dice: Total number of dice in game.
        :param second_state: Second state of the transition. 
        First state was the starting state and this state results from the
        action that was taken by the agent.
        :param game_index: Index of current game. For data analysis.

        /!\ legal_actions_mask 
        """

        if len(self._reward_history) != len(self._state_history):
            raise ValueError(f"Trying to push a transition with self._reward_history and self._state_history not having same length ({len(self._reward_history)} VS {len(self._state_history)})")

        former_state = self._state_history[-1]
        former_action = self._action_history[-1]
        former_action_index = self._action_dict[tuple(former_action)]
        former_reward = self._reward_history[-1]
        legal_actions_mask = self._mask_function(last_bet = last_bet, 
                                                 total_dice = total_dice,
                                                 action_dict = self._action_dict)

        self._memory.push(former_state, 
                          former_action_index, 
                          second_state, 
                          former_reward, 
                          legal_actions_mask,
                          game_index)

    def _rest_histories(self):
        self._state_history = []
        self._action_history = []
        self._reward_history = []