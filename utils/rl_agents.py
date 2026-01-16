# ---- Regular Packages ----
import numpy as np
import random

# ---- Deep Learning Packages ----
import torch
import torch.nn as nn

# ---- Custom Packages ----
from .action_management import get_legal_actions_mask, get_possible_actions
from .reinforcement_learning import ReplayMemory

# --------------------------------------------------------
# Agent class
# --------------------------------------------------------

class RLAgent():

    def __init__(self, n_states, n_actions, action_dict, reverse_action_dict, DQN_class):

        self.policy_network = DQN_class(n_states, n_actions)
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_dict = action_dict
        self.reverse_action_dict = reverse_action_dict

    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return q_values

    def select_action(self, last_bet, total_dice, state):

        # Computing legal actions mask
        legal_actions_mask = get_legal_actions_mask(last_bet = last_bet, 
                                                    total_dice = total_dice,
                                                    action_dict = self.action_dict)
        tensor_mask = torch.from_numpy(legal_actions_mask)
        
        # Q values inference
        action_q_values = self.get_q_values(state)
        # Masking illegal actions
        action_q_values[~tensor_mask] = -1e9
        # Action selection
        action_index = action_q_values.argmax().item()
        action = self.reverse_action_dict[action_index]

        return list(action)
    

# --------------------------------------------------------
# Class for Online Training
# --------------------------------------------------------


class RLAgentOnline():

    def __init__(self, 
                 n_states, 
                 n_actions, 
                 action_dict, 
                 reverse_action_dict,
                 reward_dict,
                 transitions_tuple,
                 memory_capacity,
                 DQN_class):

        self.policy_network = DQN_class(n_states, n_actions)
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_dict = action_dict
        self.reward_dict = reward_dict
        self.reverse_action_dict = reverse_action_dict

        self._mask_function = get_legal_actions_mask


        self.memory = ReplayMemory(transitions_tuple = transitions_tuple, 
                                             capacity = memory_capacity)
        
        # Initialising histories
        self._rest_histories()

    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return q_values

    def select_action(self, last_bet, total_dice, state):

        # Computing legal actions mask
        legal_actions_mask = self._mask_function(last_bet = last_bet, 
                                                    total_dice = total_dice,
                                                    action_dict = self.action_dict)
        tensor_mask = torch.from_numpy(legal_actions_mask)
        
        # Q values inference
        action_q_values = self.get_q_values(state)
        # Masking illegal actions
        action_q_values[~tensor_mask] = -1e9
        # Action selection
        action_index = action_q_values.argmax().item()
        action = self.reverse_action_dict[action_index]

        return list(action)
    
    def epsilon_greedy_select_action(self, 
                                     last_bet, 
                                     total_dice, 
                                     state, 
                                     epsilon, 
                                     game_index, 
                                     verbose):
        
        sample = random.random()
        if sample <= epsilon:
            possible_actions = get_possible_actions(last_bet = last_bet, 
                                                    total_dice = total_dice)
            action_index = np.random.randint(len(possible_actions))
            action = possible_actions[action_index]
            
        else:
            action = self.select_action(last_bet = last_bet, 
                                        total_dice = total_dice, 
                                        state = state)
            
        self._witness_state(last_bet = last_bet,
                            total_dice = total_dice,
                            current_state = state,
                            action = action,
                            game_index = game_index)

        if verbose:
            if action == [-1, -1]:
                print("agent calls liar")
            elif action == [0, 0]:
                print("agent calls exact")
            else:
                print(f"agent outbids with: {action}")

        return action


    # ---- Memory management ----
    def _rest_histories(self):
        self._state_history = []
        self._action_history = []
        self._reward_history = []

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
        former_action_index = self.action_dict[tuple(former_action)]
        former_reward = self._reward_history[-1]
        legal_actions_mask = self._mask_function(last_bet = last_bet, 
                                                    total_dice = total_dice,
                                                    action_dict = self.action_dict)

        self.memory.push(former_state, 
                          former_action_index, 
                          second_state, 
                          former_reward, 
                          legal_actions_mask,
                          game_index)
        

    def _witness_state(self, 
                      last_bet,
                      total_dice,
                      current_state,
                      action,
                      game_index):

        if self._state_history:
            
            self._push_transition(last_bet = last_bet, 
                                  total_dice = total_dice, 
                                  second_state = current_state,
                                  game_index = game_index)

        # Saving histories
        self._state_history.append(current_state)
        self._action_history.append(action)


    def receive_outcome(self, 
                        bet_outcome_index, 
                        game_index):

        if self._state_history:
            reward = self.reward_dict[bet_outcome_index]
            self._reward_history.append(reward)
    
            # Last state was terminal.
            if bet_outcome_index != 2:
                former_state = self._state_history[-1]
                former_action = self._action_history[-1]
                former_action_index = self.action_dict[tuple(former_action)]
                current_state = None
                legal_actions_mask = np.zeros(len(self.action_dict), dtype=bool)
    
                self.memory.push(former_state, 
                              former_action_index, 
                              current_state, 
                              reward, 
                              legal_actions_mask, 
                              game_index)
    
                self._rest_histories()