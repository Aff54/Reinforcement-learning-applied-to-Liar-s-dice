# ---- Regular Packages ----

# ---- Deep Learning Packages ----
import torch
import torch.nn as nn

# ---- Custom Packages ----
from .action_management import get_legal_actions_mask

# --------------------------------------------------------
# Agent class
# --------------------------------------------------------

class RLAgent():

    def __init__(self, n_states, n_actions, action_dict, reverse_action_dict, DQN_class):

        self.policy_network = DQN_class(n_states, n_actions)

    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return q_values

    def select_action(self, last_bet, total_dice, state):

        # Computing legal actions mask
        legal_actions_mask = get_legal_actions_mask(last_bet = last_bet, 
                                                    total_dice = total_dice,
                                                    action_dict = action_dict)
        tensor_mask = torch.from_numpy(legal_actions_mask)
        
        # Q values inference
        action_q_values = self.get_q_values(state)
        # Masking illegal actions
        action_q_values[~tensor_mask] = -1e9
        # Action selection
        action_index = action_q_values.argmax().item()
        action = reverse_action_dict[action_index]

        return list(action)