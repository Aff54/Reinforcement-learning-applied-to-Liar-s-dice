# ---- Regular Packages ----
from collections import namedtuple, deque    # for memory buffer
import os
import pickle
import random


# ---- Replay buffer ----
class ReplayMemory(object):
    """Memory buffer for storing transitions for reinforcement learning.
    """

    def __init__(self, transitions_tuple = None, capacity = 10000):
        """Instantiates class.

        Args:
            transitions_tuple (namedtuple): named tuple containing transitions.
            Usual format: namedtuple('Transition',
                        ('state', 
                        'action', 
                        'next_state', 
                        'reward', 
                        'legal_actions_mask', 
                        'new_legal_actions_mask')
                        )
            capacity (int, optional): Amount of transitions kept in memory. 
            Pushing a transition when memory is full will remove the oldest 
            transition.
            Defaults to 10000.
        """
        self.capacity = capacity
        self.transitions_tuple = transitions_tuple
        self.memory = deque([], maxlen = capacity)

    def __iter__(self):
        """Makes buffer instances iterable.
        """
        return iter(self.memory)



    def push(self, *args):
        """Stores a transition.
        
        Args:
            *args: transition elements
        """
        self.memory.append(self.transitions_tuple(*args))


    def sample(self, batch_size = 256, source = None):
        """Returns random transitions from memory without replacement.

        Args:
            batch_size (int, optional): Number of drawn element. Defaults to 256.

        Returns:
            _type_: transitions tuples.
        """
        if source is None:
            return random.sample(self.memory, batch_size)
        elif source == "train":
            return random.sample(self.train_memory, batch_size)
        elif source == "test":
            return random.sample(self.test_memory, batch_size)
        else:
            raise ValueError(f"Unknown source: {source}")
    

    def split(self, rate):
        """Splits memory into a training and testing memory.

        Args:
            rate (float): rate by which we keep transitions 
            in memory and put the rest in test_memory_length.
        """

        if not 0 < rate < 1:
            raise ValueError("rate must be in (0, 1)")
        
        transition_list = list(self.memory)
        random.shuffle(transition_list)

    
        memory_length = len(self.memory)
        train_memory_length = int(memory_length * rate)
        self.train_memory = deque(transition_list[:train_memory_length],
                                   maxlen = self.capacity)
        self.test_memory = deque(transition_list[train_memory_length:], 
                                 maxlen = self.capacity)

    def pop(self):
        return self.memory.pop()
    

    def __len__(self):
        return len(self.memory)
    

    def get_full_memory(self, source = None):
        """Returns full memory.
        """
        if source is None:
            return self.memory
        elif source == "train":
            return self.train_memory 
        elif source == "test":
            return self.test_memory
        else:
            raise ValueError(f"Unknown source: {source}")


    def load(self, path):
        """_summary_

        Args:
            path (str): Path leading to memory deque saved with .save method.

        Raises:
            FileNotFoundError: deque not found in given path.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Replay buffer file not found: {path}")
    
        with open(path, "rb") as f:
            data = pickle.load(f)
    
        self.capacity = data["capacity"]
        self.memory = deque(data["memory"], maxlen = self.capacity)


    def save(self, path):
        """Save memory deque with metadata.

        Args:
            path (str): location to store memory deque into.
        """
        with open(path, "wb") as f:
            pickle.dump({
                "capacity": self.capacity,
                "memory": list(self.memory),
                },f)