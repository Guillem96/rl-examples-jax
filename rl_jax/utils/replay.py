import random
from typing import NamedTuple, Union, Sequence
from collections import deque

from rl_jax.typing import JaxTensor


class Transition(NamedTuple):
    state: JaxTensor
    next_state: JaxTensor
    action: Union[JaxTensor, int]
    reward: int
    is_terminal: bool


class ReplayMemory(object):

    def __init__(self, size: int = 1e5):
        self.memory = deque(maxlen=int(size))
    
    def experience(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, sample_size: int) -> Sequence[Transition]:
        return random.sample(self.memory, k=sample_size)
    
    def __len__(self):
        return len(self.memory)