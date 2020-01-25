import random
from typing import NamedTuple
from collections import deque

from rl_jax.typing import JaxTensor


class Transition(NamedTuple):
    state: JaxTensor
    next_state: JaxTensor
    action: Union[JaxTensor, int]
    reward: int


class ReplayMemory(object):

    def __init__(self, size: int = 1e5):
        self.memory = deque(maxlen=size)
    
    def experience(transition: Transition):
        self.memory.append(transition)

    def sample(sample_size: int) -> Sequence[Transition]:
        return random.sample(self.memory, k=sample_size)