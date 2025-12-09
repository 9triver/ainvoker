from abc import ABC, abstractmethod
from ..world_state import WorldState


class Action(ABC):
    @abstractmethod
    def execute(self, state: WorldState, **kwargs) -> WorldState:
        pass
