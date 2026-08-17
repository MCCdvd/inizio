from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, state_size, action_size, config=None):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.config = config or {}

    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError
