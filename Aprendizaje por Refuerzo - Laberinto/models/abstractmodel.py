
from abc import ABC, abstractmethod  

class AbstractModel(ABC):  
    def __init__(self, maze, **kwargs):
        self.environment = maze  
        self.name = kwargs.get("name", "model")  

    def load(self, filename): 
        pass

    def save(self, filename):
        pass

    def train(self, stop_at_convergence=False, **kwargs):
        pass

    @abstractmethod
    def q(self, state):
        pass

    @abstractmethod
    def predict(self, state):
        pass
