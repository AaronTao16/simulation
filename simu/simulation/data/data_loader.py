from abc import ABC, abstractmethod
from typing import List

# load data for simulation
class DataLoader(ABC):

    @abstractmethod
    def classifications(self) -> List[str]:
        pass

    @abstractmethod
    def load_data(self, train_size: int = None, test_size: int = None) -> (tuple, tuple):
        pass
