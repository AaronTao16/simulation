from abc import ABC, abstractmethod


class TokenHash(ABC):
    @abstractmethod
    def hash(self, text: str) -> int:
        raise NotImplementedError
