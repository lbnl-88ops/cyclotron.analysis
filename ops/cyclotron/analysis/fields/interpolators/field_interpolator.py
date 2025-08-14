from abc import ABC, abstractmethod
from typing import Any

class FieldInterpolator(ABC):
    def __init__(self, magnetic_field) -> None:
        self.magnetic_field = magnetic_field

    def __call__(self, r: float, theta: float, *args: Any, **kwds: Any) -> Any:
        return self.b(r, theta)

    @abstractmethod
    def b(self, r: float, theta: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def dbdt(self, r: float, theta: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def dbdr(self, r: float, theta: float) -> float:
        raise NotImplementedError
