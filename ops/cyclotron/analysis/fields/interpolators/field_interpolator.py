from abc import ABC, abstractmethod
from typing import Any

from ops.cyclotron.analysis.model import MagneticField

class FieldInterpolator(ABC):
    def __init__(self, 
                 magnetic_field: MagneticField,
                 l_0: float,
                 b_0: float) -> None:
        self.magnetic_field = magnetic_field
        self.l_0 = l_0
        self.b_0 = b_0

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
