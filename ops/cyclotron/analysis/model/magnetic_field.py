from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class FieldMetadata:
    """Metadata for magnetic fields, generally including the physical parameters that correspond
    to the field matrix indices.

    """
    r_min: float                    # Value of r corresponding to [,,0] in inches
    delta_r: float                  # Delta-r between index points in inches
    theta_min: float                # Value of theta corresponding to [,0,] in degrees
    delta_theta: float              # Delta-theta between theta values in degrees

    @property 
    def theta_min_rad(self) -> float:
        return np.deg2rad(self.theta_min)

    @property 
    def delta_theta_rad(self) -> float:
        return np.deg2rad(self.delta_theta)

class MagneticField:
    def __init__(self, metadata: FieldMetadata, 
                 values: np.ndarray):
        self.metadata = metadata
        self.values = values
        self._first_moment = None
        self._first_moment_squared = None
        self._square = None
        self._second_moment = None


    @property
    def r_values(self) -> List[float]:
        return [self.metadata.r_min + i*self.metadata.delta_r 
                for i in range(int(np.shape(self.values)[1]))]

    @property
    def theta_values(self) -> List[float]:
        return [self.metadata.theta_min + i*self.metadata.delta_theta 
                for i in range(int(np.shape(self.values)[0]))]

    def first_moment(self, recalculate=False) -> np.ndarray:
        if self._first_moment is None or recalculate:
            self._first_moment = np.mean(self.values, axis=0)
        return self._first_moment


    def first_moment_squared(self, recalculate=False) -> np.ndarray:
        if self._first_moment_squared is None or recalculate:
            self._first_moment_squared = np.power(self.first_moment(recalculate=True), 2)
        return self._first_moment_squared

    def square(self, recalculate=False) -> np.ndarray:
        if self._square is None or recalculate:
            self._square = np.power(self.values, 2)
        return self._square

    def second_moment(self, recalculate=False) -> np.ndarray:
        if self._second_moment is None or recalculate:
            self._second_moment = np.mean(self.square(recalculate=True), axis=0)
        return self._second_moment
