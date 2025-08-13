import numpy as np
from scipy.interpolate import CubicSpline

from .constants import INCH_GAUSS_PER_AMU as _K
from ops.cyclotron.analysis.model import MagneticField, BeamParameters

def calculate_flutter_squared(magnetic_field: MagneticField, 
                              synchronous_field: np.ndarray) -> np.ndarray:
    """For a given iron field and a related synchronous field,
    calculate the flutter, related to the azimuthal variations in
    the iron field compared to the required synchronous field.

    :param magnetic_field: the iron magnetic field
    :type magnetic_field: MagneticField
    :param synchronous_field: the synchronous field
    :type synchronous_field: np.array
    :return: an array of the flutter squared
    :rtype: np.array
    """
    return np.divide(magnetic_field.second_moment() - magnetic_field.first_moment_squared(), 
                     np.power(synchronous_field, 2))

def calculate_cyclotron_length(beam_parameters: BeamParameters, 
                               magnetic_field_unit: float) -> float:
    return (_K * beam_parameters.mass/(magnetic_field_unit * beam_parameters.charge))

def calculate_sigma(flutter_squared: np.ndarray, 
                    R_vector: np.ndarray, 
                    cyclotron_length: float, 
                    N: int = 3) -> np.ndarray:
    dfdr = CubicSpline(R_vector, flutter_squared).derivative()
    return -1/(N**2 - 1)*(flutter_squared + R_vector/2 * dfdr(R_vector))

def calculate_frequency(beam: BeamParameters, magnetic_field_unit: float) -> float:
    return (beam.charge * magnetic_field_unit)/(6.5594262E2 * beam.mass)
