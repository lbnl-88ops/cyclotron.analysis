from logging import getLogger
from math import sqrt

import numpy as np

from ops.cyclotron.analysis.physics import (calculate_required_b_field, calculate_momentum_squared, 
                                            calculate_flutter_squared, calculate_cyclotron_length, 
                                            calculate_sigma,
                                            EXTRACTION_RADIUS_IN_INCHES as _REFERENCE_RADIUS)
from ops.cyclotron.analysis.model import MagneticField, BeamParameters
from ops.cyclotron.analysis.exceptions import ConvergenceError

_log = getLogger(__name__)

# Various terms used to calculate and converge the synchronous field
def _estimate_magnetic_field_unit(beam_parameters: BeamParameters, 
                                  magnetic_field: float) -> float:
    """Estimate the magnetic field unit using a gamma calculated from the beam and a
    provided magnetic field.
    """
    gamma = sqrt(1.0 + calculate_momentum_squared(beam_parameters))
    return magnetic_field/gamma

def converge_synchronous_field(beam_parameters: BeamParameters, 
                               iron_field: MagneticField,
                               extraction_radius: float = _REFERENCE_RADIUS,
                               *,
                               max_iter: int = 100,
                               tol: float = 1E-16) -> np.ndarray | None:
    """
    Compute the synchronous magnetic field for a given set of beam parameters.

    This routine iterates until the synchronous field that satisfies extraction of the beam at the 
    is found. This method iteratively uses the flutter of the field to correct the synchronous 
    field until convergence.

    :param beam_parameters: Parameters describing the beam.
    :type beam_parameters: BeamParameters
    :param iron_field: The baseline (iron) magnetic field.
    :type iron_field: MagneticField
    :param extraction_radius: Position of the deflector used for extraction.
    :type extraction_radius: float
    :return: The synchronous magnetic field as a NumPy array.
    :rtype: numpy.ndarray

    :raises ConvergenceError: If the algorithm does not converge within the allowed number of 
        iterations.
    """
    # reference values
    reference_index = int((extraction_radius - iron_field.metadata.r_min)
                          /iron_field.metadata.delta_r)
    reference_magnetic_field = calculate_required_b_field(beam_parameters)
    reference_radius = extraction_radius

    # Initial values based on reference magnetic field
    magnetic_field_unit = _estimate_magnetic_field_unit(beam_parameters, reference_magnetic_field)
    cyclotron_length = calculate_cyclotron_length(beam_parameters, magnetic_field_unit)
    R_vector = np.array(iron_field.r_values)/cyclotron_length

    # Initial synchronous field and flutter calculation
    synchronous_field = np.divide(magnetic_field_unit, np.sqrt(1 - np.power(R_vector, 2)))
    flutter_squared = calculate_flutter_squared(iron_field, synchronous_field)
    sigma = calculate_sigma(flutter_squared, R_vector, cyclotron_length)

    # Lists for output
    magnetic_field_units = [magnetic_field_unit]
    synchronous_fields = [synchronous_field]

    for _ in range(max_iter):
        reference_sigma_term = 1 + sigma[reference_index]

        # Calcualte new magnetic field unit using sigma
        # new_magnetic_field_unit = (reference_magnetic_field
                                #    / (reference_sigma_term 
                                    #   * np.sqrt(1 - (reference_radius/cyclotron_length)**2 
                                                # * reference_sigma_term**2)
                                                # )
                                                # )
        new_magnetic_field_unit = ((reference_magnetic_field/reference_sigma_term)
                                   /np.sqrt(1 + (reference_radius/cyclotron_length)**2))

        magnetic_field_units.append(new_magnetic_field_unit)

        # Re-calculate lengths and sigma
        cyclotron_length = calculate_cyclotron_length(beam_parameters, new_magnetic_field_unit)
        R_vector = np.array(iron_field.r_values)/cyclotron_length

        # Calculate synchronous field
        synchronous_fields.append(new_magnetic_field_unit * (1 + sigma)
                                  / np.sqrt(1 - np.multiply(np.power(R_vector, 2), np.power(1 + sigma, 2))))
        # Compare magnetic field units
        difference = np.abs(magnetic_field_units[-2] - new_magnetic_field_unit)
        # _log.debug(report_variable(f'Iteration {len(magnetic_field_units) - 1}', difference, fmt='.4E'))

        if difference < tol:
            _log.info(f'Magnetic field term converged in {len(magnetic_field_units) - 1} iterations')
            return synchronous_fields[-1]
        else:
        # Correct flutter and sigma
            flutter_squared = np.multiply(flutter_squared, np.power(np.divide(synchronous_fields[-2], synchronous_fields[-1]), 2))
            sigma = calculate_sigma(flutter_squared, R_vector, cyclotron_length)
    raise ConvergenceError(max_iter=max_iter, tol=tol)
