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

def synchronize_field(iron_field: MagneticField, synchronous_field: np.ndarray) -> MagneticField:
    """
    Adjust a full iron magnetic field so that its **first moment** (radial
    synchronous field) matches a supplied target field.

    The *first moment* of a magnetic field :math:`B(\\theta, r)` is the radial
    average obtained via :meth:`MagneticField.first_moment`.  This routine adds a
    constant correction to every point of the iron field so that the corrected
    field has exactly the same first‑moment values as ``synchronous_field`` while
    preserving the original angular‑radial shape of ``iron_field``.

    The operation is pure – the input ``iron_field`` is **not** mutated; a new
    :class:`MagneticField` instance is returned.

    Parameters
    ----------
    iron_field : MagneticField
        The base (un‑corrected) magnetic field.  ``iron_field.values`` must be a
        two‑dimensional ``np.ndarray`` of shape ``(N_θ, N_r)`` where the **first
        index** corresponds to the azimuthal coordinate *θ* (in **radians**) and
        the **second index** corresponds to the radial coordinate *r* (in
        **inches**).  The object also provides the method
        ``first_moment`` which returns the radial synchronous component
        (a 1‑D array of length ``N_r``).

    synchronous_field : numpy.ndarray
        Desired first‑moment field.  This array should have the same shape as the
        output of ``iron_field.first_moment(recalculate=True)`` – i.e. a
        one‑dimensional array of length ``N_r`` containing the target radial
        values (in the same magnetic‑field units as ``iron_field`` – for
        example Tesla or Gauss).

    Returns
    -------
    MagneticField
        A new magnetic‑field object whose ``values`` are the original iron field
        plus a constant offset that forces the first moment to equal
        ``synchronous_field``.  The returned object re‑uses the metadata from
        ``iron_field`` (grid definition, units, etc.).

    Notes
    -----
    The correction is computed as

    .. math::

        \\Delta B = B_{\\text{sync}} - \\langle B_{\\text{iron}} \\rangle,

    where ``\\langle B_{\\text{iron}} \\rangle`` denotes the first moment of the
    iron field.  The same correction is added to **every** grid point:

    .. math::

        B_{\\text{new}}(\\theta, r) = B_{\\text{iron}}(\\theta, r) + \\Delta B(r).

    In NumPy the broadcasting is performed with

    .. code-block:: python

        np.einsum('ij,j->ij', np.ones_like(b_0), correction)

    because ``b_0`` has shape ``(N_θ, N_r)`` and ``correction`` is a
    1‑D array of length ``N_r``.  The ``einsum`` call expands the correction
    along the θ‑axis, which is equivalent to ``b_0 + correction[np.newaxis, :]``.

    Examples
    --------
    >>> from ops.cyclotron.analysis import MagneticField, synchronize_field
    >>> iron = MagneticField(metadata, values)          # shape (360, 100) → (θ, r)
    >>> target = np.linspace(1.0, 2.0, 100)             # desired first moment (len = N_r)
    >>> synced = synchronize_field(iron, target)
    >>> np.allclose(synced.first_moment(), target)     # True
    >>> # Original iron field is unchanged
    >>> np.array_equal(iron.values, values)            # True

    Raises
    ------
    TypeError
        If ``synchronous_field`` is not a ``numpy.ndarray`` or if its shape does
        not match the first‑moment shape of ``iron_field``.
    """
    b_0 = iron_field.values

    # Current first moment (radial synchronous field).  ``recalculate=True``
    # forces an up‑to‑date value in case the iron field has been mutated
    # elsewhere.
    current_moment = iron_field.first_moment(recalculate=True)

    # Defensive checks
    if not isinstance(synchronous_field, np.ndarray):
        raise TypeError("synchronous_field must be a numpy.ndarray")
    if synchronous_field.shape != current_moment.shape:
        raise TypeError(
            f"synchronous_field shape {synchronous_field.shape} does not match "
            f"the iron field's first moment shape {current_moment.shape}"
        )

    # Compute the constant correction that will be added to every θ slice.
    correction = synchronous_field - current_moment

    # Broadcast the 1‑D correction over the θ (first) axis.
    values = b_0 + np.einsum("ij,j->ij", np.ones_like(b_0), correction)

    # Return a fresh MagneticField that carries the same metadata as the input.
    return MagneticField(iron_field.metadata, values)

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
