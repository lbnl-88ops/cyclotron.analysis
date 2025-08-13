from math import sqrt

from .constants import (C_SQUARED_IN_MEV_PER_AMU as _C_SQUARED, 
                                           EXTRACTION_RADIUS_IN_INCHES as _REFERENCE_RADIUS, 
                                           INCH_GAUSS_PER_AMU as _K, 
                                           ELECTRON_MASS_IN_MEV as _ME, 
                                           MAIN_CURRENT_EXPANSION_CONSTANTS as _EXP_COEF)

from ops.cyclotron.analysis.model import BeamParameters

def calculate_ion_mass_in_mev(beam_parameters: BeamParameters) -> float:
    """Calculate the mass of beam ions in MeV.

    :param beam_parameters: beam of interest
    :type beam_parameters: BeamParameters
    :return: mass in MeV of the beam ions
    :rtype: float
    """
    return _C_SQUARED * beam_parameters.mass - _ME * beam_parameters.charge

def calculate_ion_mass_in_amu(beam_parameters: BeamParameters) -> float:
    """Calculate the mass of beam ions in amu.

    :param beam_parameters: beam of interest
    :type beam_parameters: BeamParameters
    :return: mass in amu of the beam ions
    :rtype: float
    """
    return calculate_ion_mass_in_mev(beam_parameters)/_C_SQUARED

def calculate_momentum_squared(beam_parameters: BeamParameters) -> float: 
    """Calculate the squared momentum of beam ions in units of mass times the speed of light.

    :param beam_parameters: beam of interest
    :type beam_parameters: BeamParameters
    :return: squared momentum in units of mass times the speed of light
    :rtype: float
    """
    rest_mass = calculate_ion_mass_in_mev(beam_parameters)
    return beam_parameters.energy/rest_mass * (2 + beam_parameters.energy/rest_mass)

def calculate_required_b_field(beam_parameters: BeamParameters, 
                               radius: float = _REFERENCE_RADIUS) -> float:
    """Calculate the required magnetic field in Gauss for a given beam and extraction radius.

    :param beam_parameters: beam of interest
    :type beam_parameters: BeamParameters
    :param extraction_radius: extraction radius in inches, defaults to _REFERENCE_RADIUS
    :type extraction_radius: float, optional
    :return: required magnetic field in Gauss
    :rtype: float
    """
    momentum = sqrt(calculate_momentum_squared(beam_parameters))
    mass = calculate_ion_mass_in_amu(beam_parameters)
    return _K * momentum * mass/(beam_parameters.charge * radius)

def calculate_b_rho(beam_parameters: BeamParameters) -> float:
    """Calculate the magnetic rigidity (Bρ) for a given beam.

    :param beam_parameters: beam of interest
    :type beam_parameters: BeamParameters
    :return: magnetic rigidity (Bρ) in kGauss-inches
    :rtype: float
    """
    momentum = sqrt(calculate_momentum_squared(beam_parameters))
    mass = calculate_ion_mass_in_amu(beam_parameters)
    return _K * momentum * mass/(beam_parameters.charge)/1000

def calculate_main_current(b_rho: float) -> float:
    """Calculate the main current for a given magnetic rigidity (Bρ).

    :param b_rho: magnetic rigidity in kGauss inches
    :type b_rho: float
    :return: main current in amperes (A)
    :rtype: float
    """
    return sum(c*b_rho**i for i, c in enumerate(_EXP_COEF))
