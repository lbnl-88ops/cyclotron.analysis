from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

from ops.cyclotron.analysis.model import MagneticField
from ops.cyclotron.analysis.fields.interpolators import FieldInterpolator
from .orbit_model import OrbitError, OrbitModel

_log = getLogger(__name__)

@dataclass(init=False)
class Orbit:
    beta: float
    gamma: float
    r_init: float
    pr_init: float
    pth_init: float
    time: float
    r_average: float 
    transfer_matrix: np.ndarray
    full_solution: scipy.integrate.OdeSolution

class Rho:
    def __init__(self) -> None:
        self._rhos = [0, 0, 0]
        self.iterations = 0
    def add_rho(self, value) -> None:
        # Add rho value and shift all values one back, drop last one
        self._rhos[2] = self._rhos[1]
        self._rhos[1] = self._rhos[0]
        self._rhos[0] = value
        self.iterations += 1
    def value(self) -> float:
        if self.iterations == 0:
            return 0
        elif self.iterations == 1:
            return self._rhos[0]
        elif self.iterations == 2:
            return 2*self._rhos[0] - self._rhos[1]
        return 3*self._rhos[0] - 3*self._rhos[1] + self._rhos[2]

def calculate_equilibrium_orbits(magnetic_field: MagneticField, 
                                 magnetic_field_interpolator: FieldInterpolator,
                                 *,
                                 plot: bool = False,
                                 maximum_radius: float = 68) -> List[Orbit]:
    orbit_model = OrbitModel(magnetic_field, magnetic_field_interpolator)
    cyclotron_length = magnetic_field_interpolator.l_0

    rho_r = Rho()
    rho_pr = Rho()
    orbit_list = []
    orbits = int(maximum_radius)

    if plot:
        plt.subplot(projection="polar")
    _log.info('Calculating orbits...')
    for j in tqdm(range(orbits)):
        _log.debug(f'Calculating orbit {j} of {orbits}')
        beta = magnetic_field.metadata.delta_r/cyclotron_length*(j + 1)
        gamma = 1/np.sqrt(1 - beta**2)
        p = beta*gamma
        r_init = beta*(rho_r.value() + 1)
        pr_init = p*rho_pr.value()
        if r_init > magnetic_field.r_values[-1]/cyclotron_length:
            _log.info('R-value outside of magnetic field, stopping.')
            break
        try:
            orbit = orbit_model.solve_equilibrium_orbit_at_beta(beta, r_init, pr_init)
        except OrbitError as e:
            _log.debug(f'Orbit failed: {e}')
            continue
        
        rho_r.add_rho(orbit.r_init/beta - 1)
        rho_pr.add_rho(orbit.pr_init/orbit.p_total)
        orbit_list.append(orbit)
        sol = orbit.full_solution
        if plot and j % 2 == 0:
            plt.plot(sol.t, sol.y[1]*cyclotron_length, '--')
    if plot: 
        plt.grid()
        plt.show()

    return orbit_list
    

