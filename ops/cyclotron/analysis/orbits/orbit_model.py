from dataclasses import dataclass
from logging import getLogger

import numpy as np
from scipy.integrate import solve_ivp, OdeSolution

from ops.cyclotron.analysis.fields.interpolators import FieldInterpolator
from ops.cyclotron.analysis.model import MagneticField

_log = getLogger(__name__)

class OrbitError(BaseException):
    pass

@dataclass(init=False)
class Orbit:
    beta: float
    gamma: float
    r_init: float
    pr_init: float
    p_total: float
    time: float
    r_average: float 
    transfer_matrices: np.ndarray
    full_solution: OdeSolution

class OrbitModel:
    def __init__(self, magnetic_field: MagneticField,
                 magnetic_field_interpolator: FieldInterpolator) -> None:
        self.magnetic_field = magnetic_field
        self.rs = np.array(magnetic_field.r_values)/magnetic_field_interpolator.l_0

        self.th = np.array(magnetic_field.theta_values 
                + [magnetic_field.theta_values[-1] 
                    + magnetic_field.metadata.delta_theta])*np.pi/180
        self.max_steps = 100
        self.include_z_solve = False
        self.n = 2
        self.tolerance = 1E-10
        self.interpolator = magnetic_field_interpolator

    def b_int(self, theta, r):
        return self.interpolator(r, theta)
    
    def dbt_int(self, theta, r):
        return self.interpolator.dbdt(r, theta)

    def dbr_int(self, theta, r):
        return self.interpolator.dbdr(r, theta)
    
    def solve_equilibrium_orbit_at_beta(self, beta: float, r_init: float, pr_init: float,
                                        do_initial_solve = True) -> Orbit:
        solved_orbit = Orbit()
        solve_steps = np.linspace(self.th[0], self.th[-1], self.n*len(self.th))
        gamma = 1/np.sqrt(1 - beta**2)
        p = beta * gamma
        p2 = p**2
        r: float = 0
        pr: float = 0
        if r_init > self.rs[-1] or r_init < self.rs[0]:
            raise OrbitError(f'r_init ({r_init}) not in range of magnetic field')
        if do_initial_solve:
            for i in range(self.max_steps):
                def orbit(t, z):
                    pr, r, px1, x1, px2, x2 = z
                    if pr**2 > p2:
                        raise OrbitError(f'Bad momentum {pr**2} > {p2}')
                    p_th = np.sqrt(p2 - pr**2) 
                    b_value = self.b_int(t, r)
                    dbdr_value = self.dbr_int(t, r)
                    dbdr_term = b_value + r * dbdr_value
                    return [
                        p_th - r*b_value,
                        r/p_th*pr,
                        -pr/p_th*px1 - dbdr_term*x1,
                        (pr/p_th)*x1 + (p2*r/(p_th**3))*px1,
                        -(pr/p_th)*px2 - dbdr_term*x2,
                        (pr/p_th)*x2 + (p2*r/(p_th**3))*px2
                    ]

                z0 = [pr_init, r_init, 0, 1, 1, 0]
                try:
                    sol = solve_ivp(orbit, [self.th[0], self.th[-1]], z0, t_eval=solve_steps,
                                    dense_output=True)
                except RuntimeError as e:
                    raise OrbitError(f'Orbit failed: {str(e)}')
                pr, r, px1, x1, px2, x2 = (v[-1] for v in sol.y)
                epsilon_1 = (r - r_init)
                epsilon_2 = pr - pr_init

                denominator = x1 + px2 - 2
                delta_r = ((px2 - 1)*epsilon_1 - x2*epsilon_2)/denominator
                delta_pr = ((x1 - 1)*epsilon_2 - px1*epsilon_1)/denominator
                convergence = np.sqrt((delta_r**2 + delta_pr**2))
                if abs(convergence) < self.tolerance:
                    _log.debug(f'Initial orbit converged in {i + 1} iterations')
                    break
                elif i + 1 == self.max_steps:
                    raise OrbitError('Orbit failed to converge')
                r_init = r_init + delta_r
                pr_init = pr_init + delta_pr
        solved_orbit.r_init = r
        solved_orbit.pr_init = pr
        solved_orbit.beta = beta
        solved_orbit.gamma = gamma
        solved_orbit.p_total = p

        def full_orbit(t, z):
            pr, r, px1, x1, px2, x2, pz1, z1, pz2, z2, _, _ = z
            if pr**2 > p2:
                raise RuntimeError(f'Bad momentum {pr**2} > {p2}')
            p_th = np.sqrt(p2 - pr**2) 
            b_value = self.b_int(t, r)
            dbdr_value = self.dbr_int(t, r)
            dbdth_value = self.dbt_int(t, r)
            dbdr_term = b_value + r * dbdr_value
            return [
                p_th - r*b_value,
                r/p_th*pr,
                -pr/p_th*px1 - dbdr_term*x1,
                (pr/p_th)*x1 + (p2*r/(p_th**3))*px1,
                -(pr/p_th)*px2 - dbdr_term*x2,
                (pr/p_th)*x2 + (p2*r/(p_th**3))*px2,
                (r*dbdr_value - pr/p_th*dbdth_value)*z1,
                (r/p_th)*pz1,
                (r*dbdr_value - pr/p_th*dbdth_value)*z2,
                (r/p_th)*pz2,
                3/(2*np.pi)* gamma*r/p_th,
                3/(2*np.pi) * r,
            ]
        z0 = [pr, r, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0] 
        try:
            sol = solve_ivp(full_orbit, [self.th[0], self.th[-1]], z0, t_eval=solve_steps,
                                    dense_output=True) 
        except RuntimeError as e:
            raise OrbitError(f'Orbit failed to converge full orbit: {str(e)}')
        solved_orbit.time = sol.y[-2][-1] 
        solved_orbit.r_average = sol.y[-1][-1] 
        px1, x1, px2, x2, pz1, z1, pz2, z2 = (v[-1] for v in sol.y[2:10]) 
        solved_orbit.transfer_matrices = np.array([ 
            [[x1, x2], [px1, px2]], 
            [[z1, z2], [pz1, pz2]]
            ])
        solved_orbit.full_solution = sol
        return solved_orbit