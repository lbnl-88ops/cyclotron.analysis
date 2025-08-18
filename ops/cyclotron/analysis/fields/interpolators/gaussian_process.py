from logging import getLogger
import numpy as np
import numdifftools as nd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .field_interpolator import FieldInterpolator
from ops.cyclotron.analysis.model import MagneticField

_log = getLogger(__name__)

class GaussianProcessFieldInterpolator(FieldInterpolator):
    def __init__(self, magnetic_field: MagneticField, l_0: float = 1.0, b_0: float = 1.0) -> None:
        super().__init__(magnetic_field, l_0, b_0)
        dth = magnetic_field.metadata.delta_theta

        th_pad_i = int(magnetic_field.theta_values[0]/dth)
        th_pad_f = int((360 - magnetic_field.theta_values[-1])/dth)
        ths_f = np.deg2rad(range(0, 360, int(dth)))
        r = [v/l_0 for v in magnetic_field.r_values[1:]]

        values = np.pad(magnetic_field.values[:,1:], [(th_pad_i, th_pad_f - 1), (0, 0)], 'wrap')/b_0
        tt, rr = np.meshgrid(ths_f, r, indexing='ij')

        X = np.vstack([tt.flatten(), rr.flatten()]).T
        Y = values.flatten()
        kernel = Matern(length_scale=(3, 20/l_0), 
                        length_scale_bounds=[(1E-8, 1E8), (1E-8, 1E8)], nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1E-5)
        _log.info('Fitting magnetic field Gaussian Process and computing derivatives.')
        self.gp.fit(X, Y)
        self.derivative = nd.Gradient(self._gp_mean)
        _log.debug(f'Fit: {self.gp.kernel_}')
        _log.info('Fitting complete.')

    def _gp_mean(self, x):
        return self.gp.predict([x])[0]

    def b(self, r: float, theta: float) -> float:
        value = self._gp_mean([theta, r])
        return float(value)

    def dbdr(self, r: float, theta: float) -> float:
        value = self.derivative([theta, r])[1]
        return float(value)

    def dbdt(self, r: float, theta: float) -> float:
        return float(self.derivative([theta, r])[0])
    