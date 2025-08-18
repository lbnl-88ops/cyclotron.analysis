from scipy.interpolate import RectBivariateSpline
import numpy as np

from ops.cyclotron.analysis.model import MagneticField
from .field_interpolator import FieldInterpolator

class SplineFieldInterpolator(FieldInterpolator):
    """
    A class for interpolating magnetic field data using spline interpolation.

    This class implements the :class:`FieldInterpolator` abstract base class and provides
    methods for computing the magnetic field and its spatial derivatives with respect to
    radial and angular coordinates.

    Attributes:
        magnetic_field (:class:`MagneticField`): The magnetic field data used for interpolation.
        cyclotron_length (float, optional): The scaling factor for the radial values. Defaults to 1.0.
        magnetic_field_unit (float, optional): The scaling factor for the magnetic field values.
                Defaults to 1.0.
        full_rs (:class:`numpy.ndarray`): The complete set of scaled radial points used for interpolation.
        _b_int (:class:`RectBivariateSpline`): The 2D interpolator for magnetic field values.
        _dbt_int (:class:`RectBivariateSpline`): The 2D interpolator for the derivative with respect to theta.
        _dbr_int (:class:`RectBivariateSpline`): The 2D interpolator for the derivative with respect to r.

    Methods:
        b(r, theta):
            Interpolate the magnetic field value at given radial and theta coordinates.

        dbdr(r, theta):
            Interpolate the derivative of the magnetic field with respect to radial coordinate.

        dbdt(r, theta):
            Interpolate the derivative of the magnetic field with respect to theta coordinate.
    """
    def __init__(self, 
                 magnetic_field: MagneticField, 
                 cyclotron_length: float = 1.0,
                 magnetic_field_unit: float = 1.0) -> None:
        super().__init__(magnetic_field, cyclotron_length, magnetic_field_unit)
        rs = np.array(magnetic_field.r_values)/cyclotron_length
        self.full_rs = np.concatenate([np.flip(-rs[1:]), rs])
        d_th = int(self.magnetic_field.metadata.delta_theta)
        th = np.array(self.magnetic_field.theta_values)
        full_th = np.concat([[v for v in range(0, th[0], d_th)], 
                             th[:-1], 
                             [v for v in range(th[-1], 360 + d_th, d_th)]])*np.pi/180
        th_pad_i = int((th[0])/d_th)
        th_pad_f = int((360 - th[-1])/d_th)
        
        self._b_int = RectBivariateSpline(
            full_th, 
            self.full_rs, 
            np.pad(
                np.pad(self.magnetic_field.values, 
                       [(0, 0), (len(rs) - 1, 0)], 
                       mode='reflect'), 
                [(th_pad_i, th_pad_f), (0, 0)], 
                mode='wrap')/magnetic_field_unit, 
            kx=2, ky=2)
                                        
        self._dbt_int = self._b_int.partial_derivative(1, 0)
        self._dbr_int = self._b_int.partial_derivative(0, 1)

    def b(self, r: float, theta: float) -> float:
        """
        Magnetic‑field magnitude **B** at the specified location.

        The value is obtained from the internal interpolator ``_b_int`` which
        expects arguments in the order ``(theta, r)``.

        :param r: Radial coordinate measured from the centre of the device,
                  **in inches**.
        :type r: float
        :param theta: Azimuthal angle measured from the reference direction,
                      **in radians**.
        :type theta: float
        :return: Magnetic‑field magnitude :math:`B(r,\\theta)`.
        :rtype: float
        """
        return float(self._b_int(theta, r)[0][0])

    def dbdr(self, r: float, theta: float) -> float:
        """
        Radial derivative of the magnetic field, :math:`\\partial B/\\partial r`.

        The derivative is evaluated by the internal routine ``_dbr_int`` (again
        ordered as ``(theta, r)``).

        :param r: Radial coordinate (**inches**).
        :type r: float
        :param theta: Azimuthal angle (**radians**).
        :type theta: float
        :return: Radial derivative of the magnetic field,
                 :math:`\\frac{\\partial B}{\\partial r}(r,\\theta)`.
        :rtype: float
        """
        return float(self._dbr_int(theta, r)[0][0])

    def dbdt(self, r: float, theta: float) -> float:
        """
        Angular derivative of the magnetic field,
        :math:`\\partial B/\\partial \\theta`.

        The derivative is obtained from ``_dbt_int`` (ordered ``(theta, r)``).

        :param r: Radial coordinate (**inches**).
        :type r: float
        :param theta: Azimuthal angle (**radians**).
        :type theta: float
        :return: Angular derivative of the magnetic field,
                 :math:`\\frac{\\partial B}{\\partial \\theta}(r,\\theta)`.
        :rtype: float
        """
        return float(self._dbt_int(theta, r)[0][0])