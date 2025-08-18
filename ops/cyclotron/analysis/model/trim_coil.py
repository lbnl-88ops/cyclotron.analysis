from typing import Tuple

import numpy as np


class TrimCoil:
    """Representation of a trim coil.

    Parameters
    ----------
    number : int
        Identifier for the coil (used for hashing).
    db_di : np.ndarray
        2‑D array of shape (Nθ, Nr) giving the magnetic‑field contribution
        per ampere of coil current.  The array follows the (θ, r) ordering
        used throughout the analysis package.

    Notes
    -----
    * The coil contribution to the field is obtained with
      ``b_field(current) == db_di * current``.
    * Optional current limits can be set with ``set_min_current`` /
      ``set_max_current`` (or ``set_current_limits``).  ``None`` for the lower
      limit means “no lower bound”.
    """
    def __init__(self, number: int, db_di: np.ndarray):
        self._db_di = db_di
        self._number = number
        self._min_current: float | None = None
        self._max_current = np.inf

    def b_field(self, coil_current_in_amps: float) -> np.ndarray:
        return self._db_di*coil_current_in_amps

    def db_di(self) -> np.ndarray:
        return self._db_di

    def set_min_current(self, to_set: float | None) -> None:
        self._min_current = to_set

    def set_max_current(self, to_set: float) -> None:
        self._max_current = to_set

    def set_current_limits(self, limits: Tuple[float | None, float]) -> None:
        self._min_current, self._max_current = limits

    @property
    def current_limits(self) -> Tuple[float | None, float]:
        return self._min_current, self._max_current

    @property
    def number(self) -> int:
        return self._number

    def __hash__(self) -> int:
        return hash(self._number)
