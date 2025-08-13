from logging import getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.interpolate import PchipInterpolator
from cyclotron.analysis.model import TrimCoil

_log = getLogger(__name__)

# File format constants
_N_LEVELS: int = 5
_N_TRIM_COILS: int = 17
_JUNK_COLUMNS: int = 11
_VALUES_PER_LINE: int = 7
_LENGTH_OF_VALUES: int = 10
_VALUES_PER_FIELD: int = 70
_LINES_PER_LEVEL = int(_VALUES_PER_FIELD/_VALUES_PER_LINE)

CURRENT_VALUES = [765, 1120, 1470, 1840, 2090]
COIL_TURNS = [3, 3, 4, 4, 3, 4, 3, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2]

def _split_by_n(s, n) -> List[str]:
    return [s[i:i + n] for i in range(0, len(s), n)]

def _read_next_field(f) -> np.ndarray:
    field_lines = [f.readline() for _ in range(_LINES_PER_LEVEL)]
    field_values = []
    for line in field_lines:
        for value in _split_by_n(line[:-_JUNK_COLUMNS], _LENGTH_OF_VALUES):
            field_values.append(value.rstrip())
    return np.array(field_values)


def read_trim_coil_data(coil_data_file: Path) -> np.ndarray:
    coil_fields = np.zeros((_N_LEVELS, _N_TRIM_COILS + 2, _VALUES_PER_FIELD))

    with open(coil_data_file, 'r') as f:
        for i in range(_N_LEVELS):
            # main field average values
            coil_fields[i,0,:] = _read_next_field(f)
        # db/di values
        for i in range(_N_LEVELS):
            # main field db/di
            coil_fields[i,1,:] = _read_next_field(f)
            for j in range(_N_TRIM_COILS):
                coil_fields[i, j + 2, :] = _read_next_field(f)

    return coil_fields

def calculate_trim_coil_fields(coil_field_tensor: np.ndarray,
                               main_current: float) -> List[TrimCoil]:
    current_idxs = range(len(CURRENT_VALUES))
    fields = []

    for coil in range(1, _N_TRIM_COILS + 1):
        current_index = np.interp(main_current, CURRENT_VALUES, current_idxs)
        field_interpolator = PchipInterpolator(
            current_idxs, coil_field_tensor[:,coil + 1,:],
            axis=0)
        fields.append(TrimCoil(coil, field_interpolator(current_index)*COIL_TURNS[coil - 1]/1500))
    return fields


