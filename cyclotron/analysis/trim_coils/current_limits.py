from typing import List

from cyclotron.analysis.model import TrimCoil

def _get_default_limits(coil_number: int) -> tuple:
    match coil_number:
        case 1:
            return (None, 750.0)
        case 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 11:
            return (None, 750)
        case 12 | 13 | 14 | 15: 
            return (None, 2000)
        case 16 | 17:
            return (None, 2500)
        case _:
            return (0, 0)

def update_current_limits(trim_coils: List[TrimCoil],
                          use_trim_coils: List[int]) -> None:
    for coil in trim_coils:
        if coil.number in use_trim_coils:
            coil.set_current_limits(_get_default_limits(coil.number))
        else:
            coil.set_current_limits((None, 0))