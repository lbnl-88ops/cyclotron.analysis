from typing import Dict, List, Optional
from logging import getLogger

import numpy as np
from scipy.optimize import minimize

from cyclotron.analysis.model import TrimCoil

_log = getLogger(__name__)

def solve_coil_currents(
        field: np.ndarray, 
        trim_coils: List[TrimCoil],
        *,
        use_coils: Optional[List[int]] = None) -> Dict[TrimCoil, float] | None:
    if use_coils is None:
        use_coils = [coil.number for coil in trim_coils]
    if len(use_coils) == 0:
        return None
    use_coils = sorted(use_coils)

    solved_currents: Dict[TrimCoil, float] = {}
    k = len(field)
    # Bounds for coils with minimum values
    comparison_required: bool = False
    positive_bounds = []
    negative_bounds = []

    sorted_coils = [coil for coil in sorted(trim_coils, key=lambda x: x.number)
                    if coil.number in use_coils]

    A = np.zeros((len(sorted_coils), k))

    for i, coil in enumerate(sorted_coils):
        min_current, max_current = coil.current_limits
        if min_current is None:
            positive_bounds.append((-max_current, max_current))
            negative_bounds.append((-max_current, max_current))
        else:
            positive_bounds.append((min_current, max_current))
            negative_bounds.append((-max_current, -min_current))
            comparison_required = True

        A[i, :] = coil.db_di()[:k]

    def residual(x) -> float: 
        r = np.power(field - np.einsum('ij,i->j', A, x), 2)
        weights = np.array([10,10,10,10,10,10,10,10,10,10,10,
                  11,12,13,14,15,16,17,18,19,20,21,22,
                  23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39] + 15*[0])[:k]
        return float(np.sqrt(np.sum(weights * r)))
    positive_result = minimize(residual, np.zeros((len(sorted_coils))), bounds=positive_bounds)
    # Set to positive result unless a comparison is required,
    # to avoid extra if/then trees and simplify remaining code
    negative_result = positive_result
    if comparison_required: 
        x = np.zeros((len(sorted_coils)))
        if positive_result['success']:
            x = positive_result['x']
        negative_result = minimize(residual, x, bounds=negative_bounds) 

    if not positive_result['success'] and negative_result['success']:
        _log.warning('Trim coil fit failed')
        return None
    elif not positive_result['success']:
        result = negative_result
    elif not negative_result['success']:
        result = positive_result
    else:
        result = positive_result if positive_result['fun'] < negative_result['fun'] else negative_result    

    x = result['x']
    for i, current in enumerate(x):
        coil = sorted_coils[i]
        if coil.number in use_coils:
            solved_currents[coil] = current
        else:
            solved_currents[coil] = 0

    return solved_currents
