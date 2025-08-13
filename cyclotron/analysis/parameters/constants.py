from typing import List

# Physical constants
C_SQUARED_IN_MEV_PER_AMU: float = 931.478
ELECTRON_MASS_IN_MEV: float = 0.51106
NUCLEON_MASS_IN_MEV: float = 938.1747
C_IN_IN_PER_SEC: float = 1.180286E10

# Conversion and expansion Constants
# Reference: Cyclotron equations, MASTER program description
MAGNETIC_REGIDITY_CONVERSION_CONSTANT:float = 1.3132653
# Reference: MASTER program binder, Appendix 3, experimental values
MAIN_CURRENT_EXPANSION_CONSTANTS: List[float] = [
    0.9893 * v for v in [
        0.0267,
        1.007,
        2.26E-2,
        -1.51574E-4,
        5.1865E-7,
        -9.624433E-10,
        9.188321E-13,
        -3.36763141E-16
        ]
    ]
# Speed of light in in/sec * kg/amu * 1/e * 10^4
INCH_GAUSS_PER_AMU: float = 1.2232799632675473E6

# 88-SPECIFIC PARAMETERS
ANGLE_OF_ZERO_INDEX_IN_DEGREES: float = 45
EXTRACTION_RADIUS_IN_INCHES: float = 38.70
