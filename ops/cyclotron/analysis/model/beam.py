from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BeamParameters:
    """Physical parameters of a charged particle beam.

    Attributes
    ----------
    mass : float
        Rest mass of the particle **in atomic mass units (amu)**.
    charge : int
        Net electric charge expressed in units of the elementary charge ``e``.
        For a proton this is ``+1``; for an electron ``‑1``; for a fully‑stripped
        carbon ion ``+6`` and so on.
    energy : float
        Kinetic energy of the beam **in mega‑electron‑volts (MeV)**.

    Notes
    -----
    The class is frozen (immutable) and uses ``slots`` to keep the memory
    footprint low – it behaves like a simple ``namedtuple`` with the added
    benefit of automatic ``__repr__`` and field metadata.
    """
    mass: float
    charge: int
    energy: float
