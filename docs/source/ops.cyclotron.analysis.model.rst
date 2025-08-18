.. _analysis_model_module:

ops.cyclotron.analysis.model
============================

.. currentmodule:: ops.cyclotron.analysis.model

Overview
--------

The :mod:`ops.cyclotron.analysis.model` package gathers the lightweight data
structures that describe the physical system being analysed.  At the moment it
exposes four public objects:

* :class:`BeamParameters` – basic beam properties (mass, charge, energy).  
* :class:`FieldMetadata` – geometric description of a magnetic‑field map.  
* :class:`MagneticField` – container for the field matrix together with helper
  methods for moments.  
* :class:`TrimCoil` – a simple representation of a trim‑coil that can return the
  magnetic‑field contribution for a given coil current and that stores optional
  current limits.

All classes are deliberately small (dataclasses or plain Python objects) so
they can be used from pure‑Python code, from compiled extensions, or from
Numba‑accelerated kernels without heavy dependencies.  The module is designed to
grow – new helper functions can be added later and will be picked up automatically
by the ``.. automodule`` directive at the bottom of this page.

-----------------------------------------------------------------------
BeamParameters
-----------------------------------------------------------------------

.. autoclass:: BeamParameters
   :members:
   :undoc-members:
   :show-inheritance:

   **Field table (units)**

   +-----------+----------+------+-----------------------------------+
   | Name      | Type     | Unit | Description                       |
   +===========+==========+======+===================================+
   | mass      | float    | amu  | Rest mass of the particle.        |
   +-----------+----------+------+-----------------------------------+
   | charge    | int      | e    | Charge expressed in elementary‑   |
   |           |          |      | charge units (e.g. ``+1`` for a   |
   |           |          |      | proton, ``‑1`` for an electron).  |
   +-----------+----------+------+-----------------------------------+
   | energy    | float    | MeV  | Kinetic energy of the beam.       |
   +-----------+----------+------+-----------------------------------+

   **Example**

   .. code-block:: python

       >>> from ops.cyclotron.analysis.model import BeamParameters
       >>> beam = BeamParameters(mass=1.007276, charge=+1, energy=2.5)
       >>> beam
       BeamParameters(mass=1.007276, charge=1, energy=2.5)

-----------------------------------------------------------------------
FieldMetadata
-----------------------------------------------------------------------

.. autoclass:: FieldMetadata
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**

   A small container that records the geometric parameters that correspond to the
   axes of a magnetic‑field matrix.  All lengths are expressed in **inches** and
   all angular quantities in **degrees**; conversion helpers are provided for
   radians.

   **Attributes**

   +----------------+----------+----------+--------------------------------------------+
   | Name           | Type     | Unit     | Meaning                                    |
   +================+==========+==========+============================================+
   | r_min          | float    | inches   | Radial coordinate of the first column.     |
   +----------------+----------+----------+--------------------------------------------+
   | delta_r        | float    | inches   | Spacing between successive radial points.  |
   +----------------+----------+----------+--------------------------------------------+
   | theta_min      | float    | degrees  | Azimuthal coordinate of the first row.     |
   +----------------+----------+----------+--------------------------------------------+
   | delta_theta    | float    | degrees  | Spacing between successive azimuthal points|
   +----------------+----------+----------+--------------------------------------------+

   **Properties (read‑only)**  

   * ``theta_min_rad`` – ``theta_min`` converted to radians.  
   * ``delta_theta_rad`` – ``delta_theta`` converted to radians.

   **Example**

   .. code-block:: python

       >>> from ops.cyclotron.analysis.model import FieldMetadata
       >>> meta = FieldMetadata(r_min=0.0, delta_r=0.1,
       ...                     theta_min=0.0, delta_theta=5.0)
       >>> meta.delta_theta_rad
       0.08726646259971647   # 5° → rad

-----------------------------------------------------------------------
MagneticField
-----------------------------------------------------------------------

.. autoclass:: MagneticField
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**

   Holds a 2‑D array ``values`` that represents a magnetic field sampled on a
   regular grid defined by a :class:`FieldMetadata` instance.  The grid follows the
   **(θ, r)** ordering:

   * **first axis** – azimuthal angle *θ* (in **degrees**; the metadata also
     provides radian conversion).  
   * **second axis** – radial coordinate *r* (in **inches**).

   The class caches a few intermediate quantities (first‑moment, squared
   moments, etc.) to avoid recomputation; they can be refreshed with the
   ``recalculate=True`` flag.

   **Key attributes**

   +--------------+------------------------------------+----------------------------------------------+
   | Attribute    | Type                               | Meaning                                      |
   +==============+====================================+==============================================+
   | metadata     | :class:`FieldMetadata`             | Geometry of the field grid.                  |
   +--------------+------------------------------------+----------------------------------------------+
   | values       | ``np.ndarray`` (shape ``(Nθ, Nr)``)| Field values on the grid.                    |
   +--------------+------------------------------------+----------------------------------------------+

   **Important methods**

   +--------------------------------------------------------+------------------------------------------+
   | Method                                                 | Purpose                                  |
   +========================================================+==========================================+
   | ``r_values``                                           | List of radial positions (in inches) for |
   |                                                        | each column.                             |
   +--------------------------------------------------------+------------------------------------------+
   | ``theta_values``                                       | List of azimuthal positions (in degrees) |
   |                                                        | for each row.                            |
   +--------------------------------------------------------+------------------------------------------+
   | ``first_moment(recalculate=False)`` → ``np.ndarray``   | Radial average of the field (1‑D array). |
   +--------------------------------------------------------+------------------------------------------+
   | ``first_moment_squared(recalculate=False)``            | Square of the first moment.              |
   +--------------------------------------------------------+------------------------------------------+
   | ``square(recalculate=False)`` → ``np.ndarray``         | Element‑wise square of ``values``.       |
   +--------------------------------------------------------+------------------------------------------+
   | ``second_moment(recalculate=False)`` → ``np.ndarray``  | Radial average of the squared field.     |
   +--------------------------------------------------------+------------------------------------------+

   **Example**

   .. code-block:: python

       >>> import numpy as np
       >>> from ops.cyclotron.analysis.model import FieldMetadata, MagneticField
       >>> meta = FieldMetadata(r_min=0.0, delta_r=0.1,
       ...                     theta_min=0.0, delta_theta=5.0)
       >>> r_grid = np.arange(0, 5, meta.delta_r)
       >>> theta_grid = np.arange(0, 360, meta.delta_theta)
       >>> values = np.tile(1 + 0.1 * r_grid, (len(theta_grid), 1))
       >>> field = MagneticField(meta, values)
       >>> field.first_moment()
       array([1. , 1.05, 1.1 , 1.15, 1.2 , 1.25, 1.3 , 1.35, 1.4 , 1.45])
       >>> field.second_moment()
       array([1.        , 1.1025    , 1.21      , 1.3225    , 1.44      ,
              1.5625    , 1.69      , 1.8225    , 1.96      , 2.1025    ])

-----------------------------------------------------------------------
TrimCoil
-----------------------------------------------------------------------

.. autoclass:: TrimCoil
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**

   ``TrimCoil`` represents a single **trimming coil** in the cyclotron.  The
   object stores a *field‑per‑amp* matrix ``_db_di`` (the change in magnetic field
   for a 1 A current) and the coil identifier ``number``.  It provides a simple
   interface to compute the actual field contribution for an arbitrary coil
   current and to query or set optional current limits.

   **Constructor**

   ``TrimCoil(number: int, db_di: np.ndarray)``

   * ``number`` – an integer identifier for the coil (used for hashing).  
   * ``db_di`` – a 2‑D ``numpy.ndarray`` (shape ``(Nθ, Nr)``) that gives the field
     contribution per ampere.  The array follows the same ``(θ, r)`` ordering used
     by :class:`MagneticField`.

   **Public methods**

   +-----------------------------------------------------------+--------------------------------------------------+
   | Method                                                    | Purpose                                          |
   +===========================================================+==================================================+
   | ``b_field(coil_current_in_amps: float) -> np.ndarray``    | Returns ``_db_di * coil_current_in_amps`` – the  |
   |                                                           | magnetic‑field contribution of this coil.        |
   +-----------------------------------------------------------+--------------------------------------------------+
   | ``db_di() -> np.ndarray``                                 | Return the stored field‑per‑amp matrix.          |
   +-----------------------------------------------------------+--------------------------------------------------+
   | ``set_min_current(to_set: float | None)``                 | Set a lower current limit (``None`` means no     |
   |                                                           | limit).                                          |
   +-----------------------------------------------------------+--------------------------------------------------+
   | ``set_max_current(to_set: float)``                        | Set an upper current limit (defaults to ``inf``).|
   +-----------------------------------------------------------+--------------------------------------------------+
   | ``set_current_limits(limits: Tuple[float | None, float])``| Set both limits at once.                         |
   +-----------------------------------------------------------+--------------------------------------------------+

   **Properties**

   +----------------------+-----------------------------------------------------------+
   | Property             | Meaning                                                   |
   +======================+===========================================================+
   | ``current_limits``   | Returns a ``(min, max)`` tuple of the current limits.     |
   +----------------------+-----------------------------------------------------------+
   | ``number``           | Returns the coil identifier.                              |
   +----------------------+-----------------------------------------------------------+

   **Special methods**

   * ``__hash__`` – hashes the coil by its ``number`` so ``TrimCoil`` instances can
     be used as keys in dictionaries or elements of sets.

   **Example**

   .. code-block:: python

       >>> import numpy as np
       >>> from ops.cyclotron.analysis.model import TrimCoil
       >>> # Dummy field‑per‑amp matrix (3 × 4 grid)
       >>> db_di = np.ones((3, 4)) * 0.02   # 20 mT per amp, for illustration
       >>> coil = TrimCoil(number=5, db_di=db_di)
       >>> coil.b_field(10.0)               # field for 10 A current
       array([[0.2, 0.2, 0.2, 0.2],
              [0.2, 0.2, 0.2, 0.2],
              [0.2, 0.2, 0.2, 0.2]])
       >>> coil.set_current_limits((0.5, 15.0))
       >>> coil.current_limits
       (0.5, 15.0)

-----------------------------------------------------------------------
Future additions
----------------

The module is intentionally minimal.  When new model‑related helpers (e.g.
relativistic rigidity calculators, beam‑envelopes, etc.) are added they should
live in ``ops/cyclotron/analysis/model/`` and be re‑exported from
``ops/cyclotron/analysis/model/__init__.py``.  The ``.. automodule`` directive
at the end of this page will automatically list them in the generated
documentation.

-----------------------------------------------------------------------
See also
--------

* :mod:`ops.cyclotron.analysis` – higher‑level analysis utilities that operate
  on :class:`~ops.cyclotron.analysis.model.BeamParameters`,
  :class:`~ops.cyclotron.analysis.model.MagneticField`, and
  :class:`~ops.cyclotron.analysis.model.TrimCoil`.
* :mod:`ops.cyclotron.analysis.fields` – functions for synchronising and
  converging magnetic fields.

-----------------------------------------------------------------------
Reference
---------

.. automodule:: ops.cyclotron.analysis.model
    :members:
    :undoc-members:
    :show-inheritance: