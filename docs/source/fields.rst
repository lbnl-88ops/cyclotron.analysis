.. _fields_module:

ops.cyclotron.analysis.fields
==============================

.. currentmodule:: ops.cyclotron.analysis.fields

Overview
--------

The :mod:`ops.cyclotron.analysis.fields` module contains two public helpers that
operate on :class:`~ops.cyclotron.analysis.MagneticField` objects:

* :func:`synchronize_field` – adds a constant correction to an iron magnetic
  field so that its **first moment** (the radial synchronous field) matches a
  user‑supplied target.

* :func:`converge_synchronous_field` – iteratively computes the **synchronous
  magnetic field** for a given set of beam parameters.  The algorithm stops
  when the magnetic‑field unit converges within a user‑specified tolerance or
  raises :class:`~ops.cyclotron.analysis.ConvergenceError` if convergence is not
  achieved.

Both functions assume that the underlying ``MagneticField.values`` array follows
the ``(θ, r)`` ordering:

* **first index** – azimuthal angle *θ* (units: **radians**)
* **second index** – radial coordinate *r* (units: **inches**)

-----------------------------------------------------------------------

synchronize_field
----------------

.. autofunction:: synchronize_field
   :noindex:

-----------------------------------------------------------------------

converge_synchronous_field
--------------------------

.. autofunction:: converge_synchronous_field
   :noindex:

-----------------------------------------------------------------------

Example usage
-------------

.. code-block:: python

    >>> import numpy as np
    >>> from ops.cyclotron.analysis import (
    ...     MagneticField,
    ...     BeamParameters,
    ...     synchronize_field,
    ...     converge_synchronous_field,
    ... )
    >>> # -----------------------------------------------------------------
    >>> # 1. Adjust an iron field so its first moment matches a target:
    >>> iron = MagneticField(metadata, values)          # shape (Nθ, Nr) → (θ, r)
    >>> target = np.linspace(1.0, 2.0, iron.values.shape[1])
    >>> synced = synchronize_field(iron, target)
    >>> np.allclose(synced.first_moment(), target)
    True
    >>> # -----------------------------------------------------------------
    >>> # 2. Compute the synchronous field for a beam:
    >>> beam = BeamParameters(...)                     # user‑defined
    >>> extraction_r = 12.5   # inches
    >>> try:
    ...     sync_field = converge_synchronous_field(
    ...         beam, iron, extraction_r, max_iter=200, tol=1e-12
    ...     )
    ... except ConvergenceError as exc:
    ...     print(f"Failed to converge: {exc}")
    >>> # sync_field is a NumPy array of shape (Nr,) representing B_sync(r)
    >>> sync_field.shape
    (iron.values.shape[1],)

-----------------------------------------------------------------------

See also
--------

* :mod:`ops.cyclotron.analysis.magnetic_field` – definition of the
  :class:`~ops.cyclotron.analysis.MagneticField` class.
* :mod:`ops.cyclotron.analysis.beam` – definition of
  :class:`~ops.cyclotron.analysis.BeamParameters`.
* :mod:`ops.cyclotron.analysis.errors` – definition of
  :class:`~ops.cyclotron.analysis.ConvergenceError`.

-----------------------------------------------------------------------

Reference
---------

.. automodule:: ops.cyclotron.analysis.fields
    :members:
    :undoc-members:
    :show-inheritance: