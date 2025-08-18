.. _fields_module:

ops.cyclotron.analysis.fields
==============================

.. currentmodule:: ops.cyclotron.analysis.fields

Overview
--------

The :mod:`ops.cyclotron.analysis.fields` module provides a single public helper,
:func:`synchronize_field`.  It modifies an *iron* magnetic field so that its
**first moment** (the radial synchronous field) is identical to a user‑supplied
target field.

| **Array layout** – ``MagneticField.values`` is a 2‑D ``numpy.ndarray`` with  
| ``shape = (N_θ, N_r)`` where  

* **first index (``θ``)** – azimuthal angle in **radians**  
* **second index (``r``)** – radius in **inches**

The function returns a brand‑new :class:`~ops.cyclotron.analysis.MagneticField`
instance; the original ``iron_field`` is left unchanged.

The correction is additive and is broadcast over the ``θ``‑axis with
``np.einsum('ij,j->ij', ...)`` – mathematically:

.. math::

    \Delta B(r) = B_{\text{sync}}(r) -
    \langle B_{\text{iron}}(r) \rangle ,\qquad
    B_{\text{new}}(\theta,r) = B_{\text{iron}}(\theta,r) + \Delta B(r).

----

.. autofunction:: synchronize_field
   :noindex:

----

Example usage
-------------

.. code-block:: python

    >>> import numpy as np
    >>> from ops.cyclotron.analysis import MagneticField, synchronize_field
    >>> # `metadata` describes the grid (θ‑steps, r‑steps, units, …)
    >>> iron = MagneticField(metadata, values)          # shape (Nθ, Nr) → (θ, r)
    >>> # Desired first‑moment (radial) field – one value per radial point
    >>> target = np.linspace(1.0, 2.0, iron.values.shape[1])
    >>> synced = synchronize_field(iron, target)
    >>> # The new field now has the requested first moment
    >>> np.allclose(synced.first_moment(), target)
    True
    >>> # The original iron field is unchanged
    >>> np.array_equal(iron.values, values)          # `values` is the original array
    True

----

See also
--------

* :mod:`ops.cyclotron.analysis.magnetic_field` – definition of the
  :class:`~ops.cyclotron.analysis.MagneticField` class.
* :mod:`ops.cyclotron.analysis` – top‑level package overview.

----

Reference
---------

.. automodule:: ops.cyclotron.analysis.fields
    :members:
    :undoc-members:
    :show-inheritance: