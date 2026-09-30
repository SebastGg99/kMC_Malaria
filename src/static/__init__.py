"""Línea ESTÁTICA: sobresaturación fija (fixed_sigma o concentración constante),
anisotropía x/y, solvente e incorporación.

Origen: src/bkl_v4.py (+ opciones de bkl_v5.py) + src/lattice_v3.py + src/params_v4.py.
"""

from .params import KMCParamsStatic
from .lattice import LatticeSOSStatic
from .engine import KMC_BKL_Static, SelectiveKMC_Static, KMC_NoDesNoMig_Static

__all__ = [
    "KMCParamsStatic",
    "LatticeSOSStatic",
    "KMC_BKL_Static",
    "SelectiveKMC_Static",
    "KMC_NoDesNoMig_Static",
]
