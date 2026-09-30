"""Parámetros de la línea ESTÁTICA: `KMCParamsStatic`.

Origen: `src/params_v4.py` (clase `KMCParams_v4`). Sin cambios salvo el nombre.

`fixed_sigma` se interpreta como σ = C/C_eq − 1, y el motor usa S = ln(1 + σ)
en las tasas (auditoria.md §5.2.3).
"""

from dataclasses import dataclass
from typing import Optional


_DISPLAY_TEMPERATURE_K = 298.15


@dataclass
class KMCParamsStatic:
    """Parametros efectivos para kMC-SOS con anisotropia de forma y solvente.

    Los factores de solvente multiplican las tasas base:
    - 1.0: sin efecto
    - <1: ralentiza
    - >1: acelera
    """

    K0_plus: float
    E_pb_over_kT_x: float
    E_pb_over_kT_y: float
    phi_over_kT: float
    delta_x: float
    delta_y: float
    V: float
    C_eq: float
    # Incorporation prefactor (0.0 keeps it disabled unless explicitly set).
    K_inc_plus: float = 0.0
    fixed_sigma: Optional[float] = None
    S_floor: float = -5.0
    S_ceil: float = 8.0

    # Factores de solvente (adimensionales)
    solvent_ads_factor: float = 1.0
    solvent_des_factor: float = 1.0
    solvent_mig_factor: float = 1.0
    solvent_inc_factor: float = 1.0

    @property
    def T(self) -> float:
        """Shim de compatibilidad solo-lectura para display."""
        return _DISPLAY_TEMPERATURE_K
