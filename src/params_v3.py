from dataclasses import dataclass
from typing import Optional


_DISPLAY_TEMPERATURE_K = 298.15


@dataclass
class KMCParams_v3:
    """Parametros efectivos para kMC-SOS con anisotropia de forma.

    Esta version separa las contribuciones por direccion x/y en los
    parametros de enlace y en delta para adsorcion.
    """

    K0_plus: float
    E_pb_over_kT_x: float
    E_pb_over_kT_y: float
    phi_over_kT: float
    delta_x: float
    delta_y: float
    V: float
    C_eq: float
    fixed_sigma: Optional[float] = None
    S_floor: float = -5.0
    S_ceil: float = 8.0

    @property
    def T(self) -> float:
        """Shim de compatibilidad solo-lectura para display."""
        return _DISPLAY_TEMPERATURE_K
