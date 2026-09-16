from dataclasses import dataclass
from typing import Optional


_DISPLAY_TEMPERATURE_K = 298.15


@dataclass
class KMCParams_v2:
    """
    Parámetros físicos efectivos para el modelo kMC-SOS adaptativo.

    El núcleo del motor usa únicamente magnitudes adimensionales o ya
    normalizadas por kT, por lo que T no forma parte del parámetro físico
    del modelo en esta variante.
    """

    K0_plus: float
    E_pb_over_kT: float
    phi_over_kT: float
    delta: float
    V: float
    C_eq: float
    fixed_sigma: Optional[float] = None  # Sigma estático del paper: sigma = C/C_eq - 1
    S_floor: float = -5.0
    S_ceil: float = 8.0

    @property
    def T(self) -> float:
        """
        Shim de compatibilidad de solo lectura para código de visualización
        que todavía muestra una temperatura de display.
        """
        return _DISPLAY_TEMPERATURE_K


# class LysozymeParams_v2:
#     """
#     Factorías de parámetros para lisozima HEW tetragonal calibrados a partir
#     del artículo de Nagpal et al. (2024).
#     """

#     @staticmethod
#     def face_110(
#         C_eq: float = 15.0,
#         K0_plus: float = 0.211,
#         fixed_sigma: Optional[float] = None,
#     ) -> KMCParams_v2:
#         return KMCParams_v2(
#             K0_plus=K0_plus,
#             E_pb_over_kT=0.48,
#             phi_over_kT=3.76,
#             delta=0.63,
#             V=1.0,
#             C_eq=C_eq,
#             fixed_sigma=fixed_sigma,
#             S_floor=-2.0,
#             S_ceil=8.0,
#         )

#     @staticmethod
#     def face_101(
#         C_eq: float = 15.0,
#         K0_plus: float = 0.211,
#         fixed_sigma: Optional[float] = None,
#     ) -> KMCParams_v2:
#         return KMCParams_v2(
#             K0_plus=K0_plus,
#             E_pb_over_kT=2.12,
#             phi_over_kT=4.27,
#             delta=0.30,
#             V=1.0,
#             C_eq=C_eq,
#             fixed_sigma=fixed_sigma,
#             S_floor=-2.0,
#             S_ceil=8.0,
#         )

#     @staticmethod
#     def custom(
#         E_pb: float,
#         phi: float,
#         delta: float,
#         C_eq: float = 15.0,
#         K0_plus: float = 0.211,
#         fixed_sigma: Optional[float] = None,
#         S_floor: float = -5.0,
#         S_ceil: float = 10.0,
#     ) -> KMCParams_v2:
#         return KMCParams_v2(
#             K0_plus=K0_plus,
#             E_pb_over_kT=E_pb,
#             phi_over_kT=phi,
#             delta=delta,
#             V=1.0,
#             C_eq=C_eq,
#             fixed_sigma=fixed_sigma,
#             S_floor=S_floor,
#             S_ceil=S_ceil,
#         )
