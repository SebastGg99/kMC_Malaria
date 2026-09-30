"""Utilidades numéricas robustas compartidas por ambas líneas del motor.

Origen: `src/utils.py` (idéntico en `utils_v2.py`). Sin cambios de comportamiento.
"""

import numpy as np

# np.exp(700) ~ 1e304: límite seguro en float64 antes del overflow.
_MAX_EXP_ARG = 700.0


def _safe_exp(x: float) -> float:
    """exp(x) con protección contra overflow/underflow (recorta el argumento)."""
    if x > _MAX_EXP_ARG:
        x = _MAX_EXP_ARG
    elif x < -_MAX_EXP_ARG:
        x = -_MAX_EXP_ARG
    return float(np.exp(x))


def _finite_or_zero(x: float) -> float:
    """Devuelve x si es finito; 0.0 en caso contrario (NaN o ±inf)."""
    return float(x) if np.isfinite(x) else 0.0
