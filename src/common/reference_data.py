"""Datos de referencia del paper base (Nagpal et al., Chem. Eng. Sci. 299, 2024).

Única copia de `FACE_DATA`. Antes estaba triplicada en `utils.py`, `utils_v2.py` y
`growthRate.py`.

Contiene, por cara cristalina de lisozima HEW, los parámetros del modelo
(δ, E_pb/kT, φ/kT) y los puntos experimentales digitalizados de Yoshizaki et al.
(tasa de crecimiento frente a sobresaturación).
"""

import numpy as np

FACE_DATA = {
    "110": {
        "delta": 0.63,
        "E_pb_over_kT": 0.48,
        "phi_over_kT": 3.76,
        "exp_sigma": np.array([1.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.10, 0.17, 0.21, 0.28, 0.35, 0.42], dtype=float),
        "title": "(a) Cara 110",
        "marker": "+",
    },
    "101": {
        "delta": 0.30,
        "E_pb_over_kT": 2.12,
        "phi_over_kT": 4.27,
        "exp_sigma": np.array([1.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.00, 0.09, 0.10, 0.17, 0.21, 0.28, 0.42], dtype=float),
        "title": "(b) Cara 101",
        "marker": "*",
    },
}
