"""Helpers de línea de comandos compartidos por los scripts de ejecución.

Antes estaban copiados en los 7 scripts (`parse_times`/`_parse_times`,
`_parse_bool`, `_parse_sizes`, `parse_sigma_range`).
"""

import re
from typing import List, Tuple

import numpy as np


def parse_times(values: List[str]) -> np.ndarray:
    """Tiempos de snapshot.

    Formatos admitidos:
      --times 0 1 2 3 4
      --times 0,1,2,3,4
      --times 0:40:1        (inicio:fin:paso, fin excluido como en np.arange)
    """
    if len(values) == 1 and ":" in values[0]:
        parts = values[0].split(":")
        if len(parts) != 3:
            raise ValueError("Formato inválido para --times. Usa inicio:fin:paso.")
        start, end, step = (float(p) for p in parts)
        if step <= 0:
            raise ValueError("El paso de --times debe ser mayor que 0.")
        return np.arange(start, end, step)

    times: List[float] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                times.append(float(chunk))
    if not times:
        raise ValueError("--times requiere al menos un valor.")
    return np.array(times, dtype=float)


def parse_sigma_range(values: List[str], sigma_step: float) -> np.ndarray:
    """Barrido de σ en [sigma0, sigmaf] con paso `sigma_step` (extremo incluido)."""
    if len(values) != 2:
        raise ValueError("--fixed-sigma requiere exactamente dos valores: sigma0 sigmaf")
    sigma0, sigmaf = float(values[0]), float(values[1])
    if sigma_step <= 0:
        raise ValueError("--sigma-step debe ser mayor que 0.")
    if sigmaf < sigma0:
        raise ValueError("sigmaf debe ser mayor o igual que sigma0.")
    # Mismo criterio que isotropicRun.py: medio paso extra para incluir sigmaf.
    return np.arange(sigma0, sigmaf + 0.5 * sigma_step, sigma_step)


def parse_sizes(values: List[str]) -> List[Tuple[int, int]]:
    """Lista de tamaños de red: `10x10 20x20` o `10x10,20x20`."""
    sizes: List[Tuple[int, int]] = []
    for value in values:
        for chunk in (c.strip() for c in value.split(",") if c.strip()):
            m = re.fullmatch(r"(\d+)x(\d+)", chunk)
            if not m:
                raise ValueError(f"Formato inválido de tamaño: {chunk}. Usa NXxNY, p. ej. 20x20")
            sizes.append((int(m.group(1)), int(m.group(2))))
    if not sizes:
        raise ValueError("--sizes requiere al menos un tamaño")
    return sizes


def parse_bool(value: str) -> bool:
    """Convierte texto a booleano (true/false, 1/0, yes/no, y/n)."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Valor booleano inválido: {value}")
