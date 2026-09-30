"""Observables de superficie reutilizables por ambas líneas.

Origen: los métodos `mean_height`, `roughness`, `step_density` y
`count_by_coordination` de `lattice_v2.py` (descartado). Se perdieron al pasar a
`lattice_v3.py`; aquí se recuperan como funciones independientes de la versión de red.

IMPORTANTE: `step_density` y `count_by_coordination` delegan en
`lattice.desorption_bonds` / `lattice.adsorption_bonds`, así que su resultado depende
de la convención de conteo de enlaces de cada red (ver auditoria.md §5.1):
- dinámica  (`LatticeSOSDynamic`): vecinos con h >= h0
- estática  (`LatticeSOSStatic`):  vecinos con h == nivel
"""

from typing import Dict

import numpy as np


def mean_height(heights: np.ndarray) -> float:
    """Altura media de la superficie (en capas)."""
    return float(np.mean(heights))


def roughness(heights: np.ndarray) -> float:
    """Rugosidad como desviación estándar de las alturas (ancho de interfaz)."""
    return float(np.std(heights))


def step_density(lattice) -> float:
    """Fracción de columnas ocupadas cuyo tope tiene exactamente 1 enlace lateral.

    Replica `LatticeSOS_v2.step_density`: cuenta los sitios con h > 0 y
    `desorption_bonds == 1` sobre el total de sitios ocupados.
    """
    count = 0
    total = 0
    for site in lattice.get_sites():
        if lattice.get_height(site) > 0:
            if lattice.desorption_bonds(site) == 1:
                count += 1
            total += 1
    return count / total if total > 0 else 0.0


def count_by_coordination(lattice) -> Dict[int, int]:
    """Histograma de sitios por número de enlaces de adsorción (0..4).

    Replica `LatticeSOS_v2.count_by_coordination`. Para la red estática se usa el
    total ix+iy de `adsorption_bonds`.
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for site in lattice.get_sites():
        i = int(lattice.adsorption_bonds(site))
        counts[min(max(i, 0), 4)] += 1
    return counts
