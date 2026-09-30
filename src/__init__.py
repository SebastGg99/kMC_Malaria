"""Paquete kMC-BKL-SOS con dos líneas de motor.

- `src.dynamic`: concentración dinámica (origen: bkl.py, lattice.py, params.py)
- `src.static`:  sobresaturación fija / concentración constante, anisotropía x/y,
                 solvente e incorporación (origen: bkl_v4.py + opciones de bkl_v5.py,
                 lattice_v3.py, params_v4.py)
- `src.common`:  numérica, datos de referencia, observables, E/S y pickles antiguos
- `src.plotting`: Plotter unificado (estilos "classic", "v2", "academic", "paper")

Imports relativos: basta con que la carpeta que CONTIENE `src/` esté en sys.path
(antes hacía falta añadir `src/` misma).

Los nombres heredados (KMC_BKL, KMC_BKL_v4, Plotter_v2, ...) se exportan como alias
durante la transición para que los notebooks sigan funcionando con `from src import *`.
"""

# ---- Nombres canónicos ----
from .dynamic import (
    KMCParamsDynamic,
    LatticeSOSDynamic,
    KMC_BKL_Dynamic,
    SelectiveKMC_Dynamic,
    KMC_NoDesNoMig_Dynamic,
)
from .static import (
    KMCParamsStatic,
    LatticeSOSStatic,
    KMC_BKL_Static,
    SelectiveKMC_Static,
    KMC_NoDesNoMig_Static,
)
from .plotting import Plotter, Plotter_v2, Plotter_v3
from .common import (
    _safe_exp,
    _finite_or_zero,
    FACE_DATA,
    mean_height,
    roughness,
    step_density,
    count_by_coordination,
    load_legacy_pickle,
)

# ---- Alias heredados: línea dinámica (bkl.py / lattice.py / params.py) ----
KMCParams = KMCParamsDynamic
LatticeSOS = LatticeSOSDynamic
KMC_BKL = KMC_BKL_Dynamic
SelectiveKMC = SelectiveKMC_Dynamic
KMC_NoDesNoMig = KMC_NoDesNoMig_Dynamic

# ---- Alias heredados: línea estática (bkl_v4.py / lattice_v3.py / params_v4.py) ----
KMCParams_v4 = KMCParamsStatic
LatticeSOS_v3 = LatticeSOSStatic
LatticeSOS_v4 = LatticeSOSStatic
KMC_BKL_v4 = KMC_BKL_Static
SelectiveKMC_v4 = SelectiveKMC_Static
KMC_NoDesNoMig_v4 = KMC_NoDesNoMig_Static
# KMC_BKL_v5 no se exporta: no tenía consumidores. Equivalente:
#   KMC_BKL_Static(..., record_adsorption_probs=False) + kmc.crystal_fraction_percent

__all__ = [
    # canónicos
    "KMCParamsDynamic", "LatticeSOSDynamic", "KMC_BKL_Dynamic",
    "SelectiveKMC_Dynamic", "KMC_NoDesNoMig_Dynamic",
    "KMCParamsStatic", "LatticeSOSStatic", "KMC_BKL_Static",
    "SelectiveKMC_Static", "KMC_NoDesNoMig_Static",
    "Plotter", "Plotter_v2", "Plotter_v3",
    "_safe_exp", "_finite_or_zero", "FACE_DATA",
    "mean_height", "roughness", "step_density", "count_by_coordination",
    "load_legacy_pickle",
    # alias heredados
    "KMCParams", "LatticeSOS", "KMC_BKL", "SelectiveKMC", "KMC_NoDesNoMig",
    "KMCParams_v4", "LatticeSOS_v3", "LatticeSOS_v4", "KMC_BKL_v4",
    "SelectiveKMC_v4", "KMC_NoDesNoMig_v4",
]
