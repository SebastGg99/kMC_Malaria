"""Carga de pickles generados con los módulos antiguos (estilo script).

Los 38 `kmc.pkl` de `results/outputs_*` guardan referencias de clase con rutas como
`bkl_v4.KMC_BKL_v4`, `lattice_v3.LatticeSOS_v3`, `lattice_v3._LatticeSize` y
`params_v4.KMCParams_v4`. Tras la refactorización esos módulos ya no existen, así que
`pickle.load` normal fallaría. `load_legacy_pickle` redirige cada referencia antigua
a su clase nueva con un mapa explícito, sin necesidad de dejar archivos "shim" con
nombres `_vN` en `src/`.

`snaps.pkl` y `stats.pkl` no necesitan esto: solo contienen tuplas, listas,
diccionarios y arrays de numpy.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

# Nombre del paquete raíz ("src"), calculado para no fijarlo a mano.
_ROOT = __name__.rsplit(".", 2)[0]

# (módulo_antiguo, clase_antigua) -> (módulo_nuevo, clase_nueva)
_LEGACY_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # --- Línea estática (todas las corridas de results/) ---
    ("bkl_v4", "KMC_BKL_v4"): (f"{_ROOT}.static.engine", "KMC_BKL_Static"),
    ("bkl_v4", "SelectiveKMC_v4"): (f"{_ROOT}.static.engine", "SelectiveKMC_Static"),
    ("bkl_v4", "KMC_NoDesNoMig_v4"): (f"{_ROOT}.static.engine", "KMC_NoDesNoMig_Static"),
    ("bkl_v5", "KMC_BKL_v5"): (f"{_ROOT}.static.engine", "KMC_BKL_Static"),
    ("lattice_v3", "LatticeSOS_v3"): (f"{_ROOT}.static.lattice", "LatticeSOSStatic"),
    ("lattice_v3", "_LatticeSize"): (f"{_ROOT}.static.lattice", "_LatticeSize"),
    ("lattice_v4", "LatticeSOS_v4"): (f"{_ROOT}.static.lattice", "LatticeSOSStatic"),
    ("params_v4", "KMCParams_v4"): (f"{_ROOT}.static.params", "KMCParamsStatic"),
    # --- Línea dinámica ---
    ("bkl", "KMC_BKL"): (f"{_ROOT}.dynamic.engine", "KMC_BKL_Dynamic"),
    ("bkl", "SelectiveKMC"): (f"{_ROOT}.dynamic.engine", "SelectiveKMC_Dynamic"),
    ("bkl", "KMC_NoDesNoMig"): (f"{_ROOT}.dynamic.engine", "KMC_NoDesNoMig_Dynamic"),
    ("lattice", "LatticeSOS"): (f"{_ROOT}.dynamic.lattice", "LatticeSOSDynamic"),
    ("lattice", "_LatticeSize"): (f"{_ROOT}.dynamic.lattice", "_LatticeSize"),
    ("params", "KMCParams"): (f"{_ROOT}.dynamic.params", "KMCParamsDynamic"),
}

# Las mismas rutas antiguas pueden aparecer con prefijo de paquete ("src.bkl_v4").
_LEGACY_MAP.update({
    (f"src.{mod}", cls): target for (mod, cls), target in list(_LEGACY_MAP.items())
})


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler que redirige las clases antiguas a su nueva ubicación."""

    def find_class(self, module: str, name: str):
        # Si (módulo, clase) está en el mapa se usa el destino nuevo; si no, el normal
        # (numpy, builtins, etc.).
        module, name = _LEGACY_MAP.get((module, name), (module, name))
        return super().find_class(module, name)


def load_legacy_pickle(path: Union[str, Path]):
    """Carga un .pkl generado con los módulos antiguos (bkl_v4, lattice_v3, ...).

    También sirve para pickles nuevos: si una referencia no está en el mapa, se
    resuelve de forma normal.
    """
    with open(path, "rb") as f:
        return _LegacyUnpickler(f).load()
