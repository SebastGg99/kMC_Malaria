"""Genera las referencias "golden" ejecutando el código ORIGINAL (Fase 0).

Se ejecuta UNA sola vez, antes de refactorizar, y sus salidas (`golden/*.npz`) se
versionan. No modifica nada fuera de `tests/golden/`.

Importa los módulos originales en estilo script:
- `ORIGINAL_SRC` (hoy `.descartables/pre_refactor/src/`, antes `<raíz>/src/`) para
  bkl.py, bkl_v4.py, bkl_v5.py, lattice*.py, params_v4.py, utils.py
- `<raíz>/.descartables/scripts/versions/` para params.py (KMCParams), que faltaba en src/

OJO: regenerarlas sobrescribe las referencias versionadas. Solo tiene sentido si
cambia la versión de numpy (ver tests/golden/ENV.txt).

Uso (desde la raíz del repositorio):
    python tests/make_golden.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
sys.path.insert(0, str(HERE.parent))
from tests._paths import DESCARTABLES, ORIGINAL_SRC  # noqa: E402

if ORIGINAL_SRC is None:
    sys.exit("El código original (src/bkl_v4.py, ...) ya no existe: no se pueden "
             "regenerar las referencias golden.")

# Rutas del código original, en estilo script (sin importar el paquete `src`).
sys.path.insert(0, str(DESCARTABLES))
sys.path.insert(0, str(ORIGINAL_SRC))
sys.path.insert(0, str(HERE))

import bkl  # noqa: E402  (motor dinámico original)
import bkl_v4  # noqa: E402  (motor estático original)
import bkl_v5  # noqa: E402  (bifurcación optimizada, solo para documentar equivalencia)
import lattice  # noqa: E402
import lattice_v4  # noqa: E402
import params  # noqa: E402  (desde .descartables)
import params_v4  # noqa: E402

from golden_cases import CASES, run_case  # noqa: E402


def original_api() -> dict:
    """Mapea las claves genéricas de golden_cases a las clases originales."""
    return {
        "dyn_params": params.KMCParams,
        "dyn_lattice": lattice.LatticeSOS,
        "dyn_engine": bkl.KMC_BKL,
        "dyn_selective": bkl.SelectiveKMC,
        "dyn_nodesnomig": bkl.KMC_NoDesNoMig,
        "sta_params": params_v4.KMCParams_v4,
        "sta_lattice": lattice_v4.LatticeSOS_v4,
        "sta_engine": bkl_v4.KMC_BKL_v4,
        "sta_selective": bkl_v4.SelectiveKMC_v4,
        "sta_nodesnomig": bkl_v4.KMC_NoDesNoMig_v4,
    }


def check_v4_v5_equivalence(api: dict) -> None:
    """Verifica la afirmación de auditoria.md §4.2: con fixed_sigma, v4 y v5 producen
    la misma trayectoria (alturas, tiempo, historial) para la misma semilla."""
    api_v5 = dict(api, sta_engine=bkl_v5.KMC_BKL_v5)
    for name in ("sta_iso_fixed_sigma", "sta_aniso_fixed_sigma", "sta_constant_conc"):
        a = run_case(name, api)
        b = run_case(name, api_v5)
        same = all(
            np.array_equal(a[k], b[k])
            for k in ("heights", "t", "hist_t", "hist_code", "hist_site", "counts", "N_inc")
        )
        same_hh = np.array_equal(a["height_history"], b["height_history"])
        print(f"  v4 == v5 [{name}]: trayectoria={same} | height_history={same_hh} | "
              f"conversión final v4={float(a['conversion_percent']):.4f}% "
              f"v5={float(b['conversion_percent']):.4f}%")


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    api = original_api()
    for name in CASES:
        res = run_case(name, api)
        np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **res)
        print(f"✔ {name}: {int(res['counts'].sum())} eventos, t={float(res['t']):.6g}")

    # Registro del entorno: las referencias solo son válidas con esta versión de numpy.
    (GOLDEN_DIR / "ENV.txt").write_text(
        f"numpy=={np.__version__}\npython=={sys.version.split()[0]}\n", encoding="utf-8"
    )

    print("\nComprobación v4 frente a v5 (auditoria.md §4.2):")
    check_v4_v5_equivalence(api)


if __name__ == "__main__":
    main()
