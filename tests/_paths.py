"""Rutas usadas por los tests, independientes de dónde viva el paquete.

- PKG_ROOT:     carpeta que contiene `src/` (la raíz del repositorio).
- REPO_ROOT:    raíz del repositorio git (primer ancestro con `.git`).
- RESULTS_DIR:  REPO_ROOT/results (corridas históricas).
- ORIGINAL_SRC: carpeta con el código ORIGINAL anterior a la refactorización
                (bkl_v4.py, plotter.py, ...). Tras promover la refactorización vive en
                `.descartables/pre_refactor/src/`; si no existe, es None y los tests
                de paridad con el original se omiten solos.
- DESCARTABLES: REPO_ROOT/.descartables/scripts/versions (params.py, plotter_v2.py).
"""

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from src.common.io import find_repo_root  # noqa: E402

REPO_ROOT = find_repo_root(PKG_ROOT) or PKG_ROOT
RESULTS_DIR = REPO_ROOT / "results"


def _find_original_src():
    """Primera carpeta candidata que todavía contiene el código original (bkl_v4.py)."""
    candidates = [
        REPO_ROOT / ".descartables" / "pre_refactor" / "src",  # tras promover (actual)
        REPO_ROOT / "src",                                      # antes de promover
    ]
    for folder in candidates:
        if (folder / "bkl_v4.py").exists():
            return folder
    return None


ORIGINAL_SRC = _find_original_src()
DESCARTABLES = REPO_ROOT / ".descartables" / "scripts" / "versions"
