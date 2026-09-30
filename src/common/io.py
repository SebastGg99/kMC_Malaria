"""Entrada/salida de corridas: guardado, carga y reconstrucción desde metadata.

Unifica código que estaba duplicado en:
- `isotropicRun.py`, `isotropicRun_v2.py`, `anisotropicRun.py`
  (`save_pickle`, `save_json`, `save_conversion_history`)
- `morphologies.ipynb` y `results_1.ipynb` (`reconstruct_kmc_from_metadata`)
"""

import json
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .legacy import load_legacy_pickle

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Guardado / carga básicos
# ---------------------------------------------------------------------------
def save_pickle(obj: Any, path: PathLike) -> Path:
    """Serializa `obj` con pickle y devuelve la ruta."""
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: PathLike) -> Any:
    """Carga un pickle, admitiendo también los generados con módulos antiguos."""
    return load_legacy_pickle(path)


def save_json(obj: dict, path: PathLike) -> Path:
    """Guarda un diccionario como JSON legible (UTF-8, indentado)."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: PathLike) -> dict:
    """Carga un JSON como diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Historia de conversión
# ---------------------------------------------------------------------------
def conversion_history_to_frame(conv_hist) -> pd.DataFrame:
    """Convierte `stats["conversion_history"]` en un DataFrame (time, conversion).

    Acepta tuplas (t, conv) —formato real del motor— o (t, _, conv) —formato de
    snapshots—, como la versión robusta de `anisotropicRun.py`.
    """
    if not conv_hist:
        raise ValueError("stats['conversion_history'] está vacío.")

    first_item = conv_hist[0]
    if len(first_item) >= 3:
        conv_df = pd.DataFrame(conv_hist, columns=["time", "_unused", "conversion"])
        conv_df = conv_df[["time", "conversion"]]
    elif len(first_item) == 2:
        conv_df = pd.DataFrame(conv_hist, columns=["time", "conversion"])
    else:
        raise ValueError(
            "Formato no reconocido en stats['conversion_history']. "
            "Se esperaba una secuencia de tuplas de longitud 2 o 3."
        )

    conv_df["time"] = pd.to_numeric(conv_df["time"], errors="coerce")
    conv_df["conversion"] = pd.to_numeric(conv_df["conversion"], errors="coerce")
    if conv_df["time"].isna().any() or conv_df["conversion"].isna().any():
        raise ValueError("No se pudo convertir correctamente time/conversion a valores numéricos.")
    return conv_df


def save_conversion_history(stats: Dict, run_dir: PathLike) -> Tuple[Path, Path]:
    """Guarda la historia de conversión como CSV y NPY (mismo formato que *Run.py)."""
    run_dir = Path(run_dir)
    conv_df = conversion_history_to_frame(stats["conversion_history"])
    csv_path = run_dir / "conversion_history.csv"
    npy_path = run_dir / "conversion_history.npy"
    conv_df.to_csv(csv_path, index=False)
    np.save(npy_path, conv_df.values)
    return csv_path, npy_path


# ---------------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------------
def find_repo_root(start: PathLike) -> Optional[Path]:
    """Primer ancestro de `start` (incluido) que contiene `.git`; None si no hay.

    Permite localizar `results/` y el commit sin suponer a qué profundidad está el
    paquete (hoy en la raíz; durante la refactorización vivía en refac/).
    """
    start = Path(start).resolve()
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return None


def git_commit(repo_root: Optional[PathLike]) -> Optional[str]:
    """Hash del commit actual leyendo `.git` directamente (sin llamar a `git`).

    Devuelve None si no se puede determinar (no es un repositorio, HEAD separado
    en un formato no previsto, etc.).
    """
    if repo_root is None:
        return None
    git_dir = Path(repo_root) / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head  # HEAD separado: contiene el hash directamente
        ref = head.split(" ", 1)[1]
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip()
        # La referencia puede estar empaquetada en packed-refs.
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Carga de corridas y reconstrucción del motor
# ---------------------------------------------------------------------------
def load_run(run_dir: PathLike, load_kmc: bool = False) -> Dict[str, Any]:
    """Carga metadata, snapshots y estadísticas de una carpeta de corrida.

    Si `load_kmc=True` también carga `kmc.pkl` (unos 23 MB; admite pickles antiguos).
    """
    run_dir = Path(run_dir)
    out: Dict[str, Any] = {
        "meta": load_json(run_dir / "metadata.json"),
        "snaps": load_pickle(run_dir / "snaps.pkl"),
        "stats": load_pickle(run_dir / "stats.pkl"),
    }
    if load_kmc:
        out["kmc"] = load_pickle(run_dir / "kmc.pkl")
    return out


def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Se queda solo con las claves que son campos del dataclass `cls`."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def rebuild_from_metadata(meta: Dict[str, Any], verbose: bool = True):
    """Reconstruye un motor equivalente al de la corrida, SIN volver a ejecutar run().

    Sustituye a `reconstruct_kmc_from_metadata` de los notebooks. Diferencias:
    - Usa `init_mode`/`init_kwargs` si están en la metadata. Las corridas antiguas no
      los guardaban; en ese caso se supone `flat` y se avisa, porque las corridas
      hechas con `isotropicRun_v2.py` usaban `random` (auditoria.md §6).
    - Lee `engine`, `N_bulk0`, `constant_concentration`, `use_solvent` si existen.
    """
    # Imports locales para evitar dependencias circulares common -> motores.
    from ..dynamic import KMC_BKL_Dynamic, KMCParamsDynamic, LatticeSOSDynamic
    from ..static import KMC_BKL_Static, KMCParamsStatic, LatticeSOSStatic

    engine = meta.get("engine", "static")
    size = tuple(meta["size"])
    lattice_seed = meta.get("lattice_seed", 42)

    init_mode = meta.get("init_mode")
    init_kwargs = dict(meta.get("init_kwargs", {}))
    if init_mode is None:
        init_mode = "flat"
        if verbose:
            print("⚠️ metadata sin 'init_mode' (formato antiguo): se asume 'flat'. "
                  "Si la corrida usó 'random' la red inicial reconstruida NO es la original.")

    if engine == "static":
        lat = LatticeSOSStatic(size=size, seed=lattice_seed)
        init_kwargs.setdefault("max_height", 1)
        init_kwargs.setdefault("n_seeds", meta.get("n_seeds", 0))
        lat.initialize(mode=init_mode, **init_kwargs)
        pdata = dict(meta.get("params", {}))
        if "fixed_sigma" in meta:
            pdata["fixed_sigma"] = meta["fixed_sigma"]
        params = KMCParamsStatic(**_filter_dataclass_kwargs(KMCParamsStatic, pdata))
        return KMC_BKL_Static(
            lattice=lat,
            params=params,
            N_bulk0=meta.get("N_bulk0", 2000),
            rng_seed=meta.get("rng_seed"),
            time_scale=meta.get("time_scale", 1.0),
            n_seeds=meta.get("n_seeds", 0),
            constant_concentration=meta.get("constant_concentration", True),
            use_solvent=meta.get("use_solvent", True),
            record_adsorption_probs=meta.get("record_adsorption_probs", True),
        )

    if engine == "dynamic":
        lat = LatticeSOSDynamic(size=size, seed=lattice_seed)
        lat.initialize(init_mode, **init_kwargs)
        params = KMCParamsDynamic(**_filter_dataclass_kwargs(KMCParamsDynamic, meta["params"]))
        if meta.get("fixed_sigma") is not None:
            params.fixed_sigma = meta["fixed_sigma"]
        return KMC_BKL_Dynamic(
            lattice=lat,
            params=params,
            N_bulk0=meta.get("N_bulk0", 2000),
            rng_seed=meta.get("rng_seed"),
            time_scale=meta.get("time_scale", 1.0),
            n_seeds=meta.get("n_seeds", 0),
        )

    raise ValueError(f"engine desconocido en metadata: {engine!r}")
