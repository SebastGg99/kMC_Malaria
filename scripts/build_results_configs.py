"""Genera scripts/configs/results/<carpeta>.json a partir de results/outputs_*/.

Cada JSON describe TODAS las corridas de una carpeta con la especificación necesaria
para reproducirlas con el código refactorizado (`reproduce_results.py`).

Por qué hace falta (hallazgo de la refactorización, ver README.md §4.2):
- metadata.json no guardaba el modo de inicio de la red.
- En la mayoría de corridas el `n_seeds` de metadata NO se aplicó (el primer
  snapshot tiene una sola partícula): se generaron sin semillas.
- Algunas carpetas mezclan corridas con distinto time_scale, K0 e inicio.

El modo de inicio EFECTIVO se deduce del primer snapshot (t=0, tras el primer
evento):
- ocupación > 30 %                       -> 'random'
- >= n_seeds sitios y forma compacta     -> 'seeds' (isla central, n_seeds de metadata)
- <= 2 sitios ocupados                   -> 'flat' sin semillas (n_seeds efectivo 0)
- otro caso                              -> 'flat' con semillas del motor (n_seeds de metadata)
La deducción se verifica con tests/test_results_reproduction.py.

Uso (desde la raíz del repositorio):
    python scripts/build_results_configs.py [--results-dir results]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import find_repo_root, load_json, load_pickle, save_json  # noqa: E402

OUT_DIR = PROJECT_ROOT / "scripts" / "configs" / "results"


def infer_init(first_heights: np.ndarray, n_seeds_meta: int):
    """Devuelve (init_mode, init_kwargs, engine_n_seeds) a partir del primer snapshot."""
    occ = first_heights > 0
    n_occ = int(occ.sum())
    if occ.mean() > 0.30:
        return "random", {"max_height": 1, "n_seeds": 0}, 0
    if n_occ <= 2:
        return "flat", {"max_height": 1, "n_seeds": 0}, 0
    ii, jj = np.nonzero(occ)
    bbox_area = (ii.max() - ii.min() + 1) * (jj.max() - jj.min() + 1)
    if n_occ >= n_seeds_meta and n_occ / bbox_area > 0.3:
        return "seeds", {"n_seeds": n_seeds_meta}, n_seeds_meta
    return "flat", {"max_height": 1, "n_seeds": n_seeds_meta}, n_seeds_meta


def build_folder(folder: Path) -> dict:
    """Especificación de todas las corridas de una carpeta outputs_*."""
    runs = []
    for run_dir in sorted(folder.glob("size_*")):
        meta = load_json(run_dir / "metadata.json")
        snaps = load_pickle(run_dir / "snaps.pkl")
        init_mode, init_kwargs, engine_n_seeds = infer_init(snaps[0][1], int(meta["n_seeds"]))
        params = {k: v for k, v in meta["params"].items() if k != "fixed_sigma"}
        runs.append({
            "run_id": meta["run_id"],
            "fixed_sigma": meta["fixed_sigma"],
            "size": meta["size"],
            "time_scale": meta["time_scale"],
            "times": meta["times"],
            "rng_seed": meta["rng_seed"],
            "lattice_seed": meta["lattice_seed"],
            "init_mode": init_mode,
            "init_kwargs": init_kwargs,
            "engine_n_seeds": engine_n_seeds,
            "n_seeds_metadata": meta["n_seeds"],
            "N_bulk0": 2000,
            "constant_concentration": True,
            "params": params,
        })
    return {
        "folder": folder.name,
        "engine": "static",
        "description": (f"Especificación reconstruida de results/{folder.name} a partir de "
                        "metadata.json + primer snapshot (init efectivo)."),
        "note": ("init_mode/engine_n_seeds son los EFECTIVOS deducidos de los datos, no los "
                 "de metadata.json. Verificado por tests/test_results_reproduction.py."),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir",
                        default=str((find_repo_root(PROJECT_ROOT) or PROJECT_ROOT) / "results"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in sorted(Path(args.results_dir).glob("outputs_*")):
        spec = build_folder(folder)
        save_json(spec, OUT_DIR / f"{folder.name}.json")
        summary = sorted({(r["init_mode"], r["engine_n_seeds"], r["n_seeds_metadata"],
                           r["time_scale"], r["params"]["K0_plus"]) for r in spec["runs"]})
        print(f"{folder.name}: {len(spec['runs'])} corridas | (init, n_seeds efectivo, "
              f"n_seeds metadata, time_scale, K0) = {summary}")


if __name__ == "__main__":
    main()
