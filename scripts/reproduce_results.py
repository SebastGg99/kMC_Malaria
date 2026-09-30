"""Reproduce (o verifica) corridas de results/ con el código refactorizado.

Lee scripts/configs/results/<carpeta>.json (generado por build_results_configs.py).

Modos:
- --check N : ejecuta solo los primeros N eventos de cada corrida y los compara con
              el historial guardado en kmc.pkl (tipo de evento y sitio exactos;
              tiempo con tolerancia relativa 1e-12, porque los tiempos difieren en
              el último bit según la versión de numpy/libm).
- sin --check: ejecuta la corrida completa y guarda los mismos archivos que los
              scripts antiguos en --output-dir.

Ejemplos (desde la raíz del repositorio):
    python scripts/reproduce_results.py --folder outputs_aniso_3 --check 300
    python scripts/reproduce_results.py --folder outputs_aniso_3 \
        --run size_80x80_sigma_0.3_seed_123 --output-dir repro_aniso_3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import (  # noqa: E402
    find_repo_root, git_commit, load_json, load_pickle, save_conversion_history, save_json, save_pickle,
)
from src.static import KMC_BKL_Static, KMCParamsStatic, LatticeSOSStatic  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "scripts" / "configs" / "results"


def build_from_spec(run: Dict, record_adsorption_probs: bool = True) -> KMC_BKL_Static:
    """Construye el motor estático exactamente como la corrida original."""
    lat = LatticeSOSStatic(size=tuple(run["size"]), seed=run["lattice_seed"])
    lat.initialize(mode=run["init_mode"], **run["init_kwargs"])
    params = KMCParamsStatic(**run["params"], fixed_sigma=run["fixed_sigma"])
    return KMC_BKL_Static(
        lattice=lat, params=params, N_bulk0=run["N_bulk0"], rng_seed=run["rng_seed"],
        time_scale=run["time_scale"], n_seeds=run["engine_n_seeds"],
        constant_concentration=run["constant_concentration"],
        record_adsorption_probs=record_adsorption_probs,
    )


def check_run(run: Dict, stored_history, n_events: int) -> str:
    """Compara los primeros n_events con el historial guardado. Devuelve 'OK' o el motivo."""
    kmc = build_from_spec(run, record_adsorption_probs=False)
    kmc.run(t_end=1e18, max_events=n_events)
    for i, (a, b) in enumerate(zip(kmc.history, stored_history[:n_events])):
        site_a = tuple(int(x) for x in a[2]) if a[2] is not None else None
        site_b = tuple(int(x) for x in b[2]) if b[2] is not None else None
        if a[1] != b[1] or site_a != site_b:
            return f"diverge en el evento {i}: {a[1]}{site_a} vs {b[1]}{site_b}"
        if abs(float(a[0]) - float(b[0])) > 1e-12 * abs(float(b[0])):
            return f"tiempo distinto en el evento {i}: {float(a[0])!r} vs {float(b[0])!r}"
    return "OK"


def full_run(run: Dict, out_root: Path) -> Path:
    """Ejecuta la corrida completa y guarda los archivos (formato de *Run.py)."""
    run_dir = out_root / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    kmc = build_from_spec(run)
    times = np.array(run["times"], dtype=float)
    snaps, stats = kmc.run(t_end=times[-1], snapshot_times=times)
    save_conversion_history(stats, run_dir)
    save_pickle(snaps, run_dir / "snaps.pkl")
    save_pickle(stats, run_dir / "stats.pkl")
    save_json({**run, "engine": "static", "git_commit": git_commit(find_repo_root(PROJECT_ROOT)),
               "numpy_version": np.__version__}, run_dir / "metadata.json")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folder", required=True, help="Carpeta de results/, p. ej. outputs_aniso_3")
    p.add_argument("--run", nargs="*", default=None, help="run_id concretos (por defecto todos)")
    p.add_argument("--check", type=int, default=None, metavar="N",
                   help="Solo verifica los primeros N eventos contra kmc.pkl")
    p.add_argument("--results-dir",
                   default=str((find_repo_root(PROJECT_ROOT) or PROJECT_ROOT) / "results"))
    p.add_argument("--output-dir", default="reproduced")
    args = p.parse_args()

    spec = load_json(CONFIG_DIR / f"{args.folder}.json")
    runs = [r for r in spec["runs"] if args.run is None or r["run_id"] in args.run]
    for run in runs:
        if args.check:
            stored = load_pickle(Path(args.results_dir) / args.folder / run["run_id"] / "kmc.pkl")
            print(f"{run['run_id']}: {check_run(run, stored.history, args.check)}")
        else:
            print(f"{run['run_id']}: guardado en {full_run(run, Path(args.output_dir))}")


if __name__ == "__main__":
    main()
