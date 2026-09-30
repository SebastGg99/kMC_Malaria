"""Barrido de fixed_sigma con el motor ESTÁTICO; guarda una carpeta por corrida.

Sustituye a `src/isotropicRun.py`, `src/isotropicRun_v2.py` y `src/anisotropicRun.py`,
que eran el mismo script con parámetros distintos escritos en el código. Ahora los
parámetros viven en un JSON (`--config`), y la CLI es la misma que antes:

    python scripts/run_sigma_scan.py --config scripts/configs/aniso_flat.json \
        --times 0:5.5:0.5 --size 80 80 --n-seeds 100 --fixed-sigma 0.3 0.7 \
        --sigma-step 0.1 --time-scale 70 --output-dir outputs_aniso_3

Salida por corrida (mismo formato que los scripts antiguos):
    conversion_history.csv / .npy, snaps.pkl, stats.pkl, kmc.pkl, metadata.json
y en la raíz: all_conversion_histories.csv

Novedades:
- metadata.json registra también engine, init_mode/init_kwargs, N_bulk0,
  constant_concentration, use_solvent, record_adsorption_probs, git_commit y la
  versión de numpy (auditoria.md §6).
- --no-adsorption-probs: omite adsorption_probs_history (1.8-2x más rápido, medido).
- --no-kmc-pickle: no guarda kmc.pkl (~23 MB por corrida); el objeto se puede
  reconstruir con src.common.io.rebuild_from_metadata.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# La carpeta que contiene `src/` (la raíz del repositorio) se añade al path; sin rutas absolutas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import parse_sigma_range, parse_times  # noqa: E402
from src.common.io import (  # noqa: E402
    find_repo_root, git_commit, load_json, save_conversion_history, save_json, save_pickle,
)
from src.static import KMC_BKL_Static, KMCParamsStatic, LatticeSOSStatic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI compatible con isotropicRun.py/anisotropicRun.py, más --config."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="JSON con los parámetros del modelo.")
    p.add_argument("--times", nargs="+", required=True,
                   help="Tiempos de snapshot. Ej: --times 0:40:1  o  --times 0 1 2 3 4")
    p.add_argument("--size", nargs=2, type=int, required=True, metavar=("NX", "NY"))
    p.add_argument("--n-seeds", type=int, required=True,
                   help="Semillas iniciales (se pasan a la red y al motor, como antes).")
    p.add_argument("--fixed-sigma", nargs=2, required=True, metavar=("SIGMA0", "SIGMAF"))
    p.add_argument("--sigma-step", type=float, default=1.0)
    p.add_argument("--time-scale", type=float, required=True)
    p.add_argument("--output-dir", type=str, default="outputs_sigma_scan")
    p.add_argument("--rng-seed", type=int, default=123)
    p.add_argument("--lattice-seed", type=int, default=42)
    p.add_argument("--no-adsorption-probs", action="store_true",
                   help="No registrar adsorption_probs_history (más rápido).")
    p.add_argument("--no-kmc-pickle", action="store_true",
                   help="No guardar kmc.pkl.")
    return p


def run_one(sigma_value: float, cfg: Dict, args, times: np.ndarray, output_root: Path) -> Dict:
    """Ejecuta una corrida para un valor de sigma y guarda sus archivos."""
    size = (args.size[0], args.size[1])
    run_id = f"size_{size[0]}x{size[1]}_sigma_{sigma_value:.6g}_seed_{args.rng_seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[{run_id}] Inicializando simulación...")

    # Red inicial: modo y kwargs del config; n_seeds de la CLI (igual que antes).
    lat_cfg = cfg.get("lattice", {})
    init_mode = lat_cfg.get("init_mode", "flat")
    init_kwargs = dict(lat_cfg.get("init_kwargs", {}))
    init_kwargs["n_seeds"] = args.n_seeds
    lat = LatticeSOSStatic(size=size, seed=args.lattice_seed)
    lat.initialize(mode=init_mode, **init_kwargs)

    params = KMCParamsStatic(**cfg["params"], fixed_sigma=sigma_value)

    ekw = dict(cfg.get("engine_kwargs", {}))
    ekw.setdefault("N_bulk0", 2000)
    ekw.setdefault("constant_concentration", True)
    kmc = KMC_BKL_Static(
        lattice=lat,
        params=params,
        rng_seed=args.rng_seed,
        time_scale=args.time_scale,
        n_seeds=args.n_seeds,
        record_adsorption_probs=not args.no_adsorption_probs,
        **ekw,
    )

    logging.info(f"[{run_id}] Ejecutando KMC hasta t_end={times[-1]}...")
    snaps, stats = kmc.run(t_end=times[-1], snapshot_times=times)
    logging.info(f"[{run_id}] Simulación terminada. Guardando datos...")

    conversion_csv, conversion_npy = save_conversion_history(stats, run_dir)
    paths = {
        "conversion_csv": str(conversion_csv),
        "conversion_npy": str(conversion_npy),
        "snaps_pkl": str(save_pickle(snaps, run_dir / "snaps.pkl")),
        "stats_pkl": str(save_pickle(stats, run_dir / "stats.pkl")),
    }
    if not args.no_kmc_pickle:
        paths["kmc_pkl"] = str(save_pickle(kmc, run_dir / "kmc.pkl"))

    metadata = {
        "run_id": run_id,
        "engine": "static",
        "config": str(args.config),
        "size": [size[0], size[1]],
        "n_seeds": args.n_seeds,
        "time_scale": args.time_scale,
        "fixed_sigma": sigma_value,
        "rng_seed": args.rng_seed,
        "lattice_seed": args.lattice_seed,
        "init_mode": init_mode,
        "init_kwargs": init_kwargs,
        "N_bulk0": ekw["N_bulk0"],
        "constant_concentration": ekw["constant_concentration"],
        "use_solvent": ekw.get("use_solvent", True),
        "record_adsorption_probs": not args.no_adsorption_probs,
        "N_seed0": kmc.N_seed0,  # masa realmente sembrada (auditoria: n_seeds inertes)
        "times": times.tolist(),
        "params": {k: (float(v) if isinstance(v, np.floating) else v)
                   for k, v in params.__dict__.items()},
        "git_commit": git_commit(find_repo_root(PROJECT_ROOT)),
        "numpy_version": np.__version__,
        "paths": paths,
    }
    save_json(metadata, run_dir / "metadata.json")
    logging.info(f"[{run_id}] Guardado completo en: {run_dir}")
    return {"run_id": run_id, "sigma": sigma_value, "conversion_npy": conversion_npy}


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    cfg = load_json(args.config)
    if cfg.get("engine", "static") != "static":
        raise SystemExit("run_sigma_scan.py solo admite engine='static' (fixed_sigma es σ).")

    times = parse_times(args.times)
    sigma_values = parse_sigma_range(args.fixed_sigma, args.sigma_step)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 47)
    logging.info(f"Config     = {args.config} ({cfg.get('description', '')})")
    logging.info(f"size       = {tuple(args.size)} | n_seeds = {args.n_seeds}")
    logging.info(f"time_scale = {args.time_scale} | times = {times}")
    logging.info(f"sigma scan = {sigma_values}")
    logging.info(f"output_dir = {output_root.resolve()}")
    logging.info("=" * 47)

    results = []
    for i, sigma_value in enumerate(sigma_values, start=1):
        logging.info(f"--- Corrida {i}/{len(sigma_values)} | sigma = {sigma_value:.6g} ---")
        results.append(run_one(sigma_value, cfg, args, times, output_root))

    # CSV maestro con todas las historias de conversión
    rows = []
    for r in results:
        for t, c in np.load(r["conversion_npy"]):
            rows.append({"run_id": r["run_id"], "sigma": r["sigma"],
                         "time": float(t), "conversion": float(c)})
    master_csv = output_root / "all_conversion_histories.csv"
    pd.DataFrame(rows).to_csv(master_csv, index=False)
    logging.info(f"CSV maestro guardado en: {master_csv}")


if __name__ == "__main__":
    main()
