import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/sgaviria/MalariaProject/')

from src import *


# =========================
# PARSEO DE ARGUMENTOS
# =========================
def parse_times(values: List[str]) -> np.ndarray:
    """
    Soporta:
      --times 0 1 2 3 4
      --times 0,1,2,3,4
      --times 0:40:1
    """
    if len(values) == 1 and ":" in values[0]:
        parts = values[0].split(":")
        if len(parts) != 3:
            raise ValueError("Formato inválido para --times. Usa start:end:step.")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2])
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
    """
    Recibe:
      --fixed-sigma sigma0 sigmaf

    Genera valores en el intervalo [sigma0, sigmaf] con paso sigma_step.
    """
    if len(values) != 2:
        raise ValueError("--fixed-sigma requiere exactamente dos valores: sigma0 sigmaf")

    sigma0 = float(values[0])
    sigmaf = float(values[1])

    if sigma_step <= 0:
        raise ValueError("--sigma-step debe ser mayor que 0.")
    if sigmaf < sigma0:
        raise ValueError("sigmaf debe ser mayor o igual que sigma0.")

    return np.arange(sigma0, sigmaf + 0.5 * sigma_step, sigma_step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta simulaciones KMC para un barrido de fixed_sigma y guarda resultados por corrida."
    )

    parser.add_argument(
        "--times",
        nargs="+",
        required=True,
        help="Tiempos de snapshot. Ej: --times 0:40:1  o  --times 0 1 2 3 4",
    )

    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        required=True,
        metavar=("NX", "NY"),
        help="Tamaño de la red. Ej: --size 70 70",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        required=True,
        help="Número de seeds iniciales. Ej: --n-seeds 100",
    )

    parser.add_argument(
        "--fixed-sigma",
        nargs=2,
        required=True,
        metavar=("SIGMA0", "SIGMAF"),
        help="Rango de fixed_sigma. Ej: --fixed-sigma 0.5 5.0",
    )

    parser.add_argument(
        "--sigma-step",
        type=float,
        default=1.0,
        help="Paso para el barrido de fixed_sigma. Default: 1.0",
    )

    parser.add_argument(
        "--time-scale",
        type=float,
        required=True,
        help="Valor de time_scale para KMC_BKL_v4.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_sigma_scan",
        help="Directorio raíz para guardar los resultados.",
    )

    parser.add_argument(
        "--rng-seed",
        type=int,
        default=123,
        help="Seed base para el RNG. Default: 123",
    )

    parser.add_argument(
        "--lattice-seed",
        type=int,
        default=42,
        help="Seed para la red inicial. Default: 42",
    )

    return parser


# =========================
# UTILIDADES DE GUARDADO
# =========================
def save_pickle(obj, path: Path) -> Path:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(obj: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def save_conversion_history(stats_run, run_dir: Path) -> Tuple[Path, Path]:
    """
    Guarda conversion_history como CSV y NPY.
    Se asume formato: [(t, conv), ...]
    """
    conv_hist = stats_run["conversion_history"]
    #arr = np.asarray(conv_hist, dtype=float)

    #if arr.ndim != 2 or arr.shape[1] < 2:
    #    raise ValueError("stats_run['conversion_history'] no tiene el formato esperado.")

    # conv_df = pd.DataFrame(
    #     {
    #         "time": arr[:, 0],
    #         "conversion": arr[:, 2],
    #     }
    # )

    conv_df = pd.DataFrame(conv_hist, columns=["time", "conversion"])

    csv_path = run_dir / "conversion_history.csv"
    npy_path = run_dir / "conversion_history.npy"

    conv_df.to_csv(csv_path, index=False)
    np.save(npy_path, conv_df.values)

    return csv_path, npy_path


# =========================
# CORRIDA DE UNA SIMULACIÓN
# =========================
def run_single_sigma_simulation(
    sigma_value: float,
    times: np.ndarray,
    size: Tuple[int, int],
    n_seeds: int,
    time_scale: float,
    lattice_seed: int,
    rng_seed: int,
    output_root: Path,
):
    run_id = f"size_{size[0]}x{size[1]}_sigma_{sigma_value:.6g}_seed_{rng_seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[{run_id}] Inicializando simulación...")

    Lat = LatticeSOS_v4(size=size, seed=lattice_seed)
    Lat.initialize(mode="flat", max_height=1, n_seeds=n_seeds)

    params = KMCParams_v4(
        K0_plus=0.2116718,
        K_inc_plus=0.5069325183371898,
        E_pb_over_kT_x=1.2704636027368605,
        E_pb_over_kT_y=1.2704636027368605,
        phi_over_kT=1.4792478012079329,
        delta_x=1.3,
        delta_y=1.3,
        V=1,
        C_eq=15.0,
        fixed_sigma=sigma_value,
        S_floor=-5.0,
        S_ceil=9.0,
    )

    kmc_run = KMC_BKL_v4(
        lattice=Lat,
        params=params,
        N_bulk0=2000,
        rng_seed=rng_seed,
        time_scale=time_scale,
        n_seeds=n_seeds,
        constant_concentration=True,
    )

    logging.info(f"[{run_id}] Ejecutando KMC hasta t_end={times[-1]}...")
    snaps_run, stats_run = kmc_run.run(t_end=times[-1], snapshot_times=times)
    logging.info(f"[{run_id}] Simulación terminada. Guardando data...")

    conversion_csv, conversion_npy = save_conversion_history(stats_run, run_dir)
    snaps_pkl = save_pickle(snaps_run, run_dir / "snaps.pkl")
    stats_pkl = save_pickle(stats_run, run_dir / "stats.pkl")
    kmc_pkl = save_pickle(kmc_run, run_dir / "kmc.pkl")

    metadata = {
        "run_id": run_id,
        "size": [size[0], size[1]],
        "n_seeds": n_seeds,
        "time_scale": time_scale,
        "fixed_sigma": sigma_value,
        "rng_seed": rng_seed,
        "lattice_seed": lattice_seed,
        "times": times.tolist(),
        "paths": {
            "conversion_csv": str(conversion_csv),
            "conversion_npy": str(conversion_npy),
            "snaps_pkl": str(snaps_pkl),
            "stats_pkl": str(stats_pkl),
            "kmc_pkl": str(kmc_pkl),
        },
    }
    metadata_json = save_json(metadata, run_dir / "metadata.json")

    logging.info(f"[{run_id}] Guardado completo en: {run_dir}")
    logging.info(f"[{run_id}]  - conversion_history.csv")
    logging.info(f"[{run_id}]  - conversion_history.npy")
    logging.info(f"[{run_id}]  - snaps.pkl")
    logging.info(f"[{run_id}]  - stats.pkl")
    logging.info(f"[{run_id}]  - kmc.pkl")
    logging.info(f"[{run_id}]  - metadata.json")

    return {
        "run_id": run_id,
        "sigma": sigma_value,
        "run_dir": run_dir,
        "conversion_csv": conversion_csv,
        "conversion_npy": conversion_npy,
        "snaps_pkl": snaps_pkl,
        "stats_pkl": stats_pkl,
        "kmc_pkl": kmc_pkl,
        "metadata_json": metadata_json,
    }


# =========================
# MAIN
# =========================
def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    times = parse_times(args.times)
    size = (args.size[0], args.size[1])
    n_seeds = args.n_seeds
    sigma_values = parse_sigma_range(args.fixed_sigma, args.sigma_step)
    time_scale = args.time_scale
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("===============================================")
    logging.info("Inicio del barrido de simulaciones")
    logging.info(f"size       = {size}")
    logging.info(f"n_seeds    = {n_seeds}")
    logging.info(f"time_scale = {time_scale}")
    logging.info(f"times      = {times}")
    logging.info(f"sigma scan = {sigma_values}")
    logging.info(f"output_dir = {output_root.resolve()}")
    logging.info("===============================================")

    all_results = []

    for i, sigma_value in enumerate(sigma_values, start=1):
        logging.info(f"--- Corrida {i}/{len(sigma_values)} | sigma = {sigma_value:.6g} ---")
        result = run_single_sigma_simulation(
            sigma_value=sigma_value,
            times=times,
            size=size,
            n_seeds=n_seeds,
            time_scale=time_scale,
            lattice_seed=args.lattice_seed,
            rng_seed=args.rng_seed,
            output_root=output_root,
        )
        all_results.append(result)
        logging.info(f"--- Corrida {i}/{len(sigma_values)} finalizada ---")

    # Consolidado maestro de conversion
    master_rows = []
    for r in all_results:
        conv_arr = np.load(r["conversion_npy"])
        for t, c in conv_arr:
            master_rows.append(
                {
                    "run_id": r["run_id"],
                    "sigma": r["sigma"],
                    "time": float(t),
                    "conversion": float(c),
                }
            )

    master_df = pd.DataFrame(master_rows)
    master_csv = output_root / "all_conversion_histories.csv"
    master_df.to_csv(master_csv, index=False)

    logging.info(f"CSV maestro guardado en: {master_csv}")
    logging.info("Todas las simulaciones terminaron correctamente.")


if __name__ == "__main__":
    main()