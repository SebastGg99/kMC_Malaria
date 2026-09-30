"""Corrida única (o barrido de tamaños de red) con gráficos; cualquiera de las dos líneas.

Sustituye a `src/pRun.py`, `src/pRunAni.py`, `src/pRun_v2.py` y `src/pRun_v3.py`:
- pRun.py / pRunAni.py -> configs/single_iso.json / single_aniso.json (un tamaño)
- pRun_v2.py           -> configs/sizes_fixed_seeds.json   (--sizes, n_seeds fijo)
- pRun_v3.py           -> configs/sizes_density_seeds.json (--sizes, densidad fija)
- NUEVO: la línea dinámica también tiene script (configs/dynamic_final_results.json,
  parámetros de final_results.ipynb, sección "Concentración dinámica").

Ejemplos:
    python scripts/run_single.py --config scripts/configs/single_iso.json --times 0:8:1 --size 10 10
    python scripts/run_single.py --config scripts/configs/sizes_density_seeds.json \
        --times 0:8:1 --size 10 10 --n-seeds 100 --sizes 10x10 20x20 30x30

Salida en --output-dir (antes pRun.py escribía en el directorio actual):
    <run_id>/conversion_history.csv, conversion_<run_id>.png, <snapshot>.png,
    [<gif>], metadata.json
    y con varios tamaños: all_conversion_histories.csv, runs_info.csv,
    conversion_overlay.png (antes con sufijo _1 en pRun_v3.py)

Diferencia deliberada con los originales: en pRun.py/pRunAni.py `--n-seeds` no tenía
efecto (el motor recibía n_seeds=10 escrito en el código y la red en modo 'flat' lo
ignora). Aquí el valor por defecto sale del config (10, el efectivo) y `--n-seeds`
lo sobrescribe de verdad.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import parse_bool, parse_sizes, parse_times  # noqa: E402
from src.common.io import find_repo_root, git_commit, load_json, save_json  # noqa: E402
from src.dynamic import KMC_BKL_Dynamic, KMCParamsDynamic, LatticeSOSDynamic  # noqa: E402
from src.plotting import Plotter  # noqa: E402
from src.static import KMC_BKL_Static, KMCParamsStatic, LatticeSOSStatic  # noqa: E402

# Estilo de figuras de los pRun*.py originales
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["font.size"] = 11


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="JSON con engine, parámetros y red.")
    p.add_argument("--times", nargs="+", default=["0:8:1"],
                   help="Tiempos de snapshot (lista, 'a,b,c' o inicio:fin:paso). Default 0:8:1")
    p.add_argument("--size", nargs=2, type=int, default=[10, 10], metavar=("NX", "NY"),
                   help="Tamaño de red (y referencia de densidad con seed-scaling=density).")
    p.add_argument("--sizes", nargs="+", default=None,
                   help="Varios tamaños: 10x10 20x20 ... (una corrida por tamaño).")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="Semillas iniciales (por defecto, las del config).")
    p.add_argument("--seed-scaling", choices=("fixed", "density"), default=None,
                   help="fixed: mismo n_seeds en cada tamaño (pRun_v2); density: misma "
                        "densidad que --size/--n-seeds (pRun_v3). Default: el del config.")
    p.add_argument("--snapshot-name", default="crystal_growth_prueba_run.png")
    p.add_argument("--snapshot-names", nargs="+", default=None,
                   help="Un PNG por cada tiempo de --times (solo con un tamaño).")
    p.add_argument("--gif", nargs="?", const="true", default="false",
                   help="Generar GIF (--gif, --gif true/false).")
    p.add_argument("--gif-name", default="crystal_growth_prueba_run.gif")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--rng-seed", type=int, default=None, help="Sobrescribe el del config.")
    p.add_argument("--lattice-seed", type=int, default=None, help="Sobrescribe el del config.")
    return p


def build_kmc(cfg: Dict, size: Tuple[int, int], n_seeds: int, rng_seed: int, lattice_seed: int):
    """Construye red, parámetros y motor según `cfg['engine']`."""
    engine = cfg.get("engine", "static")
    lat_cfg = cfg.get("lattice", {})
    init_mode = lat_cfg.get("init_mode", "flat")
    init_kwargs = dict(lat_cfg.get("init_kwargs", {}))
    ekw = dict(cfg.get("engine_kwargs", {}))
    ekw.pop("rng_seed", None)

    if engine == "static":
        lat = LatticeSOSStatic(size=size, seed=lattice_seed)
        init_kwargs["n_seeds"] = n_seeds
        lat.initialize(mode=init_mode, **init_kwargs)
        params = KMCParamsStatic(**cfg["params"])
        kmc = KMC_BKL_Static(lattice=lat, params=params, rng_seed=rng_seed,
                             n_seeds=n_seeds, **ekw)
    elif engine == "dynamic":
        lat = LatticeSOSDynamic(size=size, seed=lattice_seed)
        lat.initialize(init_mode, **init_kwargs)
        pdata = dict(cfg["params"])
        fixed_sigma = pdata.pop("fixed_sigma", None)
        params = KMCParamsDynamic(**pdata)
        if fixed_sigma is not None:
            # En la línea dinámica fixed_sigma es un atributo opcional y se lee como S.
            params.fixed_sigma = fixed_sigma
        kmc = KMC_BKL_Dynamic(lat, params, rng_seed=rng_seed, n_seeds=n_seeds, **ekw)
    else:
        raise ValueError(f"engine desconocido: {engine!r}")
    return kmc, init_mode, init_kwargs, params


def run_one(cfg, size, n_seeds, times, run_id, out_dir: Path, args, rng_seed, lattice_seed):
    """Ejecuta una corrida, guarda CSV/figuras/metadata y devuelve su DataFrame."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    kmc, init_mode, init_kwargs, params = build_kmc(cfg, size, n_seeds, rng_seed, lattice_seed)
    out = kmc.run(t_end=times[-1], snapshot_times=times)

    # Línea estática: (snaps, stats) con historia por evento; dinámica: solo snaps.
    if isinstance(out, tuple):
        snaps, stats = out
        conversion_history = stats["conversion_history"]
    else:
        snaps = out
        conversion_history = [(t, conv) for t, _, conv in snaps]

    conversion_df = pd.DataFrame(conversion_history, columns=["time", "conversion"])
    conversion_df["size_x"] = size[0]
    conversion_df["size_y"] = size[1]
    conversion_df["n_seeds"] = n_seeds
    conversion_df["run_id"] = run_id
    conversion_df["rng_seed"] = rng_seed
    csv_path = run_dir / "conversion_history.csv"
    conversion_df.to_csv(csv_path, index=False)

    # Conversión vs tiempo (Plotter clásico, como en los pRun*)
    Plotter(kmc, style="classic").plot_conversion(
        [(t, None, conv) for t, conv in conversion_history],
        title=f"Conversión vs Tiempo - {run_id}",
        save_path=str(run_dir / f"conversion_{run_id}.png"),
    )

    # Cristal 3D y GIF (estilo del config; los pRun* usaban Plotter_v2 = "v2")
    plotter = Plotter(kmc, style=cfg.get("plot_style", "v2"))
    if args.snapshot_names is not None:
        for t_snapshot, name in zip(times, args.snapshot_names):
            plotter.plot_crystal_3d(mode="voxel", snapshots=snaps, t_snapshot=t_snapshot,
                                    save_path=str(run_dir / name))
    else:
        plotter.plot_crystal_3d(mode="voxel", snapshots=snaps, t_snapshot=times[-1],
                                save_path=str(run_dir / args.snapshot_name))
    if parse_bool(args.gif):
        plotter.crystal_growth_gif(
            snapshots=snaps, save_path=str(run_dir / args.gif_name), mode="voxel",
            elev=30, azim=45, cmap="terrain", fps=8, interval_ms=150, dpi=120,
            title_prefix="Crecimiento kMC SOS", every_n=1,
        )
    plt.close("all")

    save_json({
        "run_id": run_id,
        "engine": cfg.get("engine", "static"),
        "config": str(args.config),
        "size": list(size),
        "n_seeds": n_seeds,
        "rng_seed": rng_seed,
        "lattice_seed": lattice_seed,
        "init_mode": init_mode,
        "init_kwargs": init_kwargs,
        **{k: v for k, v in cfg.get("engine_kwargs", {}).items() if k != "rng_seed"},
        "fixed_sigma": getattr(params, "fixed_sigma", None),
        "times": [float(t) for t in times],
        "params": {k: (float(v) if isinstance(v, np.floating) else v)
                   for k, v in params.__dict__.items()},
        "git_commit": git_commit(find_repo_root(PROJECT_ROOT)),
        "numpy_version": np.__version__,
    }, run_dir / "metadata.json")
    return conversion_df, run_dir, csv_path


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_json(args.config)

    times = parse_times(args.times)
    ref_size = (args.size[0], args.size[1])
    sizes: List[Tuple[int, int]] = parse_sizes(args.sizes) if args.sizes else [ref_size]
    if args.snapshot_names is not None and (len(sizes) > 1 or len(args.snapshot_names) != len(times)):
        raise SystemExit("--snapshot-names requiere un solo tamaño y un nombre por tiempo.")

    n_seeds_ref = args.n_seeds if args.n_seeds is not None else int(cfg.get("n_seeds", 0))
    scaling = args.seed_scaling or cfg.get("seed_scaling", "fixed")
    rng_seed = args.rng_seed if args.rng_seed is not None else cfg.get("engine_kwargs", {}).get("rng_seed", 123)
    lattice_seed = (args.lattice_seed if args.lattice_seed is not None
                    else cfg.get("lattice", {}).get("seed", 42))
    seed_density = n_seeds_ref / (ref_size[0] * ref_size[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Config: {args.config} — {cfg.get('description', '')}")

    all_dfs, runs_info = [], []
    for run_index, size in enumerate(sizes, start=1):
        if scaling == "density":
            # n_seeds / (Lx*Ly) constante (pRun_v3.py)
            n_seeds = max(1, int(round(seed_density * size[0] * size[1])))
        else:
            n_seeds = n_seeds_ref
        run_id = f"size_{size[0]}x{size[1]}_run_{run_index:03d}_seed_{rng_seed}"
        print(f"\n=== Corrida {run_index}/{len(sizes)}: {run_id} (n_seeds={n_seeds}) ===")
        df, run_dir, csv_path = run_one(cfg, size, n_seeds, times, run_id, out_dir, args,
                                        rng_seed, lattice_seed)
        all_dfs.append(df)
        runs_info.append({"run_id": run_id, "size_x": size[0], "size_y": size[1],
                          "n_seeds": n_seeds, "seed_density_ref": seed_density,
                          "csv_path": str(csv_path), "run_dir": str(run_dir)})

    if len(sizes) > 1:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(out_dir / "all_conversion_histories.csv", index=False)
        pd.DataFrame(runs_info).to_csv(out_dir / "runs_info.csv", index=False)
        # Gráfica superpuesta de conversión (pRun_v2/v3)
        plt.figure(figsize=(7, 5))
        for run_id, group in master_df.groupby("run_id"):
            group = group.sort_values("time")
            plt.plot(group["time"], group["conversion"], label=run_id)
        plt.xlabel("Tiempo")
        plt.ylabel("Conversión")
        plt.title("Conversión vs Tiempo - comparación entre corridas")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "conversion_overlay.png", dpi=300)
        plt.close("all")
        print(f"\n✅ Resultados agregados en: {out_dir}")


if __name__ == "__main__":
    main()
