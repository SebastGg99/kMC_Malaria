import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Añadir ruta local si es necesario (ajustar según tu estructura)
#sys.path.append(r'c:/Users/sebas/MalariaProject/')

sys.path.append('/home/sgaviria/MalariaProject/')

from src import *

# Configuración de estilo para plots
#plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['font.size'] = 11

def _print_usage() -> None:
    # Simple CLI help to keep the script self-contained.
    print(
        "Uso: python pRun.py [opciones]\n"
        "\n"
        "Opciones:\n"
        "  --times <lista|inicio:fin:paso>\n"
        "      Ej: --times 0 1 2 3 4\n"
        "      Ej: --times 0,1,2,3,4\n"
        "      Ej: --times 0:8:1\n"
        "  --size <nx> <ny>\n"
        "      Ej: --size 10 10\n"
        "  --n-seeds <int>\n"
        "      Ej: --n-seeds 10000\n"
        "  --snapshot-name <archivo.png>\n"
        "      Ej: --snapshot-name crystal.png\n"
        "  --snapshot-names <lista_de_archivos.png>\n"
        "      Ej: --snapshot-names s0.png s1.png s2.png\n"
        "  --gif [true|false]\n"
        "      Ej: --gif true\n"
        "  --gif-name <archivo.gif>\n"
        "      Ej: --gif-name growth.gif\n"
    )


def _parse_bool(value: str) -> bool:
    # Accept common true/false representations.
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Valor booleano invalido: {value}")


def _parse_times(values: List[str]) -> np.ndarray:
    # Support list values and range format start:end:step.
    if len(values) == 1 and ":" in values[0]:
        parts = values[0].split(":")
        if len(parts) != 3:
            raise ValueError("Formato de rango invalido para --times")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2])
        if step <= 0:
            raise ValueError("El paso de --times debe ser > 0")
        return np.arange(start, end, step)

    times: List[float] = []
    for value in values:
        for chunk in value.split(","):
            if chunk.strip() == "":
                continue
            times.append(float(chunk))
    if not times:
        raise ValueError("--times requiere al menos un valor")
    return np.array(times, dtype=float)


def _parse_args(argv: List[str]) -> Dict[str, object]:
    # Minimal parser to avoid extra dependencies.
    config: Dict[str, object] = {
        "times": np.arange(0, 8, 1),
        "size": (10, 10),
        "n_seeds": 10000,
        "snapshot_name": "crystal_growth_prueba_run.png",
        "snapshot_names": None,
        "gif": False,
        "gif_name": "crystal_growth_prueba_run.gif",
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            _print_usage()
            sys.exit(0)

        if arg == "--times":
            i += 1
            values: List[str] = []
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            if not values:
                raise ValueError("--times requiere valores")
            config["times"] = _parse_times(values)
            continue

        if arg == "--size":
            if i + 2 >= len(argv):
                raise ValueError("--size requiere dos valores")
            if argv[i + 1].startswith("--") or argv[i + 2].startswith("--"):
                raise ValueError("--size requiere dos valores")
            config["size"] = (int(argv[i + 1]), int(argv[i + 2]))
            i += 3
            continue

        if arg == "--n-seeds":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("--n-seeds requiere un valor")
            n_seeds_value = int(argv[i + 1])
            if n_seeds_value <= 0:
                raise ValueError("--n-seeds debe ser > 0")
            config["n_seeds"] = n_seeds_value
            i += 2
            continue

        if arg == "--snapshot-name":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("--snapshot-name requiere un valor")
            config["snapshot_name"] = argv[i + 1]
            i += 2
            continue

        if arg == "--snapshot-names":
            i += 1
            values = []
            while i < len(argv) and not argv[i].startswith("--"):
                values.extend([chunk for chunk in argv[i].split(",") if chunk.strip()])
                i += 1
            if not values:
                raise ValueError("--snapshot-names requiere valores")
            config["snapshot_names"] = values
            continue

        if arg == "--gif":
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                config["gif"] = _parse_bool(argv[i + 1])
                i += 2
            else:
                config["gif"] = True
                i += 1
            continue

        if arg == "--gif-name":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("--gif-name requiere un valor")
            config["gif_name"] = argv[i + 1]
            i += 2
            continue

        raise ValueError(f"Argumento desconocido: {arg}")

    snapshot_names = config["snapshot_names"]
    if snapshot_names is not None:
        times = config["times"]
        if len(snapshot_names) != len(times):
            raise ValueError("--snapshot-names debe tener la misma longitud que --times")

    return config

print("✅ Módulos importados correctamente")

try:
    cli = _parse_args(sys.argv[1:])
except ValueError as exc:
    print(f"Error: {exc}")
    _print_usage()
    sys.exit(2)

times = cli["times"]
size = cli["size"]
n_seeds = cli["n_seeds"]
snapshot_name = cli["snapshot_name"]
snapshot_names = cli["snapshot_names"]
gif = cli["gif"]
gif_name = cli["gif_name"]

L_ani = LatticeSOS_v4(size=size, seed=42)
# L.initialize(mode="random", max_height=1, n_seeds=1905)
L_ani.initialize(mode="flat", max_height=1, n_seeds=10)

params_ani = KMCParams_v4(
K0_plus= 0.2116718, #0.211,#
K_inc_plus=0.5069325183371898,
E_pb_over_kT_x=1.2704636027368605,
E_pb_over_kT_y=2.2704636027368605,
phi_over_kT= 1.3792478012079329, #3.76, #1.4792478012079329, #
delta_x=0.3,
delta_y=1.,
V=1,
C_eq=15.0,
fixed_sigma=1,   # usa un valor como 3.0 si quieres sigma estático
S_floor=-5.0,
S_ceil=9.0,
)

kmc_ani = KMC_BKL_v4(
    lattice=L_ani,
    params=params_ani,
    N_bulk0=2000,
    rng_seed=123,
    time_scale=10,
    n_seeds=10,
    constant_concentration=True,  # batch dinámico
)

# params = KMCParams_v2(
# K0_plus= 0.2116718, #0.211,#
# E_pb_over_kT=  2.2704636027368605,#.48,# #1.2704636027368605,
# phi_over_kT= 1.3792478012079329, #3.76, #1.4792478012079329, #
# delta=1.3,
# V=1,
# C_eq=15.0,
# fixed_sigma=1,   # usa un valor como 3.0 si quieres sigma estático
# S_floor=-5.0,
# S_ceil=9.0,
# )



# kmc = KMC_BKL_v2(
#     lattice=L,
#     params=params,
#     N_bulk0=2000,
#     rng_seed=123,
#     time_scale=50,
#     n_seeds=n_seeds,
#     constant_concentration=True,  # batch dinámico
# )


snaps_ani, stats_ani = kmc_ani.run(t_end=times[-1], snapshot_times=times)

plotter = Plotter_v2(kmc_ani)
if snapshot_names is not None:
    # Save one image per snapshot time when names are provided.
    for t_snapshot, name in zip(times, snapshot_names):
        plotter.plot_crystal_3d(
            mode="voxel",
            snapshots=snaps_ani,
            t_snapshot=t_snapshot,
            save_path=name,
        )
else:
    # Default behavior: save only the last snapshot.
    plotter.plot_crystal_3d(
        mode="voxel",
        snapshots=snaps_ani,
        t_snapshot=times[len(times)-1],
        save_path=snapshot_name,
    )

if gif:

    plotter.crystal_growth_gif(
        snapshots=snaps_ani,
        save_path=gif_name,
        mode="voxel",        # o "voxel"
        elev=30,
        azim=45,
        cmap="terrain",
        fps=8,
        interval_ms=150,
        dpi=120,
        title_prefix="Crecimiento kMC SOS - anisotrópico",
        every_n=1,
    )

# Armar serie compatible con Plotter (t, heights, conversion)
conversion_series = [(t, None, conv) for t, conv in stats_ani["conversion_history"]]

# Graficar conversión vs tiempo
plots = Plotter(kmc_ani)
plots.plot_conversion(conversion_series, title="Conversión vs Tiempo - modelo anisotrópico 3D",
                      save_path="conversion_ani_50.png")