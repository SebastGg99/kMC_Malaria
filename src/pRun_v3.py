import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Ajusta la ruta local si es necesario
# sys.path.append(r'c:/Users/sebas/MalariaProject/')
import sys
sys.path.append('/home/sgaviria/MalariaProject/')

from src import *

# Configuración de estilo para plots
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['font.size'] = 11


def _print_usage() -> None:
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
        "  --sizes <lista de tamaños>\n"
        "      Ej: --sizes 10x10 20x20 30x30\n"
        "      Ej: --sizes 10x10,20x20,30x30\n"
        "  --n-seeds <int>\n"
        "      Ej: --n-seeds 10000\n"
        "      Nota: este valor se usa como referencia para definir la densidad\n"
        "            constante de semillas entre diferentes tamaños de red.\n"
        "  --snapshot-name <archivo.png>\n"
        "      Ej: --snapshot-name crystal.png\n"
        "  --gif [true|false]\n"
        "      Ej: --gif true\n"
        "  --gif-name <archivo.gif>\n"
        "      Ej: --gif-name growth.gif\n"
        "  --output-dir <directorio>\n"
        "      Ej: --output-dir outputs\n"
    )


def _parse_bool(value: str) -> bool:
    """Convierte texto a booleano de forma robusta."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Valor booleano inválido: {value}")


def _parse_times(values: List[str]) -> np.ndarray:
    """
    Interpreta los tiempos de snapshot.
    Formatos soportados:
      --times 0 1 2 3 4
      --times 0,1,2,3,4
      --times 0:8:1
    """
    if len(values) == 1 and ":" in values[0]:
        parts = values[0].split(":")
        if len(parts) != 3:
            raise ValueError("Formato de rango inválido para --times")
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


def _parse_sizes(values: List[str]) -> List[Tuple[int, int]]:
    """
    Acepta formatos:
      --sizes 10x10 20x20 30x30
      --sizes 10x10,20x20,30x30
    """
    sizes: List[Tuple[int, int]] = []

    for value in values:
        chunks = [c.strip() for c in value.split(",") if c.strip()]
        for chunk in chunks:
            m = re.fullmatch(r"(\d+)x(\d+)", chunk)
            if not m:
                raise ValueError(
                    f"Formato inválido de tamaño: {chunk}. "
                    "Usa nxm, por ejemplo 20x20"
                )
            nx, ny = int(m.group(1)), int(m.group(2))
            sizes.append((nx, ny))

    if not sizes:
        raise ValueError("--sizes requiere al menos un tamaño")

    return sizes


def _parse_args(argv: List[str]) -> Dict[str, object]:
    """
    Parsea argumentos de línea de comandos sin usar argparse,
    para mantener el estilo del script original.
    """
    config: Dict[str, object] = {
        "times": np.arange(0, 8, 1),
        "size": (10, 10),
        "sizes": None,
        "n_seeds": 10000,
        "snapshot_name": "crystal_growth_prueba_run.png",
        "gif": False,
        "gif_name": "crystal_growth_prueba_run.gif",
        "output_dir": "outputs",
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

        if arg == "--sizes":
            i += 1
            values = []
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            if not values:
                raise ValueError("--sizes requiere valores")
            config["sizes"] = _parse_sizes(values)
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

        if arg == "--output-dir":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("--output-dir requiere un valor")
            config["output_dir"] = argv[i + 1]
            i += 2
            continue

        raise ValueError(f"Argumento desconocido: {arg}")

    return config


def _make_run_id(size: Tuple[int, int], run_index: int, rng_seed: int) -> str:
    """Crea un identificador único para cada corrida."""
    return f"size_{size[0]}x{size[1]}_run_{run_index:03d}_seed_{rng_seed}"


def _save_conversion_overlay(master_df: pd.DataFrame, save_path: Path) -> None:
    """
    Genera una gráfica superpuesta de conversión vs tiempo
    para comparar todas las corridas.
    """
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
    plt.savefig(save_path, dpi=300)
    plt.show()


def run_single_simulation(
    size: Tuple[int, int],
    n_seeds: int,
    times: np.ndarray,
    params,
    output_dir: Path,
    run_id: str,
    rng_seed: int = 123,
    time_scale: int = 80,
    N_bulk0: int = 2000,
    constant_concentration: bool = True,
    save_snapshots: bool = True,
    save_gif: bool = False,
    snapshot_name: str = "snapshot.png",
    gif_name: str = "growth.gif",
):
    """
    Ejecuta una simulación, guarda resultados locales de la corrida
    y retorna datos útiles para agregación posterior.
    """
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Se construye la red para el tamaño actual
    L = LatticeSOS_v4(size=size, seed=42)
    L.initialize(mode="flat", max_height=1, n_seeds=n_seeds)

    # Se instancia el motor KMC con los parámetros definidos
    kmc = KMC_BKL_v4(
        lattice=L,
        params=params,
        N_bulk0=N_bulk0,
        rng_seed=rng_seed,
        time_scale=time_scale,
        n_seeds=n_seeds,
        constant_concentration=constant_concentration,
    )

    # Se ejecuta la simulación hasta el tiempo final pedido
    snaps, stats = kmc.run(t_end=times[-1], snapshot_times=times)

    # Guardar serie de conversión en dataframe
    conversion_history = stats["conversion_history"]
    # Se asume formato [(t, conv), ...]
    conversion_df = pd.DataFrame(conversion_history, columns=["time", "conversion"])
    conversion_df["size_x"] = size[0]
    conversion_df["size_y"] = size[1]
    conversion_df["n_seeds"] = n_seeds
    conversion_df["run_id"] = run_id
    conversion_df["rng_seed"] = rng_seed

    csv_path = run_dir / "conversion_history.csv"
    conversion_df.to_csv(csv_path, index=False)

    # Gráfica individual de conversión
    conversion_series = [(t, None, conv) for t, conv in conversion_history]
    plotter = Plotter(kmc)
    indiv_plot_path = run_dir / f"conversion_{run_id}.png"
    plotter.plot_conversion(
        conversion_series,
        title=f"Conversión vs Tiempo - {run_id}",
        save_path=str(indiv_plot_path)
    )

    # Guardar snapshots / gif con nombres únicos
    plotter_v2 = Plotter_v2(kmc)

    if save_snapshots:
        snapshot_path = run_dir / snapshot_name
        plotter_v2.plot_crystal_3d(
            mode="voxel",
            snapshots=snaps,
            t_snapshot=times[len(times) - 1],
            save_path=str(snapshot_path),
        )

    if save_gif:
        gif_path = run_dir / gif_name
        plotter_v2.crystal_growth_gif(
            snapshots=snaps,
            save_path=str(gif_path),
            mode="voxel",
            elev=30,
            azim=45,
            cmap="terrain",
            fps=8,
            interval_ms=150,
            dpi=120,
            title_prefix="Crecimiento kMC SOS",
            every_n=1,
        )

    return {
        "kmc": kmc,
        "snaps": snaps,
        "stats": stats,
        "conversion_df": conversion_df,
        "run_dir": run_dir,
        "csv_path": csv_path,
    }


print("✅ Módulos importados correctamente")

try:
    cli = _parse_args(sys.argv[1:])
except ValueError as exc:
    print(f"Error: {exc}")
    _print_usage()
    sys.exit(2)

# Parámetros de entrada
times = cli["times"]
size = cli["size"]
sizes = cli["sizes"]
n_seeds_ref = cli["n_seeds"]
snapshot_name = cli["snapshot_name"]
gif = cli["gif"]
gif_name = cli["gif_name"]
output_dir = Path(cli["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# DENSIDAD DE SEMILLAS DE REFERENCIA
# -------------------------------------------------------------------------
# Esta densidad se toma usando el tamaño de referencia dado por --size
# y el valor de --n-seeds. Luego, para cada tamaño en --sizes, se recalcula
# el número de semillas para conservar esta misma densidad.
seed_density = n_seeds_ref / (size[0] * size[1])

print(f"✅ Densidad de semillas de referencia: {seed_density:.8f} semillas/celda")

# Parámetros del modelo
params = KMCParams_v4(
    K0_plus=0.2116718,
    K_inc_plus=0.5069325183371898,
    E_pb_over_kT_x=2.2704636027368605,
    E_pb_over_kT_y=2.2704636027368605,
    phi_over_kT=1.3792478012079329,
    delta_x=1.3,
    delta_y=1.3,
    V=1,
    C_eq=15.0,
    fixed_sigma=1,
    S_floor=-5.0,
    S_ceil=9.0,
)

# Si no se pasan varios tamaños, se corre uno solo.
if sizes is None:
    sizes = [size]

all_conversion_dfs = []
all_runs_info = []

for run_index, current_size in enumerate(sizes, start=1):
    # ---------------------------------------------------------------------
    # Recalcular n_seeds para este tamaño manteniendo densidad constante
    # n_seeds / (Lx * Ly) = constante
    # ---------------------------------------------------------------------
    current_n_seeds = max(
        1,
        int(round(seed_density * current_size[0] * current_size[1]))
    )

    run_id = _make_run_id(current_size, run_index, rng_seed=123)

    print(
        f"\n=== Ejecutando corrida {run_index}/{len(sizes)}: {run_id} "
        f"(n_seeds={current_n_seeds}) ==="
    )

    result = run_single_simulation(
        size=current_size,
        n_seeds=current_n_seeds,
        times=times,
        params=params,
        output_dir=output_dir,
        run_id=run_id,
        rng_seed=123,
        time_scale=100,
        N_bulk0=2000,
        constant_concentration=True,
        save_snapshots=True,
        save_gif=gif,
        snapshot_name=snapshot_name,
        gif_name=gif_name,
    )

    all_conversion_dfs.append(result["conversion_df"])
    all_runs_info.append({
        "run_id": run_id,
        "size_x": current_size[0],
        "size_y": current_size[1],
        "n_seeds": current_n_seeds,
        "seed_density_ref": seed_density,
        "csv_path": str(result["csv_path"]),
        "run_dir": str(result["run_dir"]),
    })

# Guardar CSV maestro con todas las corridas
master_df = pd.concat(all_conversion_dfs, ignore_index=True)
master_csv_path = output_dir / "all_conversion_histories_1.csv"
master_df.to_csv(master_csv_path, index=False)

# Guardar resumen de corridas
runs_info_df = pd.DataFrame(all_runs_info)
runs_info_csv_path = output_dir / "runs_info_1.csv"
runs_info_df.to_csv(runs_info_csv_path, index=False)

print(f"\n✅ CSV maestro guardado en: {master_csv_path}")
print(f"✅ Resumen de corridas guardado en: {runs_info_csv_path}")

# Gráfica superpuesta final
overlay_plot_path = output_dir / "conversion_overlay_1.png"
_save_conversion_overlay(master_df, overlay_plot_path)

print(f"✅ Gráfica superpuesta guardada en: {overlay_plot_path}")
print("✅ Todas las corridas terminaron correctamente.")