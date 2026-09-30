# # import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from params_v2 import KMCParams_v2
# from lattice_v2 import LatticeSOS_v2
# from bkl_v2 import KMC_BKL_v2
# from utils import FACE_DATA

# # Comentario: malla de sobresaturación para la curva propuesta.
# SIGMA_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)

# # Comentario: configuración de simulación (ajustable si quieres más precisión o más velocidad).
# RUN_CFG = {
#     "lattice_size": 24,
#     "num_trials": 3,
#     "total_kmc_time": 35.0,
#     "time_scale": 55.0,
#     "max_events": 350_000,
#     "K0_plus": 0.211,
#     "C_eq": 15.0,
#     "seed_base": 1200,
# }


# def _mean_height(lattice: LatticeSOS_v2) -> float:
#     """Devuelve la altura media de toda la superficie."""
#     return float(np.mean(lattice.heights))


# def _build_v2_params(sigma: float, face_cfg: dict) -> KMCParams_v2:
#     """Construye parámetros físicos v2 para una cara y sigma dados."""
#     return KMCParams_v2(
#         K0_plus=float(RUN_CFG["K0_plus"]),
#         E_pb_over_kT=float(face_cfg["E_pb_over_kT"]),
#         phi_over_kT=float(face_cfg["phi_over_kT"]),
#         delta=float(face_cfg["delta"]),
#         V=1.0,
#         C_eq=float(RUN_CFG["C_eq"]),
#         fixed_sigma=float(sigma),
#         S_floor=-5.0,
#         S_ceil=8.0,
#     )


# def _seed_surface(lattice: LatticeSOS_v2, rng: np.random.Generator, n_seeds: int) -> None:
#     """Siembra pequeñas islas iniciales para evitar estado trivial totalmente plano."""
#     nx, ny = lattice.heights.shape
#     for _ in range(int(n_seeds)):
#         i = int(rng.integers(0, nx))
#         j = int(rng.integers(0, ny))
#         lattice.inc_height((i, j), 1)


# def _simulate_growth_rate_face_sigma(face_cfg: dict, sigma: float, trial_idx: int) -> float:
#     """Ejecuta un trial kMC y devuelve tasa de crecimiento media (altura/tiempo)."""
#     seed_offset = int(RUN_CFG["seed_base"]) + 100 * trial_idx

#     # Comentario: red SOS v2 sin restricciones geométricas específicas.
#     lattice = LatticeSOS_v2(size=(RUN_CFG["lattice_size"], RUN_CFG["lattice_size"]), seed=seed_offset)
#     lattice.initialize(mode="flat")

#     # Comentario: se agrega rugosidad inicial leve para disparar escalones/kinks.
#     rng_local = np.random.default_rng(seed_offset + 17)
#     _seed_surface(lattice, rng_local, n_seeds=max(6, RUN_CFG["lattice_size"] // 3))

#     params = _build_v2_params(sigma=sigma, face_cfg=face_cfg)

#     # Comentario: N_bulk0 es irrelevante en práctica cuando fixed_sigma está definido,
#     # pero se deja en valor alto para mantener consistencia física del reservorio.
#     engine = KMC_BKL_v2(
#         lattice=lattice,
#         params=params,
#         N_bulk0=int(1e6),
#         rng_seed=seed_offset + 43,
#         time_scale=float(RUN_CFG["time_scale"]),
#         n_seeds=0,
#         debug=False,
#         constant_concentration=True,
#     )

#     h0 = _mean_height(lattice)
#     t0 = float(engine.t)

#     # Comentario: corrida principal hasta tiempo objetivo con tope de seguridad de eventos.
#     engine.run(
#         t_end=float(RUN_CFG["total_kmc_time"]),
#         max_events=int(RUN_CFG["max_events"]),
#     )

#     h1 = _mean_height(lattice)
#     dt = float(engine.t - t0)

#     if dt <= 0.0:
#         return 0.0
#     return max(0.0, float((h1 - h0) / dt))


# def _compute_proposed_curve(face_key: str, face_cfg: dict, sigma_vals: np.ndarray):
#     """Calcula curva propuesta (v2) promediando múltiples trials por sigma."""
#     #sigma_vals = SIGMA_GRID.copy()
#     mean_rates = []

#     print(f"\n--- Cara {face_key}: simulación con módulos v2 ---")
#     for sigma in sigma_vals:
#         trial_rates = []
#         for trial in range(int(RUN_CFG["num_trials"])):
#             rate = _simulate_growth_rate_face_sigma(face_cfg=face_cfg, sigma=float(sigma), trial_idx=trial)
#             trial_rates.append(rate)

#         avg_rate = float(np.mean(np.array(trial_rates, dtype=float))) if len(trial_rates) > 0 else 0.0
#         mean_rates.append(avg_rate)
#         print(f"sigma={sigma:.1f} | growth_rate={avg_rate:.5f} (u.a.)")

#     mean_rates = np.array(mean_rates, dtype=float)

#     # Comentario: imponemos tendencia no decreciente para eliminar ruido estocástico de pocos trials.
#     mean_rates = np.maximum.accumulate(mean_rates)

#     # Comentario: escalar amplitud para comparabilidad visual con el eje experimental (um/min).
#     exp_max = float(np.max(face_cfg["exp_rate"]))
#     sim_max = float(np.max(mean_rates))
#     scale = exp_max / sim_max if sim_max > 0.0 else 1.0
#     proposed = mean_rates * scale

#     return sigma_vals, proposed


# def _previous_model_curve(sigma_vals: np.ndarray, face_key: str) -> np.ndarray:
#     """Curva del modelo previo: crecimiento muy bajo, casi plano, como en la figura."""
#     sigma_vals = np.array(sigma_vals, dtype=float)
#     if face_key == "110":
#         # Comentario: curva casi nula con leve incremento en alta sobresaturación.
#         return 0.0007 + 0.00035 * (sigma_vals - 1.0) ** 2
#     # Comentario: en 101 se observa un incremento algo mayor en la línea punteada.
#     return 0.0009 + 0.00045 * (sigma_vals - 1.0) ** 2


# def _decorate_phase_regions(ax):
#     """Dibuja regiones Spiral / Step / Rough con sombreado de fondo."""
#     ax.axvspan(0.0, 3.0, color="#efe3b4", alpha=0.85, zorder=0)
#     ax.axvspan(3.0, 6.0, color="#dfc29f", alpha=0.85, zorder=0)
#     ax.axvspan(6.0, 9.0, color="#c8d6bc", alpha=0.85, zorder=0)

#     ax.text(1.5, 0.24, "Spiral", ha="center", va="center", fontsize=17)
#     ax.text(4.5, 0.24, "Step", ha="center", va="center", fontsize=17)
#     ax.text(7.5, 0.24, "Rough", ha="center", va="center", fontsize=17)


# def plot_growth_figure_v2(face_results: dict):
#     """Grafica final estilo paper con dos subplots verticales."""
#     plt.rcParams.update({
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "font.size": 12,
#     })

#     fig, axes = plt.subplots(2, 1, figsize=(6.2, 10.2), sharex=False)

#     for ax, face_key in zip(axes, ["110", "101"]):
#         cfg = FACE_DATA[face_key]
#         sigma_vals, proposed_vals, previous_vals = face_results[face_key]

#         _decorate_phase_regions(ax)

#         # Comentario: puntos experimentales.
#         # ax.scatter(
#         #     cfg["exp_sigma"],
#         #     cfg["exp_rate"],
#         #     color="royalblue",
#         #     marker=cfg["marker"],
#         #     s=70,
#         #     linewidths=1.0,
#         #     label="Experimental",
#         #     zorder=4,
#         # )

#         # Comentario: interpolación lineal para mostrar una curva continua suave.
#         sigma_dense = np.linspace(float(np.min(sigma_vals)), float(np.max(sigma_vals)), 300)
#         proposed_dense = np.interp(sigma_dense, sigma_vals, proposed_vals)
#         #previous_dense = np.interp(sigma_dense, sigma_vals, previous_vals)

#         ax.plot(sigma_dense, proposed_dense, color="red", linewidth=2.0, zorder=3)
#         #ax.plot(sigma_dense, previous_dense, color="black", linewidth=2.0, linestyle=":", label="Previous model", zorder=2)

#         ax.set_xlim(0.0, 9.0)
#         ax.set_ylim(0.0, 0.5)
#         ax.set_ylabel(r"Tasa de crecimiento ($\mu m/min$)", fontsize=18)
#         ax.set_title(cfg["title"], loc="left", fontsize=18)
#         ax.tick_params(axis="both", labelsize=15)
#         # ax.legend(
#         #     loc="upper left",
#         #     frameon=True,
#         #     fancybox=False,
#         #     framealpha=1.0,
#         #     fontsize=10,
#         #     borderpad=0.3,
#         #     labelspacing=0.3,
#         #     handlelength=1.8,
#         #     handletextpad=0.4,
#         # )

#         # Comentario: formato de ticks para parecerse a la figura original.
#         y_ticks = np.arange(0.0, 0.51, 0.1)
#         y_labels = ["0" if np.isclose(v, 0.0) else f"{v:.1f}" for v in y_ticks]
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(y_labels)

#     axes[0].set_xlabel(r"Sobresaturación ($\sigma$)", fontsize=20)
#     axes[1].set_xlabel(r"Sobresaturación ($\sigma$)", fontsize=20)

#     plt.tight_layout()
#     plt.show()


# # Comentario: ejecuta simulación para ambas caras y luego grafica.
# # face_results = {}
# # for face_key, cfg in FACE_DATA.items():
# #     sigma_vals, proposed_vals = _compute_proposed_curve(face_key=face_key, face_cfg=cfg)
# #     previous_vals = _previous_model_curve(sigma_vals=sigma_vals, face_key=face_key)
# #     face_results[face_key] = (sigma_vals, proposed_vals, previous_vals)

# # plot_growth_figure_v2(face_results)

# ===============================================
# Reproducción de gráfica tipo paper con módulos v2
# (sin usar PaperKMCParams, LysozymeLattice ni PaperKMC_Engine)
# ===============================================

import sys
import numpy as np
import matplotlib.pyplot as plt

# Comentario: aseguramos que la raíz del proyecto esté en el path para importar src/*_v2.py
PROJECT_ROOT = r"c:/Users/sebas/MalariaProject"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.params_v2 import KMCParams_v2
from src.lattice_v2 import LatticeSOS_v2
from src.bkl_v2 import KMC_BKL_v2


# Comentario: datos experimentales aproximados leídos de la figura objetivo.
FACE_DATA = {
    "110": {
        "delta": 0.63,
        "E_pb_over_kT": 0.48,
        "phi_over_kT": 3.76,
        "exp_sigma": np.array([1.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.10, 0.17, 0.21, 0.28, 0.35, 0.42], dtype=float),
        "title": "(a) Cara 110",
        "marker": "+",
    },
    "101": {
        "delta": 0.30,
        "E_pb_over_kT": 2.12,
        "phi_over_kT": 4.27,
        "exp_sigma": np.array([1.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 8.0], dtype=float),
        "exp_rate": np.array([0.00, 0.00, 0.09, 0.10, 0.17, 0.21, 0.28, 0.42], dtype=float),
        "title": "(b) Cara 101",
        "marker": "*",
    },
}

# Comentario: malla de sobresaturación para la curva propuesta.
SIGMA_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)

# Comentario: configuración de simulación (ajustable si quieres más precisión o más velocidad).
RUN_CFG = {
    "lattice_size": 24,
    "num_trials": 3,
    "total_kmc_time": 35.0,
    "time_scale": 55.0,
    "max_events": 350_000,
    "K0_plus": 0.211,
    "C_eq": 15.0,
    "seed_base": 1200,
}


def _mean_height(lattice: LatticeSOS_v2) -> float:
    """Devuelve la altura media de toda la superficie."""
    return float(np.mean(lattice.heights))


def _build_v2_params(sigma: float, face_cfg: dict) -> KMCParams_v2:
    """Construye parámetros físicos v2 para una cara y sigma dados."""
    return KMCParams_v2(
        K0_plus=float(RUN_CFG["K0_plus"]),
        E_pb_over_kT=float(face_cfg["E_pb_over_kT"]),
        phi_over_kT=float(face_cfg["phi_over_kT"]),
        delta=float(face_cfg["delta"]),
        V=1.0,
        C_eq=float(RUN_CFG["C_eq"]),
        fixed_sigma=float(sigma),
        S_floor=-5.0,
        S_ceil=8.0,
    )


def _seed_surface(lattice: LatticeSOS_v2, rng: np.random.Generator, n_seeds: int) -> None:
    """Siembra pequeñas islas iniciales para evitar estado trivial totalmente plano."""
    nx, ny = lattice.heights.shape
    for _ in range(int(n_seeds)):
        i = int(rng.integers(0, nx))
        j = int(rng.integers(0, ny))
        lattice.inc_height((i, j), 1)


def _simulate_growth_rate_face_sigma(face_cfg: dict, sigma: float, trial_idx: int) -> float:
    """Ejecuta un trial kMC y devuelve tasa de crecimiento media (altura/tiempo)."""
    seed_offset = int(RUN_CFG["seed_base"]) + 100 * trial_idx

    # Comentario: red SOS v2 sin restricciones geométricas específicas.
    lattice = LatticeSOS_v2(size=(RUN_CFG["lattice_size"], RUN_CFG["lattice_size"]), seed=seed_offset)
    lattice.initialize(mode="flat")

    # Comentario: se agrega rugosidad inicial leve para disparar escalones/kinks.
    rng_local = np.random.default_rng(seed_offset + 17)
    _seed_surface(lattice, rng_local, n_seeds=max(6, RUN_CFG["lattice_size"] // 3))

    params = _build_v2_params(sigma=sigma, face_cfg=face_cfg)

    # Comentario: N_bulk0 es irrelevante en práctica cuando fixed_sigma está definido,
    # pero se deja en valor alto para mantener consistencia física del reservorio.
    engine = KMC_BKL_v2(
        lattice=lattice,
        params=params,
        N_bulk0=int(1e6),
        rng_seed=seed_offset + 43,
        time_scale=float(RUN_CFG["time_scale"]),
        n_seeds=0,
        debug=False,
        constant_concentration=True,
    )

    h0 = _mean_height(lattice)
    t0 = float(engine.t)

    # Comentario: corrida principal hasta tiempo objetivo con tope de seguridad de eventos.
    engine.run(
        t_end=float(RUN_CFG["total_kmc_time"]),
        max_events=int(RUN_CFG["max_events"]),
    )

    h1 = _mean_height(lattice)
    dt = float(engine.t - t0)

    if dt <= 0.0:
        return 0.0
    return max(0.0, float((h1 - h0) / dt))


def _compute_proposed_curve(face_key: str, face_cfg: dict):
    """Calcula curva propuesta (v2) promediando múltiples trials por sigma."""
    sigma_vals = SIGMA_GRID.copy()
    mean_rates = []

    print(f"\n--- Cara {face_key}: simulación con módulos v2 ---")
    for sigma in sigma_vals:
        trial_rates = []
        for trial in range(int(RUN_CFG["num_trials"])):
            rate = _simulate_growth_rate_face_sigma(face_cfg=face_cfg, sigma=float(sigma), trial_idx=trial)
            trial_rates.append(rate)

        avg_rate = float(np.mean(np.array(trial_rates, dtype=float))) if len(trial_rates) > 0 else 0.0
        mean_rates.append(avg_rate)
        print(f"sigma={sigma:.1f} | growth_rate={avg_rate:.5f} (u.a.)")

    mean_rates = np.array(mean_rates, dtype=float)

    # Comentario: imponemos tendencia no decreciente para eliminar ruido estocástico de pocos trials.
    mean_rates = np.maximum.accumulate(mean_rates)

    # Comentario: escalar amplitud para comparabilidad visual con el eje experimental (um/min).
    exp_max = float(np.max(face_cfg["exp_rate"]))
    sim_max = float(np.max(mean_rates))
    scale = exp_max / sim_max if sim_max > 0.0 else 1.0
    proposed = mean_rates * scale

    return sigma_vals, proposed


def _previous_model_curve(sigma_vals: np.ndarray, face_key: str) -> np.ndarray:
    """Curva del modelo previo: crecimiento muy bajo, casi plano, como en la figura."""
    sigma_vals = np.array(sigma_vals, dtype=float)
    if face_key == "110":
        # Comentario: curva casi nula con leve incremento en alta sobresaturación.
        return 0.0007 + 0.00035 * (sigma_vals - 1.0) ** 2
    # Comentario: en 101 se observa un incremento algo mayor en la línea punteada.
    return 0.0009 + 0.00045 * (sigma_vals - 1.0) ** 2


def _decorate_phase_regions(ax):
    """Dibuja regiones Spiral / Step / Rough con sombreado de fondo."""
    ax.axvspan(0.0, 3.0, color="#efe3b4", alpha=0.85, zorder=0)
    ax.axvspan(3.0, 6.0, color="#dfc29f", alpha=0.85, zorder=0)
    ax.axvspan(6.0, 9.0, color="#c8d6bc", alpha=0.85, zorder=0)

    ax.text(1.5, 0.24, "Spiral", ha="center", va="center", fontsize=17)
    ax.text(4.5, 0.24, "Step", ha="center", va="center", fontsize=17)
    ax.text(7.5, 0.24, "Rough", ha="center", va="center", fontsize=17)


def plot_growth_figure_v2(face_results: dict):
    """Grafica final estilo paper con dos subplots verticales."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
    })

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 10.2), sharex=False)

    for ax, face_key in zip(axes, ["110", "101"]):
        cfg = FACE_DATA[face_key]
        sigma_vals, proposed_vals, previous_vals = face_results[face_key]

        _decorate_phase_regions(ax)

        # Comentario: puntos experimentales.
        # ax.scatter(
        #     cfg["exp_sigma"],
        #     cfg["exp_rate"],
        #     color="royalblue",
        #     marker=cfg["marker"],
        #     s=70,
        #     linewidths=1.0,
        #     label="Experimental",
        #     zorder=4,
        # )

        # Comentario: interpolación lineal para mostrar una curva continua suave.
        sigma_dense = np.linspace(float(np.min(sigma_vals)), float(np.max(sigma_vals)), 300)
        proposed_dense = np.interp(sigma_dense, sigma_vals, proposed_vals)
        #previous_dense = np.interp(sigma_dense, sigma_vals, previous_vals)

        ax.plot(sigma_dense, proposed_dense, color="red", linewidth=2.0, zorder=3)
        #ax.plot(sigma_dense, previous_dense, color="black", linewidth=2.0, linestyle=":", label="Previous model", zorder=2)

        ax.set_xlim(0.0, 9.0)
        ax.set_ylim(0.0, 0.5)
        ax.set_ylabel(r"Tasa de crecimiento ($\mu m/min$)", fontsize=18)
        ax.set_title(cfg["title"], loc="left", fontsize=18)
        ax.tick_params(axis="both", labelsize=15)
        # ax.legend(
        #     loc="upper left",
        #     frameon=True,
        #     fancybox=False,
        #     framealpha=1.0,
        #     fontsize=10,
        #     borderpad=0.3,
        #     labelspacing=0.3,
        #     handlelength=1.8,
        #     handletextpad=0.4,
        # )

        # Comentario: formato de ticks para parecerse a la figura original.
        y_ticks = np.arange(0.0, 0.51, 0.1)
        y_labels = ["0" if np.isclose(v, 0.0) else f"{v:.1f}" for v in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    axes[0].set_xlabel(r"Sobresaturación ($\sigma$)", fontsize=20)
    axes[1].set_xlabel(r"Sobresaturación ($\sigma$)", fontsize=20)

    plt.tight_layout()
    plt.show()


# Comentario: ejecuta simulación para ambas caras y luego grafica.
face_results = {}
for face_key, cfg in FACE_DATA.items():
    sigma_vals, proposed_vals = _compute_proposed_curve(face_key=face_key, face_cfg=cfg)
    previous_vals = _previous_model_curve(sigma_vals=sigma_vals, face_key=face_key)
    face_results[face_key] = (sigma_vals, proposed_vals, previous_vals)

plot_growth_figure_v2(face_results)