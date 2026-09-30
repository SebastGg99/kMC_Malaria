"""Casos "golden" compartidos por `make_golden.py` (código original) y `test_golden.py`
(código refactorizado).

Cada caso construye una simulación pequeña (red 12x12, pocos miles de eventos) con
semillas fijas y la ejecuta. El resultado se serializa en un diccionario de arrays de
numpy que se compara **bit a bit** (igualdad exacta, sin tolerancia).

Las clases se inyectan mediante un diccionario `api`, de modo que el mismo caso se
pueda ejecutar con las clases originales (`bkl`, `bkl_v4`, ...) y con las
refactorizadas (`src.dynamic`, `src.static`) sin duplicar la definición.

Claves esperadas en `api`:
    dyn_params, dyn_lattice, dyn_engine, dyn_selective, dyn_nodesnomig,
    sta_params, sta_lattice, sta_engine, sta_selective, sta_nodesnomig
"""

from typing import Callable, Dict, List

import numpy as np

# Códigos numéricos para serializar el tipo de evento del historial.
EVENT_CODES = {"adsorption": 0, "desorption": 1, "migration": 2, "incorporation": 3}

# Tamaño de red y número máximo de eventos comunes a todos los casos.
SIZE = (12, 12)
MAX_EVENTS = 1500

# Instantes de snapshot: se eligen pequeños; los no alcanzados se rellenan al final
# con el estado final (comportamiento propio de run()).
SNAP_TIMES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Parámetros de referencia
# ---------------------------------------------------------------------------
def _dyn_params(api, **overrides):
    """Parámetros dinámicos tomados de final_results.ipynb (celda 38)."""
    base = dict(
        T=300,
        K0_plus=11.671806264584635,
        K_inc_plus=0.5069325183371898,
        E_pb_over_kT=1.2704636027368605,
        phi_over_kT=1.4792478012079329,
        delta=1.7789686274068774,
        V=0.9943756704678864,
        C_eq=6.66285823642452,
        S_floor=-5.0,
        S_ceil=8.0,
    )
    base.update(overrides)
    return api["dyn_params"](**base)


def _sta_params(api, iso: bool = True, **overrides):
    """Parámetros estáticos tomados de isotropicRun.py / anisotropicRun.py."""
    if iso:
        base = dict(
            K0_plus=1.16718,
            K_inc_plus=0.5069325183371898,
            E_pb_over_kT_x=1.2704636027368605,
            E_pb_over_kT_y=1.2704636027368605,
            phi_over_kT=1.4792478012079329,
            delta_x=1.7789686274068774,
            delta_y=1.7789686274068774,
            V=1,
            C_eq=15.0,
            fixed_sigma=1.0,
            S_floor=-5.0,
            S_ceil=9.0,
        )
    else:
        base = dict(
            K0_plus=0.516718,
            K_inc_plus=0.5069325183371898,
            E_pb_over_kT_x=0.76085943,
            E_pb_over_kT_y=2.19936039,
            phi_over_kT=1.4792478012079329,
            delta_x=0.3,
            delta_y=1.7789686274068774,
            V=1,
            C_eq=15.0,
            fixed_sigma=0.5,
            S_floor=-5.0,
            S_ceil=9.0,
        )
    base.update(overrides)
    return api["sta_params"](**base)


# ---------------------------------------------------------------------------
# Constructores de cada caso: devuelven el objeto kmc listo para run()
# ---------------------------------------------------------------------------
def _dyn_flat(api, engine_key="dyn_engine", engine_kwargs=None, params_overrides=None,
              fixed_sigma=None, init_mode="flat"):
    """Caso genérico de la línea dinámica."""
    lat = api["dyn_lattice"](size=SIZE, seed=25)
    if init_mode == "random_surface":
        lat.initialize("random_surface", max_roughness=2)
    else:
        lat.initialize("flat")
    params = _dyn_params(api, **(params_overrides or {}))
    if fixed_sigma is not None:
        # KMCParams no declara fixed_sigma; el motor lo lee con getattr().
        params.fixed_sigma = fixed_sigma
    kwargs = dict(N_bulk0=300, rng_seed=123, time_scale=1.0, n_seeds=5)
    kwargs.update(engine_kwargs or {})
    return api[engine_key](lat, params, **kwargs)


def _sta_generic(api, engine_key="sta_engine", iso=True, init=None, engine_kwargs=None,
                 params_overrides=None):
    """Caso genérico de la línea estática."""
    lat = api["sta_lattice"](size=SIZE, seed=42)
    init = init or dict(mode="flat", max_height=1, n_seeds=10)
    lat.initialize(**init)
    params = _sta_params(api, iso=iso, **(params_overrides or {}))
    kwargs = dict(
        lattice=lat,
        params=params,
        N_bulk0=2000,
        rng_seed=123,
        time_scale=1.0,
        n_seeds=10,
        constant_concentration=True,
    )
    kwargs.update(engine_kwargs or {})
    return api[engine_key](**kwargs)


CASES: Dict[str, Callable] = {
    # ---- Línea dinámica ----
    "dyn_depletion": lambda api: _dyn_flat(api),
    "dyn_fixed_sigma": lambda api: _dyn_flat(api, fixed_sigma=2.0),
    "dyn_random_surface": lambda api: _dyn_flat(api, init_mode="random_surface"),
    "dyn_selective_no_des": lambda api: _dyn_flat(
        api, "dyn_selective", engine_kwargs=dict(enable_desorption=False)),
    "dyn_selective_no_mig": lambda api: _dyn_flat(
        api, "dyn_selective", engine_kwargs=dict(enable_migration=False)),
    "dyn_nodesnomig_default": lambda api: _dyn_flat(api, "dyn_nodesnomig"),
    "dyn_nodesnomig_mig_on": lambda api: _dyn_flat(
        api, "dyn_nodesnomig", engine_kwargs=dict(enable_migration=True)),
    "dyn_nodesnomig_all_on": lambda api: _dyn_flat(
        api, "dyn_nodesnomig", engine_kwargs=dict(enable_desorption=True, enable_migration=True)),
    # ---- Línea estática ----
    "sta_iso_fixed_sigma": lambda api: _sta_generic(api),
    "sta_aniso_fixed_sigma": lambda api: _sta_generic(api, iso=False),
    "sta_constant_conc": lambda api: _sta_generic(
        api, params_overrides=dict(fixed_sigma=None), engine_kwargs=dict(N_bulk0=30)),
    # Modo dinámico dentro del motor estático (D5: "no validado", pero se congela).
    "sta_dynamic_mode": lambda api: _sta_generic(
        api, params_overrides=dict(fixed_sigma=None),
        engine_kwargs=dict(N_bulk0=40, constant_concentration=False)),
    "sta_init_random": lambda api: _sta_generic(
        api, init=dict(mode="random", max_height=1, n_seeds=10)),
    "sta_init_seeds": lambda api: _sta_generic(
        api, init=dict(mode="seeds", n_seeds=30)),
    "sta_init_screw": lambda api: _sta_generic(
        api, init=dict(mode="screw", screw_pitch=2, screw_noise=1)),
    "sta_solvent": lambda api: _sta_generic(
        api, params_overrides=dict(solvent_ads_factor=0.7, solvent_des_factor=1.3,
                                   solvent_mig_factor=0.5, solvent_inc_factor=2.0)),
    "sta_solvent_off": lambda api: _sta_generic(
        api, params_overrides=dict(solvent_ads_factor=0.7), engine_kwargs=dict(use_solvent=False)),
    "sta_selective_no_mig": lambda api: _sta_generic(
        api, "sta_selective", engine_kwargs=dict(enable_migration=False)),
    "sta_nodesnomig": lambda api: _sta_generic(api, "sta_nodesnomig"),
}


# ---------------------------------------------------------------------------
# Ejecución y serialización
# ---------------------------------------------------------------------------
def _serialize_history(history: List) -> Dict[str, np.ndarray]:
    """Convierte la lista (t, tipo, sitio) en tres arrays."""
    times = np.array([h[0] for h in history], dtype=float)
    codes = np.array([EVENT_CODES[h[1]] for h in history], dtype=np.int8)
    sites = np.array([h[2] if h[2] is not None else (-1, -1) for h in history], dtype=np.int64)
    return {"hist_t": times, "hist_code": codes, "hist_site": sites.reshape(-1, 2)}


def run_case(name: str, api: Dict) -> Dict[str, np.ndarray]:
    """Ejecuta un caso y devuelve todos sus observables como arrays."""
    kmc = CASES[name](api)
    out = kmc.run(t_end=1e9, snapshot_times=SNAP_TIMES, max_events=MAX_EVENTS)

    # La línea dinámica devuelve solo snaps; la estática (snaps, stats).
    if isinstance(out, tuple):
        snaps, stats = out
    else:
        snaps, stats = out, None

    res: Dict[str, np.ndarray] = {
        "heights": np.asarray(kmc.lat.heights).copy(),
        "t": np.array(kmc.t),
        "N_bulk": np.array(kmc.N_bulk),
        "N_inc": np.array(kmc.N_inc),
        "counts": np.array([kmc.counts[k] for k in EVENT_CODES]),
        "conversion_percent": np.array(kmc.conversion_percent),
        "snap_t": np.array([s[0] for s in snaps], dtype=float),
        "snap_h": np.stack([s[1] for s in snaps]),
        "snap_conv": np.array([s[2] for s in snaps], dtype=float),
    }
    res.update(_serialize_history(kmc.history))

    if stats is not None:
        res["height_history"] = np.array(stats["height_history"], dtype=float)
        res["conversion_history"] = np.array(stats["conversion_history"], dtype=float)
        probs = stats["adsorption_probs_history"]
        res["ads_probs"] = np.array(
            [(p[0], p[1], p[2][0], p[2][1], p[2][2]) for p in probs], dtype=float
        ).reshape(-1, 5)
        res["total_time"] = np.array(stats["total_time"])
        res["total_events"] = np.array(stats["total_events"])
    return res
