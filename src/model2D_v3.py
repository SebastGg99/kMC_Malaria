"""Modelo KMC 2D con anisotropia de forma y efectos de solvente.

Descripcion
----------
Este modulo replica la logica del modelo 2D con anisotropia (v2) y agrega
factores de solvente que escalan las tasas de cada evento. Un factor 1.0
mantiene la dinamica original sin cambios.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from utils_v2 import (
    EMPTY,
    HEMIN,
    STEP,
    VECINOS,
    in_bounds,
    count_bonds,
    count_bonds_xy,
    count_bonds_all,
    is_step_site,
    arrhenius,
)


# ----------------- Control de parametros -----------------
# Considerando step/kink
@dataclass
class Params2D:
    """Parametros del modelo 2D (v3) con anisotropia y solvente.

    Los factores de solvente multiplican las tasas:
    - 1.0: sin efecto
    - <1: ralentiza
    - >1: acelera
    """

    # Barreras base (eV)
    Ead: float = 0.05
    Ede: float = 0.003
    Eai: float = 0.003
    Eab: float = 0.003
    Eac: float = 0.003

    # Correcciones por vecino (anisotropia de forma)
    Jad_x: float = 0.04
    Jad_y: float = 0.04
    Jinc_x: float = 0.01
    Jinc_y: float = 0.01
    Jdes_x: float = 0.05
    Jdes_y: float = 0.05
    Jmove_x: float = 0.03
    Jmove_y: float = 0.03

    # Correcciones isotropicas opcionales (compatibilidad con v1)
    Jad: Optional[float] = None
    Jinc: Optional[float] = None
    Jdes: Optional[float] = None
    Jmove: Optional[float] = None

    # Potencial quimico efectivo
    mu: float = -0.1

    # Factores de solvente (adimensionales)
    solvent_ads_factor: float = 1.0
    solvent_des_factor: float = 1.0
    solvent_diff_factor: float = 1.0
    solvent_inc_factor: float = 1.0

    # Parametros fijos del sistema
    Lx: int = field(init=True, default=120)
    Ly: int = field(init=True, default=120)
    T: float = field(init=False, default=298.15)
    kB: float = field(init=False, default=8.617e-5)
    nu0: float = field(init=False, default=1e7)

    # Alias internos
    Ea_ads: float = field(init=False)
    Ea_des: float = field(init=False)
    Ea_inc: float = field(init=False)
    Ea_diff_b: float = field(init=False)
    Ea_diff_c: float = field(init=False)

    eps_ads_x: float = field(init=False)
    eps_ads_y: float = field(init=False)
    eps_inc_x: float = field(init=False)
    eps_inc_y: float = field(init=False)
    eps_des_x: float = field(init=False)
    eps_des_y: float = field(init=False)
    eps_move_x: float = field(init=False)
    eps_move_y: float = field(init=False)

    eps_ads: float = field(init=False)
    eps_inc: float = field(init=False)
    eps_des: float = field(init=False)
    eps_move: float = field(init=False)

    mu_hemin: float = field(init=False)
    activity_hemin: float = field(init=False, default=1.0)

    face_bias: float = field(init=False, default=1.0)
    enable_ads: bool = field(init=False, default=True)
    enable_des: bool = field(init=False, default=True)
    enable_diff: bool = field(init=False, default=True)
    enable_incorp: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        """Mapea entradas a nombres internos usados por KMC2D."""
        # Barreras base
        self.Ea_ads = self.Ead
        self.Ea_des = self.Ede
        self.Ea_inc = self.Eai
        self.Ea_diff_b = self.Eab
        self.Ea_diff_c = self.Eac

        # Correcciones direccionales
        self.eps_ads_x = self.Jad_x
        self.eps_ads_y = self.Jad_y
        self.eps_inc_x = self.Jinc_x
        self.eps_inc_y = self.Jinc_y
        self.eps_des_x = self.Jdes_x
        self.eps_des_y = self.Jdes_y
        self.eps_move_x = self.Jmove_x
        self.eps_move_y = self.Jmove_y

        # Correcciones isotropicas (promedio si no se especifican)
        if self.Jad is None:
            self.Jad = 0.5 * (self.Jad_x + self.Jad_y)
        if self.Jinc is None:
            self.Jinc = 0.5 * (self.Jinc_x + self.Jinc_y)
        if self.Jdes is None:
            self.Jdes = 0.5 * (self.Jdes_x + self.Jdes_y)
        if self.Jmove is None:
            self.Jmove = 0.5 * (self.Jmove_x + self.Jmove_y)

        self.eps_ads = float(self.Jad)
        self.eps_inc = float(self.Jinc)
        self.eps_des = float(self.Jdes)
        self.eps_move = float(self.Jmove)

        # Potencial quimico
        self.mu_hemin = self.mu


# ----------------- Nucleo KMC -----------------
class KMC2D:
    """Motor KMC 2D con anisotropia de forma y factores de solvente."""

    def __init__(
        self,
        p: Params2D,
        seed: int = 0,
        use_anisotropy: Optional[bool] = None,
        use_solvent: Optional[bool] = None,
        n_seeds: Optional[int] = None,
    ) -> None:
        """Inicializa la simulacion con semilla central o semillas aleatorias."""
        self.p = p
        self.grid = np.zeros((p.Lx, p.Ly), dtype=np.int8)
        self.rng = random.Random(seed)
        self.t = 0.0
        # Sembrado inicial: central por defecto, o aleatorio si se especifica n_seeds.
        if n_seeds is None:
            # Semilla: un adatom en el centro
            self.grid[p.Lx // 2, p.Ly // 2] = HEMIN
        else:
            n_pick = int(max(0, n_seeds))
            total_sites = p.Lx * p.Ly
            n_pick = min(n_pick, total_sites)
            if n_pick > 0:
                flat_indices = self.rng.sample(range(total_sites), k=n_pick)
                for flat_idx in flat_indices:
                    i = flat_idx // p.Ly
                    j = flat_idx % p.Ly
                    self.grid[i, j] = HEMIN

        # Flags de configuracion del motor
        self.use_anisotropy = True if use_anisotropy is None else bool(use_anisotropy)
        self.use_solvent = True if use_solvent is None else bool(use_solvent)

    def _solvent_factor(self, name: str) -> float:
        # Factor neutro si el solvente esta desactivado
        if not self.use_solvent:
            return 1.0
        return float(getattr(self.p, name, 1.0))

    def _bond_term(
        self,
        bonds: int,
        bx: int,
        by: int,
        eps_x: float,
        eps_y: float,
        eps_iso: float,
    ) -> float:
        # Termino de vecinos unificado para isotropia/anisotropia
        if self.use_anisotropy:
            return bx * eps_x + by * eps_y
        return bonds * eps_iso

    def local_rates(self, i: int, j: int) -> dict:
        """Calcula tasas locales de eventos en el sitio (i, j)."""
        g, p = self.grid, self.p
        rates = {}

        # Enlaces (isotropicos o direccionales)
        if self.use_anisotropy:
            bx, by = count_bonds_xy(g, i, j)
            bonds = bx + by
        else:
            bonds = count_bonds(g, i, j)
            bx, by = 0, 0

        # ------------------------------
        # Caso 1: sitio vacio (EMPTY)
        # ------------------------------
        if g[i, j] == EMPTY:
            if p.enable_ads:
                # Adsorcion: menos favorable con mas vecinos
                Ea = p.Ea_ads + self._bond_term(
                    bonds, bx, by, p.eps_ads_x, p.eps_ads_y, p.eps_ads
                )
                k_ads = arrhenius(p.nu0, Ea - p.mu_hemin, p.T, p.kB) * p.activity_hemin
                # Ajuste de solvente en la tasa de adsorcion
                k_ads *= self._solvent_factor("solvent_ads_factor")
                if bonds > 0:
                    rates["ads"] = k_ads

            if p.enable_incorp and is_step_site(g, i, j):
                # Incorporacion: mas favorable con mas vecinos
                Ea_inc = p.Ea_inc - self._bond_term(
                    bonds, bx, by, p.eps_inc_x, p.eps_inc_y, p.eps_inc
                )
                if Ea_inc < 0:
                    Ea_inc = 0.0  # no barrera negativa
                k_inc = arrhenius(p.nu0, Ea_inc, p.T, p.kB)
                # Ajuste de solvente en la tasa de incorporacion
                rates["incorp"] = k_inc * p.face_bias * self._solvent_factor("solvent_inc_factor")

        # ------------------------------
        # Caso 2: sitio con hemina (HEMIN)
        # ------------------------------
        elif g[i, j] == HEMIN:
            if p.enable_des:
                # Desorcion: mas dificil con mas vecinos
                Ea_des = p.Ea_des + self._bond_term(
                    bonds, bx, by, p.eps_des_x, p.eps_des_y, p.eps_des
                )
                k_des = arrhenius(p.nu0, Ea_des, p.T, p.kB)
                # Ajuste de solvente en la tasa de desorcion
                rates["desorb"] = k_des * self._solvent_factor("solvent_des_factor")

            if p.enable_diff:
                # Difusion anisotropica (b vs c) con sesgo direccional
                for dx, dy in VECINOS:
                    Ea_move = p.Ea_diff_b if dx != 0 else p.Ea_diff_c
                    Ea_move += self._bond_term(
                        bonds, bx, by, p.eps_move_x, p.eps_move_y, p.eps_move
                    )
                    kdiff = arrhenius(p.nu0, Ea_move, p.T, p.kB)
                    # Ajuste de solvente en la tasa de difusion
                    kdiff *= self._solvent_factor("solvent_diff_factor")
                    x, y = i + dx, j + dy
                    if in_bounds(x, y, *g.shape) and g[x, y] == EMPTY:
                        rates[("diff", dx, dy)] = kdiff

            if p.enable_incorp and is_step_site(g, i, j):
                # Bloqueo/incorporacion: mas favorable con mas vecinos
                Ea_inc = p.Ea_inc - self._bond_term(
                    bonds, bx, by, p.eps_inc_x, p.eps_inc_y, p.eps_inc
                )
                if Ea_inc < 0:
                    Ea_inc = 0.0
                k_lock = arrhenius(p.nu0, Ea_inc, p.T, p.kB)
                # Ajuste de solvente en la tasa de incorporacion
                rates["lock"] = k_lock * p.face_bias * self._solvent_factor("solvent_inc_factor")
        return rates

    def pick_event(self):
        """Selecciona un evento usando la ruleta de Gillespie."""
        totals = []
        Rtot = 0.0
        Lx, Ly = self.grid.shape
        for i in range(Lx):
            for j in range(Ly):
                for kind, r in self.local_rates(i, j).items():
                    if r > 0:
                        Rtot += r
                        totals.append((Rtot, kind, (i, j)))
        if Rtot == 0.0:
            return None, None, 0.0
        u = self.rng.random() * Rtot
        for cum, kind, ij in totals:
            if cum >= u:
                return kind, ij, Rtot
        return None, None, 0.0

    def do_event(self, kind, ij) -> None:
        """Aplica un evento al estado de la grilla."""
        i, j = ij
        g = self.grid
        if kind == "ads" and g[i, j] == EMPTY:
            g[i, j] = HEMIN
        elif kind == "incorp" and g[i, j] == EMPTY:
            g[i, j] = STEP
        elif kind == "desorb" and g[i, j] == HEMIN:
            g[i, j] = EMPTY
        elif kind == "lock" and g[i, j] == HEMIN:
            g[i, j] = STEP
        elif isinstance(kind, tuple) and kind[0] == "diff" and g[i, j] == HEMIN:
            dx, dy = kind[1], kind[2]
            x, y = i + dx, j + dy
            if in_bounds(x, y, *g.shape) and g[x, y] == EMPTY:
                g[i, j] = EMPTY
                g[x, y] = HEMIN

    def step(self) -> bool:
        """Ejecuta un paso KMC y avanza el tiempo."""
        kind, ij, Rtot = self.pick_event()
        if kind is None:
            return False
        self.do_event(kind, ij)
        # Avance en tiempo KMC (Gillespie)
        u = self.rng.random()
        self.t += (-10 * np.log(max(u, 1e-16)) / Rtot)
        return True

    def run_until(self, tmax: float = 10.0, snapshot_every: float = 1.0, store_snapshots: bool = False):
        """Corre la simulacion hasta un tiempo maximo."""
        times = []
        cover_total = []
        snapshots = []  # lista de (tiempo, grid)

        next_snapshot_t = snapshot_every
        i = 0

        while self.t < tmax:
            if i % 100 == 0:
                print(self.t)
            if not self.step():
                break
            occ = np.count_nonzero(self.grid > 0) / self.grid.size
            times.append(self.t)
            cover_total.append(occ)

            if store_snapshots and i % 100 == 0:
                snapshots.append((self.t, np.copy(self.grid)))
                next_snapshot_t += snapshot_every
            i += 1
        if store_snapshots:
            return np.array(times), np.array(cover_total), snapshots
        return np.array(times), np.array(cover_total), np.copy(self.grid)


# ----------------- Visualizacion -----------------
class KMC2DVisualizer:
    """Utilidades para visualizar resultados del modelo 2D."""

    def __init__(self, cmap: Optional[colors.Colormap] = None) -> None:
        """Inicializa el colormap y la normalizacion discreta."""
        # Colormap discreto para EMPTY/HEMIN/STEP
        if cmap is None:
            cmap = colors.ListedColormap(["white", "tab:blue", "tab:orange"])
        self.cmap = cmap
        self.norm = colors.BoundaryNorm([EMPTY, HEMIN, STEP, STEP + 1], self.cmap.N)

    def _unpack_snapshot(
        self,
        snapshot: Union[np.ndarray, Tuple[float, np.ndarray]],
    ) -> Tuple[Optional[float], np.ndarray]:
        """Normaliza el formato del snapshot."""
        # Acepta grilla sola o tupla (tiempo, grilla)
        if isinstance(snapshot, tuple) and len(snapshot) == 2:
            t, grid = snapshot
            return float(t), grid
        return None, snapshot

    def plot_snapshot(
        self,
        snapshot: Union[np.ndarray, Tuple[float, np.ndarray]],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        show_axes: bool = True,
        show: bool = True,
    ) -> plt.Axes:
        """Grafica un snapshot individual."""
        # Render de un snapshot como imagen
        t, grid = self._unpack_snapshot(snapshot)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(grid, cmap=self.cmap, norm=self.norm, origin="lower")
        # Mostrar dimensiones (x, y) de la red si se solicita.
        if show_axes:
            lx, ly = grid.shape
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xticks([0] if lx <= 1 else [0, lx])
            ax.set_yticks([0] if ly <= 1 else [0, ly])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if title is None and t is not None:
            title = f"t = {t:.3f}"
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return ax

    def animate_snapshots(
        self,
        snapshots: Sequence[Tuple[float, np.ndarray]],
        interval_ms: int = 200,
        show_axes: bool = True,
    ) -> animation.FuncAnimation:
        """Construye una animacion a partir de snapshots."""
        # Construye animacion con matplotlib
        if not snapshots:
            raise ValueError("snapshots is empty")
        fig, ax = plt.subplots()
        t0, grid0 = snapshots[0]
        im = ax.imshow(grid0, cmap=self.cmap, norm=self.norm, origin="lower")
        # Mostrar dimensiones (x, y) de la red si se solicita.
        if show_axes:
            lx, ly = grid0.shape
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xticks([0] if lx <= 1 else [0, lx - 1])
            ax.set_yticks([0] if ly <= 1 else [0, ly - 1])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"t = {t0:.3f}")

        def update(frame_index: int):
            t, grid = snapshots[frame_index]
            im.set_data(grid)
            ax.set_title(f"t = {t:.3f}")
            return (im,)

        return animation.FuncAnimation(
            fig,
            update,
            frames=len(snapshots),
            interval=interval_ms,
            blit=True,
        )

    def save_gif(
        self,
        snapshots: Sequence[Tuple[float, np.ndarray]],
        out_path: str,
        fps: int = 8,
        dpi: int = 120,
    ) -> None:
        """Guarda la animacion completa como GIF."""
        # Guarda la animacion como GIF (requiere Pillow instalado)
        anim = self.animate_snapshots(snapshots, interval_ms=int(1000 / fps))
        try:
            writer = animation.PillowWriter(fps=fps)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Pillow is required to save GIFs. Install it or export frames instead."
            ) from exc
        anim.save(out_path, writer=writer, dpi=dpi)
