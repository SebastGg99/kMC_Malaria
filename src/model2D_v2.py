"""Modelo KMC 2D con anisotropia de forma.

Descripcion
----------
Este modulo replica la logica del modelo 2D y agrega anisotropia de forma
usando enlaces direccionales (x/y) para ajustar las barreras de energia.
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
    count_bonds_xy,
    count_bonds_all,
    is_step_site,
    arrhenius,
)


# Considerando step/kink
@dataclass
class Params2D:
    """Parametros del modelo 2D (v2) con anisotropia de forma.

    Las correcciones por vecino ahora se separan por direccion (x/y).
    Si se mantienen iguales, el modelo se comporta isotropicamente.
    """

    # Barreras base (eV)
    Ead: float = 0.25
    Ede: float = 0.0
    Eai: float = 0.0
    Eab: float = 0.0
    Eac: float = 0.0

    # Correcciones por vecino (anisotropia de forma)
    Jad_x: float = 0.04
    Jad_y: float = 0.04
    Jinc_x: float = 0.01
    Jinc_y: float = 0.01
    Jdes_x: float = 0.05
    Jdes_y: float = 0.05
    Jmove_x: float = 0.03
    Jmove_y: float = 0.03

    # Potencial quimico efectivo
    mu: float = -0.10

    # Parametros fijos del sistema
    Lx: int = field(init=False, default=60)
    Ly: int = field(init=False, default=60)
    T: float = field(init=False, default=298.15)
    kB: float = field(init=False, default=8.617e-5)
    nu0: float = field(init=False, default=1e8)

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

        # Potencial quimico
        self.mu_hemin = self.mu


# ----------------- Nucleo KMC -----------------
class KMC2D:
    """Motor KMC 2D con anisotropia de forma (enlaces direccionales)."""

    def __init__(self, p: Params2D, seed: int = 0) -> None:
        """Inicializa la simulacion y coloca una semilla en el centro."""
        self.p = p
        self.grid = np.zeros((p.Lx, p.Ly), dtype=np.int8)
        self.rng = random.Random(seed)
        self.t = 0.0
        # Semilla: un adatom en el centro
        self.grid[p.Lx // 2, p.Ly // 2] = HEMIN

    def local_rates(self, i: int, j: int) -> dict:
        """Calcula tasas locales de eventos en el sitio (i, j)."""
        g, p = self.grid, self.p
        rates = {}

        # Enlaces direccionales (x/y) para anisotropia de forma
        bx, by = count_bonds_xy(g, i, j)
        bonds = bx + by

        # ------------------------------
        # Caso 1: sitio vacio (EMPTY)
        # ------------------------------
        if g[i, j] == EMPTY:
            if p.enable_ads:
                # Adsorcion: menos favorable con mas vecinos
                Ea = p.Ea_ads + bx * p.eps_ads_x + by * p.eps_ads_y
                k_ads = arrhenius(p.nu0, Ea - p.mu_hemin, p.T, p.kB) * p.activity_hemin
                if bonds > 0:
                    rates["ads"] = k_ads

            if p.enable_incorp and is_step_site(g, i, j):
                # Incorporacion: mas favorable con mas vecinos
                Ea_inc = p.Ea_inc - (bx * p.eps_inc_x + by * p.eps_inc_y)
                if Ea_inc < 0:
                    Ea_inc = 0.0  # no barrera negativa
                rates["incorp"] = arrhenius(p.nu0, Ea_inc, p.T, p.kB) * p.face_bias

        # ------------------------------
        # Caso 2: sitio con hemina (HEMIN)
        # ------------------------------
        elif g[i, j] == HEMIN:
            if p.enable_des:
                # Desorcion: mas dificil con mas vecinos
                Ea_des = p.Ea_des + bx * p.eps_des_x + by * p.eps_des_y
                rates["desorb"] = arrhenius(p.nu0, Ea_des, p.T, p.kB)

            if p.enable_diff:
                # Difusion anisotropica (b vs c) con sesgo direccional
                for dx, dy in VECINOS:
                    Ea_move = p.Ea_diff_b if dx != 0 else p.Ea_diff_c
                    Ea_move += bx * p.eps_move_x + by * p.eps_move_y
                    kdiff = arrhenius(p.nu0, Ea_move, p.T, p.kB)
                    x, y = i + dx, j + dy
                    if in_bounds(x, y, *g.shape) and g[x, y] == EMPTY:
                        rates[("diff", dx, dy)] = kdiff

            if p.enable_incorp and is_step_site(g, i, j):
                # Bloqueo/incorporacion: mas favorable con mas vecinos
                Ea_inc = p.Ea_inc - (bx * p.eps_inc_x + by * p.eps_inc_y)
                if Ea_inc < 0:
                    Ea_inc = 0.0
                rates["lock"] = arrhenius(p.nu0, Ea_inc, p.T, p.kB) * p.face_bias
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
        show: bool = True,
    ) -> plt.Axes:
        """Grafica un snapshot individual."""
        # Render de un snapshot como imagen
        t, grid = self._unpack_snapshot(snapshot)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(grid, cmap=self.cmap, norm=self.norm, origin="lower")
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
    ) -> animation.FuncAnimation:
        """Construye una animacion a partir de snapshots."""
        # Construye animacion con matplotlib
        if not snapshots:
            raise ValueError("snapshots is empty")
        fig, ax = plt.subplots()
        t0, grid0 = snapshots[0]
        im = ax.imshow(grid0, cmap=self.cmap, norm=self.norm, origin="lower")
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
