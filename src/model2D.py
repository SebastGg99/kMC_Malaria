"""Modelo KMC 2D con parametros, dinamica y utilidades de visualizacion.

Descripcion
----------
Este modulo implementa un modelo de crecimiento 2D por KMC (Gillespie) con:
- Un contenedor de parametros fisicos y de control.
- Un motor de simulacion con eventos locales y avance temporal.
- Una clase de visualizacion para snapshots y animaciones.

Notas
-----
Las constantes y funciones de vecindad se importan desde utils para mantener
consistencia con el resto del proyecto.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib import animation, colors
# try:
#     from .utils import EMPTY, HEMIN, STEP, VECINOS, in_bounds, count_bonds, count_bonds_all, is_step_site, arrhenius
# except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
#     from utils import EMPTY, HEMIN, STEP, VECINOS, in_bounds, count_bonds, count_bonds_all, is_step_site, arrhenius

from utils import EMPTY, HEMIN, STEP, VECINOS, in_bounds, count_bonds, count_bonds_all, is_step_site, arrhenius

#Considerando stepkink
@dataclass
class Params2D:
    """Parametros del modelo 2D y banderas de procesos.

    Descripcion
    ----------
    Define los parametros de barrera, efectos de vecinos y control de eventos.
    El modelo usa nombres de atributos internos (Ea_ads, eps_ads, mu_hemin, ...)
    que se derivan de las entradas en __post_init__.

    Parametros
    ----------
    Ead : float
        Barrera base de adsorcion (eV).
    Ede : float
        Barrera base de desorcion (eV).
    Eai : float
        Barrera base de incorporacion (eV).
    Eab : float
        Barrera base de difusion en la direccion b (eV).
    Eac : float
        Barrera base de difusion en la direccion c (eV).
    Jad : float
        Correccion por vecino para adsorcion (interacción lateral).
    Jinc : float
        Correccion por vecino para incorporacion (interacción lateral).
    Jdes : float
        Correccion por vecino para desorcion (interacción lateral).
    Jmove : float
        Correccion por vecino para difusion (interacción lateral).
    mu : float
        Potencial quimico efectivo del bano.

    Notas
    -----
    - Los tamaños de red y constantes fisicas se fijan como atributos internos.
    - Las banderas enable_* permiten activar o desactivar procesos.
    """
    Ead: float = 0.25
    Ede: float = 0.0
    Eai: float = 0.0
    Eab: float = 0.0
    Eac: float = 0.0
    Jad: float = 0.04
    Jinc: float = 0.01
    Jdes: float = 0.05
    Jmove: float = 0.03
    mu: float = -0.10

    Lx: int = field(init=False, default=60)
    Ly: int = field(init=False, default=60)
    T: float = field(init=False, default=298.15)
    kB: float = field(init=False, default=8.617e-5)
    nu0: float = field(init=False, default=1e8)

    Ea_ads: float = field(init=False)
    Ea_des: float = field(init=False)
    Ea_inc: float = field(init=False)
    Ea_diff_b: float = field(init=False)
    Ea_diff_c: float = field(init=False)

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

    def __post_init__(self):
        """Construye atributos internos compatibles con KMC2D.

        Descripcion
        ----------
        Mapea las entradas (Ead, Jad, mu, ...) a los nombres usados por el motor
        de simulacion (Ea_ads, eps_ads, mu_hemin, ...).

        Retorna
        -------
        None
        """
        # Map inputs to the attribute names used by KMC2D
        self.Ea_ads = self.Ead
        self.Ea_des = self.Ede
        self.Ea_inc = self.Eai
        self.Ea_diff_b = self.Eab
        self.Ea_diff_c = self.Eac
        self.eps_ads = self.Jad
        self.eps_inc = self.Jinc
        self.eps_des = self.Jdes
        self.eps_move = self.Jmove
        self.mu_hemin = self.mu

# ----------------- Núcleo KMC -----------------
class KMC2D:
    """Motor KMC 2D basado en eventos locales y seleccion estocastica.

    Descripcion
    ----------
    Representa una grilla 2D con estados EMPTY/HEMIN/STEP y ejecuta pasos KMC
    mediante la tasa total de eventos disponibles.

    Parametros
    ----------
    p : Params2D
        Conjunto de parametros fisicos y banderas de procesos.
    seed : int, optional
        Semilla del generador de numeros aleatorios.
    """
    def __init__(self, p, seed=0):
        """Inicializa la simulacion y coloca una semilla en el centro.

        Parametros
        ----------
        p : Params2D
            Conjunto de parametros para la simulacion.
        seed : int, optional
            Semilla para reproducibilidad.

        Retorna
        -------
        None
        """
        self.p = p
        self.grid = np.zeros((p.Lx, p.Ly), dtype=np.int8)
        self.rng = random.Random(seed)
        self.t = 0.0
        # Semilla: un adátomo en el centro
        self.grid[p.Lx//2, p.Ly//2] = HEMIN

    def local_rates(self, i, j):
      """Calcula tasas locales de eventos en el sitio (i, j).

      Descripcion
      ----------
      Construye un diccionario de eventos posibles y sus tasas de Arrhenius
      considerando el estado local (EMPTY/HEMIN), la vecindad y banderas.

      Parametros
      ----------
      i : int
          Coordenada x del sitio.
      j : int
          Coordenada y del sitio.

      Retorna
      -------
      dict
          Mapea tipo de evento a tasa, por ejemplo {"ads": k, ("diff", dx, dy): k}.
      """
      g, p = self.grid, self.p
      rates = {}

      # contar vecinos ocupados (HEMIN o STEP)
      bonds = count_bonds(g, i, j)

      # ------------------------------
      # Caso 1: sitio vacío (EMPTY)
      # ------------------------------
      if g[i,j] == EMPTY:
          if p.enable_ads:
              # Adsorción: menos favorable con más vecinos
              Ea = p.Ea_ads + bonds * p.eps_ads
              k_ads = arrhenius(p.nu0, Ea - p.mu_hemin, p.T, p.kB) * p.activity_hemin
              if bonds > 0:
                rates["ads"] = k_ads


          if p.enable_incorp and is_step_site(g, i, j):
              # Incorporación: más favorable con más vecinos
              Ea_inc = p.Ea_inc - bonds * p.eps_inc
              if Ea_inc < 0: Ea_inc = 0.0  # no barrera negativa
              rates["incorp"] = arrhenius(p.nu0, Ea_inc, p.T, p.kB) * p.face_bias

        # ------------------------------
        # Caso 2: sitio con hemina (HEMIN)
        # ------------------------------
      elif g[i,j] == HEMIN:
          if p.enable_des:
              # Desorción: más difícil con más vecinos
              Ea_des = p.Ea_des + bonds * p.eps_des
              rates["desorb"] = arrhenius(p.nu0, Ea_des, p.T, p.kB)

          if p.enable_diff:
              # Difusión anisotrópica
              for dx, dy in VECINOS:
                  Ea_move = p.Ea_diff_b if dx != 0 else p.Ea_diff_c
                  Ea_move += bonds * p.eps_move
                  kdiff = arrhenius(p.nu0, Ea_move, p.T, p.kB)
                  x, y = i+dx, j+dy
                  if in_bounds(x, y, *g.shape) and g[x, y] == EMPTY:
                      rates[("diff", dx, dy)] = kdiff

          if p.enable_incorp and is_step_site(g, i, j):
              # Bloqueo/incorporación: más favorable con más vecinos
              Ea_inc = p.Ea_inc - bonds * p.eps_inc
              if Ea_inc < 0: Ea_inc = 0.0
              rates["lock"] = arrhenius(p.nu0, Ea_inc, p.T, p.kB) * p.face_bias
      return rates


    def pick_event(self):
        """Selecciona un evento usando la ruleta de Gillespie.

        Descripcion
        ----------
        Recorre todos los sitios, acumula tasas y elige un evento proporcional
        a su tasa total.

        Retorna
        -------
        tuple
            (kind, ij, Rtot), donde kind es el evento, ij el sitio y Rtot la
            tasa total. Si no hay eventos, devuelve (None, None, 0.0).
        """
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

    def do_event(self, kind, ij):
        """Aplica un evento al estado de la grilla.

        Parametros
        ----------
        kind : str o tuple
            Tipo de evento ("ads", "incorp", "desorb", "lock" o ("diff", dx, dy)).
        ij : tuple
            Coordenadas (i, j) donde ocurre el evento.

        Retorna
        -------
        None
        """
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
            x, y = i+dx, j+dy
            if in_bounds(x, y, *g.shape) and g[x, y] == EMPTY:
                g[i, j] = EMPTY
                g[x, y] = HEMIN

    def step(self):
        """Ejecuta un paso KMC y avanza el tiempo.

        Descripcion
        ----------
        Selecciona un evento, lo aplica y actualiza el tiempo usando
        el algoritmo de Gillespie.

        Retorna
        -------
        bool
            True si se ejecuto un evento, False si no hay eventos disponibles.
        """
        kind, ij, Rtot = self.pick_event()
        if kind is None:
            return False
        self.do_event(kind, ij)
        # Avance en tiempo KMC (Gillespie)
        u = self.rng.random()
        self.t += (-10*np.log(max(u, 1e-16)) / Rtot)

        return True

    def run_until(self, tmax=10.0, snapshot_every=1.0, store_snapshots=False):
        """Corre la simulacion hasta un tiempo maximo.

        Parametros
        ----------
        tmax : float, optional
            Tiempo maximo de simulacion.
        snapshot_every : float, optional
            Intervalo de tiempo para guardar snapshots (si se activan).
        store_snapshots : bool, optional
            Si True, guarda snapshots en la lista de salida.

        Retorna
        -------
        tuple
            (times, cover_total, snapshots|grid) segun store_snapshots.
        """
        times = []
        cover_total = []
        snapshots = []   # lista de (tiempo, grid)

        next_snapshot_t = snapshot_every
        i=0
        
        while self.t < tmax:

            if(i%100==0.0):
              print(self.t)
            if not self.step():
                break
            occ = np.count_nonzero(self.grid > 0) / self.grid.size
            times.append(self.t)
            cover_total.append(occ)

            if store_snapshots and i%100==0.0:
                snapshots.append((self.t, np.copy(self.grid)))
                next_snapshot_t += snapshot_every
            i=i+1
        if store_snapshots:
            return np.array(times), np.array(cover_total), snapshots
        else:
            return np.array(times), np.array(cover_total), np.copy(self.grid)

# ----------------- Visualizacion -----------------
class KMC2DVisualizer:
    """Utilidades para visualizar resultados del modelo 2D.

    Descripcion
    ----------
    Provee metodos para graficar snapshots individuales y construir
    animaciones a partir de listas de snapshots.

    Parametros
    ----------
    cmap : matplotlib.colors.Colormap, optional
        Colormap discreto para los estados EMPTY/HEMIN/STEP.
    """
    def __init__(self, cmap: Optional[colors.Colormap] = None):
        """Inicializa el colormap y la normalizacion discreta.

        Parametros
        ----------
        cmap : matplotlib.colors.Colormap, optional
            Colormap discreto para la visualizacion.

        Retorna
        -------
        None
        """
        # Use a discrete colormap for EMPTY/HEMIN/STEP values
        if cmap is None:
            cmap = colors.ListedColormap(["white", "tab:blue", "tab:orange"])
        self.cmap = cmap
        self.norm = colors.BoundaryNorm([EMPTY, HEMIN, STEP, STEP + 1], self.cmap.N)

    def _unpack_snapshot(
        self,
        snapshot: Union[np.ndarray, Tuple[float, np.ndarray]],
    ) -> Tuple[Optional[float], np.ndarray]:
        """Normaliza el formato del snapshot.

        Parametros
        ----------
        snapshot : np.ndarray o tuple
            Puede ser la grilla sola o un par (tiempo, grilla).

        Retorna
        -------
        tuple
            (tiempo, grilla) donde tiempo puede ser None si no fue provisto.
        """
        # Accept either a raw grid or a (time, grid) tuple
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
        """Grafica un snapshot individual.

        Parametros
        ----------
        snapshot : np.ndarray o tuple
            Grilla o par (tiempo, grilla) a visualizar.
        ax : matplotlib.axes.Axes, optional
            Eje donde se dibuja. Si es None, se crea una figura nueva.
        title : str, optional
            Titulo explicito. Si no se da y hay tiempo, usa "t = ...".
        show : bool, optional
            Si True, ejecuta plt.show().

        Retorna
        -------
        matplotlib.axes.Axes
            Eje con la imagen dibujada.
        """
        # Render a single snapshot as an image
        t, grid = self._unpack_snapshot(snapshot)
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(grid, cmap=self.cmap, norm=self.norm, origin="lower")
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
        """Construye una animacion a partir de snapshots.

        Parametros
        ----------
        snapshots : sequence
            Lista de (tiempo, grilla) en orden temporal.
        interval_ms : int, optional
            Intervalo entre frames en milisegundos.

        Retorna
        -------
        matplotlib.animation.FuncAnimation
            Animacion lista para mostrar o guardar.
        """
        # Build a matplotlib animation from a list of snapshots
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
        """Guarda la animacion completa como GIF.

        Parametros
        ----------
        snapshots : sequence
            Lista de (tiempo, grilla) en orden temporal.
        out_path : str
            Ruta de salida del GIF.
        fps : int, optional
            Frames por segundo.
        dpi : int, optional
            Resolucion para el guardado.

        Retorna
        -------
        None

        Notas
        -----
        Requiere Pillow instalado para usar el writer de GIF de matplotlib.
        """
        # Save the animation as GIF (requires Pillow installed for matplotlib)
        anim = self.animate_snapshots(snapshots, interval_ms=int(1000 / fps))
        try:
            writer = animation.PillowWriter(fps=fps)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Pillow is required to save GIFs. Install it or export frames instead."
            ) from exc
        anim.save(out_path, writer=writer, dpi=dpi)
