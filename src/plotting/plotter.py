"""Plotter unificado para ambas líneas del motor.

Fusiona tres clases que tenían firmas idénticas pero estilos y métodos distintos:
- `src/plotter.py`      -> style="classic"  (Plotter)     + plot_conversion
- `plotter_v2.py`       -> style="v2"       (Plotter_v2)  + métodos de análisis
  (estaba en .descartables pero lo usaban los 4 pRun*.py y 2 notebooks)
- `src/plotter_v3.py`   -> style="academic" (Plotter_v3)

Los métodos de análisis y `plot_conversion` están disponibles con cualquier estilo.

Cambios respecto a los originales:
- Ya no importa `bkl` (el motor dinámico) para anotar tipos: eso acoplaba la línea
  estática a la dinámica. Se usa un `Protocol` con la interfaz mínima.
- El renderizado 3D/GIF de cada estilo reproduce el original; lo comprueba
  `tests/test_plotter.py` comparando PNG con los plotters originales.
- `plt.cm.get_cmap` (eliminado en matplotlib >= 3.9) se sustituye por
  `matplotlib.colormaps[...]`, que devuelve el mismo colormap.
- style="academic" con mode="surface" antes producía una figura vacía sin avisar;
  ahora lanza ValueError.
- `plot_growth_rate_analysis` mostraba `p.E_pb_over_kT` y `p.delta`, que no existen
  en los parámetros estáticos (x/y): ahora muestra el valor escalar o el par x/y.

Estilo nuevo (sin original de referencia):
- style="paper": morfología como en la Fig. 9 de Nagpal et al. (2024),
  `references/modern_kMC/`. Azul = capas completamente llenas (niveles por debajo
  de min(heights)); rojo = capa(s) en crecimiento (todo lo que está por encima).
  Se dibuja solo una losa fina (`base_layers` capas azules bajo el frente de
  crecimiento), sin ejes. `plot_morphology_sequence` compone varios snapshots en
  fila, como la figura del artículo.
"""

from typing import Dict, List, Optional, Protocol, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource, to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proyección 3D en mpl antiguos)

Snapshot = Tuple[float, np.ndarray, float]

STYLES = ("classic", "v2", "academic", "paper")

# Estilos que solo admiten mode="voxel"
_VOXEL_ONLY_STYLES = ("academic", "paper")

# ---- Paleta del estilo "paper" (Fig. 9 de Nagpal et al. 2024) ----
# Azul verdoso: capa completamente llena. Rojo oscuro: capa en crecimiento.
# Las aristas usan un tono más oscuro del mismo color (la rejilla fina del artículo).
_PAPER_FULL_FACE = "#2B8A96"
_PAPER_FULL_EDGE = "#14464D"
_PAPER_GROW_FACE = "#8B1212"
_PAPER_GROW_EDGE = "#470707"

# Valores por defecto de cada estilo, tomados de las firmas originales.
_STYLE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "classic": dict(elev=45, cmap="viridis", gif_mode="voxel", gif_interval_ms=100,
                    gif_title_prefix="Crystal growth"),
    "v2": dict(elev=30, cmap="terrain", gif_mode="surface", gif_interval_ms=200,
               gif_title_prefix="Crecimiento cristalino"),
    "academic": dict(elev=45, cmap="hemozoina", gif_mode="voxel", gif_interval_ms=100,
                     gif_title_prefix="Crystal Growth"),
    # cmap no se usa en "paper" (colores fijos por capa)
    "paper": dict(elev=35, cmap=None, gif_mode="voxel", gif_interval_ms=150,
                  gif_title_prefix="Crecimiento cristalino"),
}


class KMCLike(Protocol):
    """Interfaz mínima que el plotter necesita de un motor (cualquier línea)."""

    lat: object          # con atributo `heights` (np.ndarray 2D)
    t: float
    p: object            # parámetros (se usa `T` y, en análisis, energías/deltas)

    @property
    def conversion_percent(self) -> float: ...


def _get_cmap(name: str):
    """Obtiene un colormap por nombre (reemplazo de plt.cm.get_cmap)."""
    return matplotlib.colormaps[name]


def _column_voxels(heights: np.ndarray, depth: int) -> np.ndarray:
    """Matriz booleana (Lx, Ly, depth) con las columnas llenas hasta su altura."""
    Lx, Ly = heights.shape
    voxels = np.zeros((Lx, Ly, depth), dtype=bool)
    for i in range(Lx):
        for j in range(Ly):
            h = int(heights[i, j])
            if h > 0:
                voxels[i, j, :h] = True
    return voxels


def _paper_window_depth(heights: np.ndarray, base_layers: int) -> int:
    """Número de capas de vóxeles que necesita el estilo 'paper' para `heights`."""
    h = np.asarray(heights)
    return int(h.max()) - int(h.min()) + int(base_layers)


def _paper_voxels(heights: np.ndarray, base_layers: int = 1,
                  depth: Optional[int] = None):
    """Vóxeles y colores del estilo 'paper'.

    Criterio del artículo: una capa está "completamente llena" si todas las columnas
    llegan a ella, es decir, los niveles z < h_full con h_full = min(heights). Lo
    que queda por encima es la capa en crecimiento (rojo).

    La ventana vertical empieza `base_layers` capas por debajo de h_full. Los
    niveles z < 0 se tratan como cristal (sustrato SOS), así que la base azul
    existe aunque la superficie esté en h = 0.

    Parámetros:
        heights: alturas enteras (Lx, Ly)
        base_layers: capas azules que se dibujan bajo el frente (>= 1)
        depth: capas totales de la ventana; None = justo las necesarias. En GIF y
            secuencias se fija al máximo de todos los frames para que la escala
            vertical no cambie.

    Devuelve:
        (filled, facecolors, edgecolors, h_full)
    """
    if base_layers < 1:
        raise ValueError("base_layers debe ser >= 1 (si no, no hay base azul que dibujar).")
    h = np.asarray(heights, dtype=int)
    h_full = int(h.min())
    z0 = h_full - int(base_layers)  # nivel real de la primera capa de vóxeles
    if depth is None:
        depth = _paper_window_depth(h, base_layers)
    z = z0 + np.arange(max(int(depth), 1))

    # Una celda (i, j, k) está llena si su nivel real es menor que la altura de la columna
    filled = z[None, None, :] < h[:, :, None]
    full_layer = filled & (z[None, None, :] < h_full)
    growing = filled & ~full_layer

    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    edgecolors = np.zeros(filled.shape + (4,), dtype=float)
    facecolors[full_layer] = to_rgba(_PAPER_FULL_FACE)
    edgecolors[full_layer] = to_rgba(_PAPER_FULL_EDGE)
    facecolors[growing] = to_rgba(_PAPER_GROW_FACE)
    edgecolors[growing] = to_rgba(_PAPER_GROW_EDGE)
    return filled, facecolors, edgecolors, h_full


def _draw_paper(ax, heights: np.ndarray, base_layers: int = 1,
                depth: Optional[int] = None) -> int:
    """Dibuja una losa estilo 'paper' en `ax` (3D) y devuelve h_full."""
    filled, face, edge, h_full = _paper_voxels(heights, base_layers, depth)
    ax.voxels(filled, facecolors=face, edgecolors=edge, linewidth=0.3)
    Lx, Ly, n_layers = filled.shape
    # Vóxeles cúbicos: la losa se ve fina, como en el artículo
    ax.set_box_aspect((Lx, Ly, n_layers), zoom=1.15)
    ax.set_axis_off()
    return h_full


def _paper_legend_handles(h_full: Optional[int] = None) -> List[Patch]:
    """Leyenda del estilo 'paper' (capa completa / capa en crecimiento).

    Con `h_full` se indica el nivel de la última capa completa (una sola figura);
    en secuencias cambia de panel a panel y se omite.
    """
    full_label = "Capa completa" if h_full is None else f"Capa completa (h_min = {h_full})"
    return [
        Patch(facecolor=_PAPER_FULL_FACE, edgecolor=_PAPER_FULL_EDGE, label=full_label),
        Patch(facecolor=_PAPER_GROW_FACE, edgecolor=_PAPER_GROW_EDGE,
              label="Capa en crecimiento"),
    ]


def _regime_from_sigma(sigma: float) -> str:
    """Régimen cualitativo por umbrales de σ (criterio de plotter_v3)."""
    if sigma < 2.5:
        return "Spiral"
    if sigma < 6.0:
        return "Step"
    return "Rough"


class Plotter:
    """Visualización 3D, animaciones y análisis de una simulación kMC.

    Parámetros:
        kmc: motor de cualquiera de las dos líneas (KMC_BKL_Dynamic / KMC_BKL_Static)
        style: "classic" (Plotter), "v2" (Plotter_v2), "academic" (Plotter_v3)
               o "paper" (paleta azul/rojo de Nagpal et al. 2024)
    """

    def __init__(self, kmc: KMCLike, style: str = "classic"):
        if style not in STYLES:
            raise ValueError(f"style debe ser uno de {STYLES}, no {style!r}")
        self.kmc = kmc
        self.style = style

    # =====================================================================
    # Helpers comunes
    # =====================================================================
    def _default(self, key: str, value):
        """Devuelve `value` o, si es None, el valor por defecto del estilo."""
        return _STYLE_DEFAULTS[self.style][key] if value is None else value

    def _current_sigma(self) -> float:
        """σ = C/C_eq − 1 si el motor la expone; si no, se reconstruye desde S."""
        if hasattr(self.kmc, "sigma"):
            return float(self.kmc.sigma)
        if hasattr(self.kmc, "supersaturation"):
            return float(np.expm1(self.kmc.supersaturation))
        return 0.0

    def _current_S(self) -> float:
        """Sobresaturación logarítmica interna del motor, S = ln(1 + σ)."""
        if hasattr(self.kmc, "supersaturation"):
            return float(self.kmc.supersaturation)
        sigma = max(self._current_sigma(), -1.0 + 1e-15)
        return float(np.log1p(sigma))

    def _sigma_is_constant(self) -> bool:
        """True si σ no cambia en la corrida (σ fija o concentración constante).

        Solo entonces el σ actual del motor vale para cualquier snapshot; en la
        línea dinámica σ baja al agotarse el reservorio.
        """
        if getattr(self.kmc.p, "fixed_sigma", None) is not None:
            return True
        reservoir_is_dynamic = getattr(self.kmc, "_reservoir_is_dynamic", None)
        return reservoir_is_dynamic is not None and not reservoir_is_dynamic()

    def _paper_title(self, t_sel: float, conv: float) -> str:
        """Título del estilo 'paper': t y conversión del snapshot (+ σ si es constante)."""
        title = f"t = {t_sel:.2f} | conv = {conv:.1f} %"
        if self._sigma_is_constant():
            title += f" | σ = {self._current_sigma():.2f}"
        return title

    def _select_snapshot(self, snapshots: Optional[List[Snapshot]],
                         t_snapshot: Optional[float]) -> Tuple[float, np.ndarray, float]:
        """Elige el snapshot más cercano a t_snapshot, el último, o el estado actual."""
        if snapshots is not None and len(snapshots) > 0 and t_snapshot is not None:
            # Busca el snapshot con tiempo más cercano
            idx = int(np.argmin([abs(t - t_snapshot) for t, _, _ in snapshots]))
            t_sel, heights, conv = snapshots[idx]
            print(f"🧩 Snapshot seleccionado: t={t_sel:.3f} (conv={conv:.2f}%)")
        elif snapshots is not None and len(snapshots) > 0:
            # Toma el último snapshot si no se especifica tiempo
            t_sel, heights, conv = snapshots[-1]
            print(f"🧩 Usando último snapshot disponible: t={t_sel:.3f} (conv={conv:.2f}%)")
        else:
            # Usa el estado actual del cristal
            heights = self.kmc.lat.heights.copy()
            t_sel = self.kmc.t
            conv = self.kmc.conversion_percent
            print(f"🧩 Usando estado actual: t={t_sel:.3f} (conv={conv:.2f}%)")
        return t_sel, heights, conv

    # ---- Estilo académico (de plotter_v3) ----
    @staticmethod
    def _apply_academic_style(ax) -> None:
        """Estética académica: fondo blanco, texto negro, rejilla tenue."""
        fig = ax.figure
        fig.patch.set_facecolor("#FFFFFF")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            except Exception:
                pass
        text_color = "#000000"
        ax.tick_params(colors=text_color, labelcolor=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        try:
            ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.5, alpha=0.5)
        except Exception:
            pass
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontsize(14)
        ax.title.set_fontweight("bold")

    @staticmethod
    def _academic_colors(heights: np.ndarray, voxels: np.ndarray) -> np.ndarray:
        """Colores académicos: cuerpo cian y capa superior roja opaca."""
        Lx, Ly = heights.shape
        colors = np.zeros(voxels.shape + (4,), dtype=float)
        bulk_rgba = list(to_rgba("#00FFFF"))  # cian: cuerpo del cristal
        top_rgba = list(to_rgba("#FF0000"))   # rojo: capa superior
        top_rgba[3] = 1.0
        for i in range(Lx):
            for j in range(Ly):
                h = int(heights[i, j])
                if h > 0:
                    for z in range(h - 1):
                        colors[i, j, z, :] = bulk_rgba
                    colors[i, j, h - 1, :] = top_rgba
        return colors

    # =====================================================================
    # Figura 3D estática
    # =====================================================================
    def plot_crystal_3d(self, mode: str = "voxel", elev: Optional[int] = None, azim: int = 45,
                        cmap: Optional[str] = None, save_path: Optional[str] = None,
                        title: Optional[str] = None,
                        snapshots: Optional[List[Snapshot]] = None,
                        t_snapshot: Optional[float] = None, base_layers: int = 1):
        """Visualiza el cristal 3D (superficie continua o cubos discretos).

        Parámetros:
            mode: 'surface' o 'voxel' ('academic' y 'paper' solo admiten 'voxel')
            elev, azim: ángulos de cámara (elev por defecto según el estilo)
            cmap: colormap del modo superficie (por defecto según el estilo)
            save_path: ruta opcional para guardar la imagen
            title: título opcional (si es None se genera uno según el estilo)
            snapshots: lista opcional de snapshots generada por run()
            t_snapshot: tiempo del snapshot a mostrar (el más cercano)
            base_layers: solo estilo 'paper', capas azules bajo el frente de crecimiento
        """
        elev = self._default("elev", elev)
        cmap = self._default("cmap", cmap)
        if mode not in ("surface", "voxel"):
            raise ValueError("mode debe ser 'surface' o 'voxel'")
        if self.style in _VOXEL_ONLY_STYLES and mode != "voxel":
            raise ValueError(f"El estilo {self.style!r} solo admite mode='voxel'.")

        t_sel, heights, conv = self._select_snapshot(snapshots, t_snapshot)
        Lx, Ly = heights.shape
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        # Tamaño de figura de cada plotter original
        figsize = (6, 6) if self.style == "v2" else (7, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if self.style == "academic":
            self._apply_academic_style(ax)
        ax.view_init(elev=elev, azim=azim)

        if self.style == "classic":
            self._draw_3d_classic(fig, ax, heights, X, Y, mode, cmap)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("height")
            if title is None:
                title = f"Crystal at t={t_sel:.2f}, conv={conv:.1f}%"
            ax.set_title(title)
        elif self.style == "v2":
            self._draw_3d_v2(fig, ax, heights, X, Y, mode, cmap)
            ax.set_xlabel("X (lattice units)")
            ax.set_ylabel("Y (lattice units)")
            ax.set_zlabel("Altura (layers)")
            if title is None:
                title = (f"Cristal ({self.kmc.p.T:.0f}K) | t={t_sel:.2f} | "
                         f"σ={self._current_sigma():.2f} | S={self._current_S():.2f}")
            ax.set_title(title)
        elif self.style == "paper":
            fig.patch.set_facecolor("white")
            h_full = _draw_paper(ax, heights, base_layers)
            ax.legend(handles=_paper_legend_handles(h_full), loc="upper left",
                      frameon=False, fontsize=9)
            ax.set_title(self._paper_title(t_sel, conv) if title is None else title)
        else:  # academic
            max_h = max(int(np.max(heights)), 1)
            voxels = _column_voxels(heights, max_h)
            colors = self._academic_colors(heights, voxels)
            ax.voxels(voxels, facecolors=colors, edgecolor="#000000", linewidth=0.5)
            ax.set_box_aspect((Lx, Ly, max_h * 0.55))
            if title is None:
                sigma = self._current_sigma()
                title = (f"{_regime_from_sigma(sigma)} Regime (σ={sigma:.2f}) | t={t_sel:.2f} s | "
                         f"conv={conv:.1f}% | S={self._current_S():.2f}")
            ax.set_title(title, pad=15)

        if save_path:
            if self.style in _VOXEL_ONLY_STYLES:
                # Fondo blanco para publicación
                plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white",
                            transparent=False)
            else:
                plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")

        if self.style == "v2":
            plt.tight_layout()
        plt.show()

    @staticmethod
    def _draw_3d_classic(fig, ax, heights, X, Y, mode, cmap) -> None:
        """Renderizado de plotter.py: superficie con cmap o vóxeles azules uniformes."""
        Lx, Ly = heights.shape
        if mode == "surface":
            surf = ax.plot_surface(X, Y, heights, cmap=cmap, linewidth=0, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=10, label="Altura")
        else:
            max_h = int(np.max(heights))
            voxels = _column_voxels(heights, max_h + 1)
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            colors[..., :] = [0.2, 0.3, 0.8, 0.9]  # RGBA (azul translúcido)
            ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
            ax.set_box_aspect((Lx, Ly, max_h))

    @staticmethod
    def _draw_3d_v2(fig, ax, heights, X, Y, mode, cmap) -> None:
        """Renderizado de plotter_v2.py: superficie iluminada o vóxeles con gradiente."""
        Lx, Ly = heights.shape
        if mode == "surface":
            # Superficie continua con iluminación para relieve
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(heights.T, _get_cmap(cmap), vert_exag=0.1)
            ax.plot_surface(X, Y, heights, facecolors=rgb, linewidth=0, antialiased=True,
                            shade=False)
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(heights)
            fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Altura (layers)")
        else:
            max_h = max(int(np.max(heights)), 1)
            voxels = _column_voxels(heights, max_h)
            # Gradiente de color con la altura
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            for z in range(max_h):
                intensity = 0.3 + 0.7 * (z / max(max_h - 1, 1))
                colors[:, :, z, :] = [0.2 * intensity, 0.4 * intensity, 0.8, 0.9]
            # Capa superior de cada columna en verde marino
            top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
            ii, jj = np.where(heights > 0)
            colors[ii, jj, heights[ii, jj].astype(int) - 1, :] = top_layer_color
            ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.15)
            ax.set_box_aspect((Lx, Ly, max_h * 0.5))

    # =====================================================================
    # Animación GIF
    # =====================================================================
    def crystal_growth_gif(self, snapshots: List[Snapshot], save_path: str,
                           mode: Optional[str] = None, elev: Optional[int] = None,
                           azim: int = 45, cmap: Optional[str] = None, fps: int = 10,
                           interval_ms: Optional[int] = None, dpi: int = 150,
                           title_prefix: Optional[str] = None, repeat: bool = True,
                           every_n: int = 1, base_layers: int = 1) -> None:
        """Genera un GIF animado del crecimiento a partir de snapshots.

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            save_path: ruta de salida (debe terminar en .gif)
            mode: 'surface' o 'voxel' (por defecto según el estilo)
            elev, azim, cmap: cámara y colormap (por defecto según el estilo)
            fps, interval_ms, dpi: parámetros de la animación y del guardado
            title_prefix: prefijo del título de cada frame
            repeat: si la animación se repite
            every_n: usa 1 de cada N snapshots
            base_layers: solo estilo 'paper', capas azules bajo el frente de crecimiento
        """
        mode = self._default("gif_mode", mode)
        elev = self._default("elev", elev)
        cmap = self._default("cmap", cmap)
        interval_ms = self._default("gif_interval_ms", interval_ms)
        title_prefix = self._default("gif_title_prefix", title_prefix)

        # Validaciones (unión de las de los tres originales)
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos para generar el GIF.")
        if every_n < 1:
            raise ValueError("every_n debe ser >= 1.")
        if mode not in ("surface", "voxel"):
            raise ValueError("mode debe ser 'surface' o 'voxel'.")
        if self.style in _VOXEL_ONLY_STYLES and mode != "voxel":
            raise ValueError(f"El estilo {self.style!r} solo admite mode='voxel'.")
        if not save_path.lower().endswith(".gif"):
            raise ValueError("save_path debe terminar en '.gif'.")

        # Submuestreo y escala Z global (evita parpadeo entre frames)
        frames_data = snapshots[::every_n]
        _, h0, _ = frames_data[0]
        Lx, Ly = h0.shape
        global_max_h = max(int(max(np.max(h) for _, h, _ in frames_data)), 1)
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")
        # Estilo 'paper': misma altura de ventana en todos los frames (sin saltos de escala)
        paper_depth = max(_paper_window_depth(h, base_layers) for _, h, _ in frames_data)

        figsize = (9, 7) if self.style == "v2" else (7, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if self.style == "academic":
            self._apply_academic_style(ax)

        def _axes_frame() -> None:
            """Limpia y configura el eje común a cada frame."""
            ax.cla()
            if self.style == "paper":
                # Sin ejes: los límites los fija ax.voxels con la ventana común
                fig.patch.set_facecolor("white")
                ax.view_init(elev=elev, azim=azim)
                return
            if self.style == "academic":
                self._apply_academic_style(ax)
            ax.view_init(elev=elev, azim=azim)
            if self.style == "classic":
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("height")
            elif self.style == "v2":
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Altura")
            else:
                ax.set_xlabel("x", labelpad=10)
                ax.set_ylabel("y", labelpad=10)
                ax.set_zlabel("height", labelpad=10)
            ax.set_xlim(0, max(Lx - 1, 1))
            ax.set_ylim(0, max(Ly - 1, 1))
            ax.set_zlim(0, global_max_h)

        def _update(frame_idx: int):
            """Dibuja un frame de la animación."""
            t_sel, heights, conv = frames_data[frame_idx]
            _axes_frame()
            max_h_frame = max(int(np.max(heights)), 1)

            if self.style == "classic":
                if mode == "surface":
                    ax.plot_surface(X, Y, heights, cmap=cmap, linewidth=0, antialiased=True,
                                    vmin=0, vmax=global_max_h)
                else:
                    voxels = _column_voxels(heights, max_h_frame)
                    colors = np.zeros(voxels.shape + (4,), dtype=float)
                    colors[..., :] = [0.2, 0.3, 0.8, 0.9]
                    ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                    ax.set_box_aspect((Lx, Ly, global_max_h))
                ax.set_title(f"{title_prefix} | t={t_sel:.2f}, conv={conv:.1f}%")

            elif self.style == "v2":
                if mode == "surface":
                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(heights.T, _get_cmap(cmap), vmin=0, vmax=global_max_h,
                                   vert_exag=0.1)
                    ax.plot_surface(X, Y, heights, facecolors=rgb, linewidth=0,
                                    antialiased=True, shade=False)
                else:
                    voxels = _column_voxels(heights, max_h_frame)
                    colors = np.zeros(voxels.shape + (4,), dtype=float)
                    colors[..., :] = [0.2, 0.3, 0.8, 0.85]
                    top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
                    ii, jj = np.where(heights > 0)
                    colors[ii, jj, heights[ii, jj].astype(int) - 1, :] = top_layer_color
                    ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                    ax.set_box_aspect((Lx, Ly, global_max_h))
                # Nota: como en el original, σ y S son los del estado final del motor.
                ax.set_title(f"{title_prefix} | t={t_sel:.2f} | σ={self._current_sigma():.2f} | "
                             f"S={self._current_S():.2f}")

            elif self.style == "paper":
                h_full = _draw_paper(ax, heights, base_layers, depth=paper_depth)
                ax.legend(handles=_paper_legend_handles(h_full), loc="upper left",
                          frameon=False, fontsize=9)
                ax.set_title(f"{title_prefix}\n{self._paper_title(t_sel, conv)}")

            else:  # academic
                voxels = _column_voxels(heights, max_h_frame)
                colors = self._academic_colors(heights, voxels)
                ax.voxels(voxels, facecolors=colors, edgecolor="#000000", linewidth=0.5)
                ax.set_box_aspect((Lx, Ly, global_max_h * 0.55))
                regimen = _regime_from_sigma(self._current_sigma())
                ax.set_title(f"{title_prefix} ({regimen} Regime)\nt={t_sel:.2f} s | "
                             f"conv={conv:.1f}%", pad=15)
            return (ax,)

        anim = FuncAnimation(fig, _update, frames=len(frames_data), interval=interval_ms,
                             blit=False, repeat=repeat)
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"💾 GIF guardado en: {save_path}")

    # =====================================================================
    # Secuencia de morfologías (estilo Fig. 9 de Nagpal et al. 2024)
    # =====================================================================
    def plot_morphology_sequence(self, snapshots: List[Snapshot],
                                 times: Optional[List[float]] = None, n_panels: int = 3,
                                 base_layers: int = 1, elev: Optional[int] = None,
                                 azim: int = 45, panel_size: Tuple[float, float] = (4.5, 4.0),
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None):
        """Varios snapshots en fila con la paleta del artículo (azul/rojo).

        Siempre usa el renderizado 'paper', sea cual sea `self.style`.

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            times: tiempos a mostrar (se toma el snapshot más cercano a cada uno);
                None = `n_panels` snapshots equiespaciados, incluidos el primero y el último
            n_panels: número de paneles si `times` es None
            base_layers: capas azules bajo el frente de crecimiento
            elev, azim: cámara (elev por defecto 35, como el estilo 'paper')
            panel_size: tamaño (ancho, alto) de cada panel en pulgadas
            title: título general opcional
            save_path: ruta opcional para guardar la figura
        """
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos.")
        elev = _STYLE_DEFAULTS["paper"]["elev"] if elev is None else elev

        # Selección de snapshots
        if times is None:
            n = min(max(int(n_panels), 1), len(snapshots))
            idx = np.unique(np.linspace(0, len(snapshots) - 1, n).round().astype(int))
        else:
            snap_times = np.array([t for t, _, _ in snapshots])
            idx = [int(np.argmin(np.abs(snap_times - t))) for t in times]
        selected = [snapshots[i] for i in idx]

        # Ventana vertical común para que todas las losas tengan el mismo grosor
        depth = max(_paper_window_depth(h, base_layers) for _, h, _ in selected)

        n = len(selected)
        fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))
        fig.patch.set_facecolor("white")
        for k, (t_sel, heights, conv) in enumerate(selected):
            ax = fig.add_subplot(1, n, k + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            _draw_paper(ax, heights, base_layers, depth=depth)
            # Etiqueta de tiempo abajo a la izquierda, como en el artículo
            ax.text2D(0.02, 0.02, f"t = {t_sel:.2f}\nconv = {conv:.1f} %",
                      transform=ax.transAxes, fontsize=11, style="italic")

        fig.legend(handles=_paper_legend_handles(), loc="lower center", ncol=2,
                   frameon=False, fontsize=10)
        if title is None and self._sigma_is_constant():
            title = f"σ = {self._current_sigma():.2f}"
        if title:
            fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.92, wspace=0.0)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
            print(f"💾 Secuencia guardada en: {save_path}")
        plt.show()

    # =====================================================================
    # Series temporales y análisis
    # =====================================================================
    def plot_conversion(self, snapshots: List[Snapshot], title: str,
                        figsize: Tuple[int, int] = (8, 5), save_path: Optional[str] = None):
        """Conversión vs tiempo a partir de tuplas (t, heights|None, conversion). (plotter.py)"""
        times = [t for t, _, _ in snapshots]
        conversions = [conv for _, _, conv in snapshots]

        plt.figure(figsize=figsize)
        plt.plot(times, conversions, marker="o", linestyle="-", color="blue")
        plt.xlabel("Tiempo")
        plt.ylabel("Conversión (%)")
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")
        plt.show()

    def _param_text(self, scalar: str, x: str, y: str) -> str:
        """Texto de un parámetro escalar (línea dinámica) o del par x/y (estática)."""
        p = self.kmc.p
        if hasattr(p, scalar):
            return f"{getattr(p, scalar):.2f}"
        return f"{getattr(p, x):.2f} / {getattr(p, y):.2f} (x / y)"

    def plot_growth_rate_analysis(self, stats: Dict, sigma: float, face: str,
                                  d_interplanar: float = 3.5e-3,
                                  time_unit_to_min: float = 1.0,
                                  save_path: Optional[str] = None):
        """Velocidad de crecimiento vs tiempo con análisis de régimen (estilo Fig. 7).
        (plotter_v2.py; requiere `stats['height_history']` de la línea estática.)

        Parámetros:
            stats: diccionario retornado por run() con 'height_history'
            sigma: σ de la simulación
            face: cara cristalina ("110" o "101")
            d_interplanar: espaciado interplanar en μm (3.5e-3 para lisozima)
            time_unit_to_min: conversión de unidades de tiempo a minutos
        """
        height_history = np.array(stats["height_history"])
        if len(height_history) == 0:
            print("⚠️ No hay datos de historia de altura")
            return

        t = height_history[:, 0]
        h = height_history[:, 1]
        sigma_actual = height_history[:, 2]

        # Pendiente descartando el 20 % inicial (transitorio)
        skip = max(1, int(0.2 * len(t)))
        if len(t) > skip + 10:
            coeffs = np.polyfit(t[skip:], h[skip:], 1)
            growth_rate_layers = coeffs[0]
            fit_values = np.polyval(coeffs, t[skip:])
        else:
            growth_rate_layers = (h[-1] - h[0]) / (t[-1] - t[0]) if len(t) > 1 else 0
            fit_values = h[skip:]

        # Conversión a μm/min como en el paper
        growth_rate_um_min = growth_rate_layers * d_interplanar / time_unit_to_min

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: altura vs tiempo con ajuste lineal
        ax1 = axes[0, 0]
        ax1.plot(t, h, "b-", alpha=0.7, label="Simulación")
        ax1.plot(t[skip:], fit_values, "r--", linewidth=2,
                 label=f"Ajuste lineal: v={growth_rate_layers:.3f} layers/u.t.")
        ax1.set_xlabel("Tiempo (u.t.)")
        ax1.set_ylabel("Altura promedio (layers)")
        ax1.set_title(f"Crecimiento en cara ({face}) | σ = {sigma:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: verificación de σ constante
        ax2 = axes[0, 1]
        ax2.plot(t, sigma_actual, "g-", linewidth=1.5)
        ax2.axhline(y=sigma, color="r", linestyle="--", alpha=0.5, label="σ objetivo")
        ax2.set_xlabel("Tiempo (u.t.)")
        ax2.set_ylabel("Sigma")
        ax2.set_title("Verificación: σ constante (modo paper)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: velocidad instantánea (derivada suavizada)
        ax3 = axes[1, 0]
        if len(t) > 10:
            window = max(3, len(t) // 20)
            h_smooth = np.convolve(h, np.ones(window) / window, mode="same")
            v_instant = np.gradient(h_smooth, t)
            ax3.plot(t, v_instant, "purple", alpha=0.7, label="Velocidad instantánea")
            ax3.axhline(y=growth_rate_layers, color="r", linestyle="--",
                        label=f"Velocidad media: {growth_rate_layers:.3f}")
        ax3.set_xlabel("Tiempo (u.t.)")
        ax3.set_ylabel("Velocidad (layers/u.t.)")
        ax3.set_title("Velocidad de crecimiento instantánea")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: resumen
        ax4 = axes[1, 1]
        ax4.axis("off")
        if sigma < 2.5:
            regimen, color_reg = "SPIRAL (baja σ)", "yellow"
        elif sigma < 6.0:
            regimen, color_reg = "STEP (media σ)", "orange"
        else:
            regimen, color_reg = "ROUGH (alta σ)", "red"

        info_text = f"""
        PARÁMETROS DE SIMULACIÓN
        ─────────────────────────
        Cara cristalina: ({face})
        Sigma σ: {sigma:.2f}
        S = ln(1 + σ): {np.log1p(max(sigma, -1.0 + 1e-15)):.2f}
        Temperatura: {self.kmc.p.T:.1f} K

        PARÁMETROS DEL MODELO
        ─────────────────────
        E_pb/kT: {self._param_text("E_pb_over_kT", "E_pb_over_kT_x", "E_pb_over_kT_y")}
        φ/kT: {self.kmc.p.phi_over_kT:.2f}
        δ (delta): {self._param_text("delta", "delta_x", "delta_y")}
        K₀⁺: {self.kmc.p.K0_plus:.3f}

        RESULTADOS
        ──────────
        Régimen: {regimen}
        Velocidad: {growth_rate_layers:.4f} layers/u.t.
        Velocidad: {growth_rate_um_min:.4f} μm/min
        Altura final: {h[-1]:.2f} layers
        Rugosidad final: {np.std(self.kmc.lat.heights):.2f}

        Eventos totales: {stats['total_events']:,}
        Adsorciones: {stats['event_counts']['adsorption']:,}
        Desorciones: {stats['event_counts']['desorption']:,}
        Migraciones: {stats['event_counts']['migration']:,}
        """
        ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor=color_reg, alpha=0.3, pad=1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Análisis guardado en: {save_path}")
        plt.show()

        return {
            "growth_rate_layers": growth_rate_layers,
            "growth_rate_um_min": growth_rate_um_min,
            "regimen": regimen,
            "final_height": h[-1],
            "final_roughness": np.std(self.kmc.lat.heights),
        }

    def plot_adsorption_probabilities(self, stats: Dict, sigma: float, face: str,
                                      save_path: Optional[str] = None):
        """Probabilidades de adsorción por tipo de sitio (estilo Fig. 8). (plotter_v2.py)

        Requiere `stats['adsorption_probs_history']` (motor estático con
        record_adsorption_probs=True).
        """
        prob_hist = stats.get("adsorption_probs_history", [])
        if len(prob_hist) == 0:
            print("⚠️ No hay datos de probabilidades de adsorción "
                  "(¿record_adsorption_probs=False?)")
            return

        times = [p[0] for p in prob_hist]
        sigmas = [p[1] for p in prob_hist]
        p_adatom = [p[2].get(0, 0) for p in prob_hist]
        p_step = [p[2].get(1, 0) for p in prob_hist]
        p_kink = [p[2].get(2, 0) for p in prob_hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Izquierda: probabilidades vs tiempo
        ax1.plot(times, p_adatom, "b-", linewidth=2, label="Adatom (i=0)", marker="o", markersize=3)
        ax1.plot(times, p_step, "r-", linewidth=2, label="Step (i=1)", marker="s", markersize=3)
        ax1.plot(times, p_kink, "y-", linewidth=2, label="Kink-like (i≥2)", marker="^", markersize=3)
        ax1.set_xlabel("Tiempo (u.t.)")
        ax1.set_ylabel("Probabilidad relativa")
        ax1.set_title(f"Evolución de probabilidades de adsorción\nCara ({face}), σ={sigma:.2f}")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Derecha: probabilidades vs σ (comparación cualitativa con Fig. 8)
        ax2.plot(sigmas, p_adatom, "b-", linewidth=2, label="Adatom (i=0)", alpha=0.7)
        ax2.plot(sigmas, p_step, "r-", linewidth=2, label="Step (i=1)", alpha=0.7)
        ax2.plot(sigmas, p_kink, "y-", linewidth=2, label="Kink-like (i≥2)", alpha=0.7)
        sigma_range = np.linspace(1, 8, 100)
        # Curvas "teóricas" aproximadas del original (solo guía visual, no del paper)
        p_kink_theory = np.exp(-sigma_range * 0.5) / (1 + np.exp(-sigma_range * 0.5))
        p_adatom_theory = 1 - p_kink_theory - 0.3
        ax2.plot(sigma_range, p_adatom_theory, "b--", alpha=0.3, label="Tendencia esperada")
        ax2.plot(sigma_range, p_kink_theory, "y--", alpha=0.3)
        ax2.set_xlabel("Sigma σ")
        ax2.set_ylabel("Probabilidad relativa")
        ax2.set_title("Probabilidades vs σ (Comparación Figura 8)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Probabilidades guardadas en: {save_path}")
        plt.show()

    def plot_surface_morphology(self, heights: Optional[np.ndarray] = None,
                                cmap: str = "terrain", show_histogram: bool = True,
                                save_path: Optional[str] = None):
        """Mapa 2D de alturas e histograma con rugosidad. (plotter_v2.py)"""
        if heights is None:
            heights = self.kmc.lat.heights

        fig = plt.figure(figsize=(14, 6) if show_histogram else (8, 7))

        # Panel 1: mapa de alturas
        ax1 = fig.add_subplot(121 if show_histogram else 111)
        im = ax1.imshow(heights, cmap=cmap, origin="lower", interpolation="nearest")
        plt.colorbar(im, ax=ax1, label="Altura (layers)", fraction=0.046)
        ax1.set_title("Morfología Superficial 2D")
        ax1.set_xlabel("X (lattice units)")
        ax1.set_ylabel("Y (lattice units)")

        if show_histogram:
            # Panel 2: histograma y estadísticos
            ax2 = fig.add_subplot(122)
            ax2.hist(heights.flatten(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            ax2.axvline(np.mean(heights), color="r", linestyle="--", linewidth=2,
                        label=f"Media: {np.mean(heights):.2f}")
            ax2.axvline(np.median(heights), color="g", linestyle="--", linewidth=2,
                        label=f"Mediana: {np.median(heights):.2f}")
            ax2.set_xlabel("Altura (layers)")
            ax2.set_ylabel("Frecuencia")
            ax2.set_title(f"Distribución de Alturas\nσ_h = {np.std(heights):.2f}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"💾 Morfología guardada en: {save_path}")
        plt.show()

    def plot_multi_sigma_comparison(self, results_list: List[Dict], face: str,
                                    save_path: Optional[str] = None):
        """Velocidad de crecimiento vs σ para varias corridas (Fig. 7 completa). (plotter_v2.py)

        Parámetros:
            results_list: dicts con 'sigma', 'growth_rate_layers' y opcional 'growth_rate_std'
        """
        if not results_list:
            print("⚠️ Lista de resultados vacía")
            return

        results_sorted = sorted(results_list, key=lambda x: x["sigma"])
        sigmas = [r["sigma"] for r in results_sorted]
        rates = [r["growth_rate_layers"] for r in results_sorted]
        rates_err = [r.get("growth_rate_std", 0) for r in results_sorted]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: velocidad vs σ con regiones de régimen
        ax1.errorbar(sigmas, rates, yerr=rates_err, marker="o", markersize=8, linewidth=2,
                     capsize=5, color="darkblue", label=f"Cara ({face}) - Modelo kMC")
        ax1.axvspan(0, 2.5, alpha=0.2, color="yellow", label="Régimen Spiral")
        ax1.axvspan(2.5, 6.0, alpha=0.2, color="orange", label="Régimen Step")
        ax1.axvspan(6.0, 8.5, alpha=0.2, color="red", label="Régimen Rough")
        ax1.set_xlabel("Supersaturación σ", fontsize=12)
        ax1.set_ylabel("Velocidad de crecimiento (layers/u.t.)", fontsize=12)
        ax1.set_title(f"Figura 7 - Crecimiento en cara ({face})\n"
                      "Lysozima HEW (Nagpal et al. 2024)", fontsize=13)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 8.5)

        # Panel 2: log-log para exponentes
        ax2.loglog(sigmas, rates, "o-", color="darkgreen", linewidth=2, markersize=8)
        ax2.set_xlabel("Supersaturación σ (log)", fontsize=12)
        ax2.set_ylabel("Velocidad (log)", fontsize=12)
        ax2.set_title("Análisis de exponentes de crecimiento")
        ax2.grid(True, alpha=0.3, which="both")
        step_mask = [(2.5 <= s <= 6.0) for s in sigmas]
        if sum(step_mask) > 2:
            step_sigmas = np.array([s for s, m in zip(sigmas, step_mask) if m])
            step_rates = np.array([r for r, m in zip(rates, step_mask) if m])
            log_fit = np.polyfit(np.log(step_sigmas), np.log(step_rates), 1)
            ax2.plot(step_sigmas, np.exp(log_fit[1]) * step_sigmas ** log_fit[0], "r--",
                     label=f"Pendiente ≈ {log_fit[0]:.2f}")
            ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Comparación guardada en: {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Alias heredados: subclases de una línea para que isinstance() siga funcionando
# ---------------------------------------------------------------------------
class Plotter_v2(Plotter):
    """Alias de `Plotter(kmc, style="v2")` (antes en plotter_v2.py)."""

    def __init__(self, kmc: KMCLike, style: str = "v2"):
        super().__init__(kmc, style=style)


class Plotter_v3(Plotter):
    """Alias de `Plotter(kmc, style="academic")` (antes en plotter_v3.py)."""

    def __init__(self, kmc: KMCLike, style: str = "academic"):
        super().__init__(kmc, style=style)
