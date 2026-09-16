import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource, LinearSegmentedColormap
try:
    from .bkl import KMC_BKL
except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
    from bkl import KMC_BKL

class Plotter_v3:
    def __init__(self, kmc: KMC_BKL):
        #self.lat = lattice
        self.kmc = kmc

    def _current_sigma(self) -> float:
        """Reconstruye sigma si el motor la expone; si no, usa la sobresaturación."""
        if hasattr(self.kmc, "sigma"):
            return float(self.kmc.sigma)
        if hasattr(self.kmc, "supersaturation"):
            return float(np.expm1(self.kmc.supersaturation))
        return 0.0

    def _current_S(self) -> float:
        """Devuelve la sobresaturación logarítmica interna del motor."""
        if hasattr(self.kmc, "supersaturation"):
            return float(self.kmc.supersaturation)
        sigma = self._current_sigma()
        sigma = max(sigma, -1.0 + 1e-15)
        return float(np.log1p(sigma))


    def _apply_academic_style(self, ax):
        """Aplica una estética profesional, nítida y académica."""
        fig = ax.figure
        fig.patch.set_facecolor("#FFFFFF")  # Fondo blanco puro de la figura
        
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Paneles blancos nítidos
            except Exception:
                pass
        
        # Usar negro puro para textos y etiquetas
        text_color = "#000000"
        ax.tick_params(colors=text_color, labelcolor=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        #ax.zaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        
        try:
            # Cuadrícula más tenue y nítida
            ax.grid(color="#CCCCCC", linestyle="-", linewidth=0.5, alpha=0.5)
        except Exception:
            pass
        
        # Aumentar tamaño de fuente para leyendas clave
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        #ax.zaxis.label.set_fontsize(12)
        ax.title.set_fontsize(14)
        ax.title.set_fontweight('bold')

    def _get_crystal_colors(self):
        """Define colores vibrantes y profesionales para el cristal: cian para el cuerpo, rojo intenso para el tope."""
        bulk_color = '#00FFFF' # Cian vibrante
        top_color = '#FF0000'  # Rojo puro
        return bulk_color, top_color

    def plot_crystal_3d(self, mode: str = "voxel", elev: int = 45, azim: int = 45,
                        cmap: str = "hemozoina", save_path: Optional[str] = None,
                        title: Optional[str] = None, snapshots: Optional[List[Tuple[float, np.ndarray, float]]] = None,
                        t_snapshot: Optional[float] = None):
        """
        Visualiza el cristal 3D (cubos discretos) con un estilo profesional y académico.

        Parámetros:
            mode: 'surface' o 'voxel'
            elev, azim: ángulos de cámara
            cmap: colormap para modo superficie
            save_path: ruta opcional para guardar la imagen
            title: título opcional
            snapshots: lista opcional de snapshots generada por run()
            t_snapshot: tiempo específico para extraer el cristal más cercano
        """

        # ============================
        # Seleccionar snapshot a graficar
        # ============================
        if snapshots is not None and len(snapshots) > 0 and t_snapshot is not None:
            # Busca el snapshot con tiempo más cercano
            times = [abs(t - t_snapshot) for t, _, _ in snapshots]
            idx = int(np.argmin(times))
            t_sel, heights, conv = snapshots[idx]
            print(f"🧩 Snapshot seleccionado: t={t_sel:.3f} s (conv={conv:.2f}%)")
        elif snapshots is not None and len(snapshots) > 0:
            # Toma el último snapshot si no se especifica tiempo
            t_sel, heights, conv = snapshots[-1]
            print(f"🧩 Usando último snapshot disponible: t={t_sel:.3f} s (conv={conv:.2f}%)")
        else:
            # Usa el estado actual del cristal
            heights = self.kmc.lat.heights.copy()
            t_sel = self.kmc.t
            conv = self.kmc.conversion_percent
            print(f"🧩 Usando estado actual: t={t_sel:.3f} s (conv={conv:.2f}%)")

        # ============================
        # Generar figura
        # ============================
        Lx, Ly = heights.shape
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        self._apply_academic_style(ax)  # Aplicar estilo académico
        ax.view_init(elev=elev, azim=azim)

        # Modo Voxel: Cubos discretos con colores nítidos
        if mode == "voxel":
            max_h = int(np.max(heights))
            max_h = max(max_h, 1)
            voxels = np.zeros((Lx, Ly, max_h), dtype=bool)
            for i in range(Lx):
                for j in range(Ly):
                    h = int(heights[i, j])
                    if h > 0:
                        voxels[i, j, :h] = True

            # Lógica de color nítida: Cuerpo cian, tope rojo
            bulk_color_hex, top_color_hex = self._get_crystal_colors()
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            
            bulk_rgba = list(plt.cm.colors.to_rgba(bulk_color_hex))
            top_rgba = list(plt.cm.colors.to_rgba(top_color_hex))
            top_rgba[3] = 1.0 # Opacidad completa

            for i in range(Lx):
                for j in range(Ly):
                    h = int(heights[i, j])
                    if h > 0:
                        # Color de las capas por debajo del tope (bulk)
                        for z in range(h - 1):
                            colors[i, j, z, :] = bulk_rgba
                        # Color de la capa superior (tope)
                        colors[i, j, h - 1, :] = top_rgba

            # Dibujar vóxeles con bordes nítidos y finos
            ax.voxels(voxels, facecolors=colors, edgecolor='#000000', linewidth=0.5)
            ax.set_box_aspect((Lx, Ly, max_h * 0.55))  # Relación de aspecto similar

        # Título dinámico enriquecido y claro
        if title is None:
            sigma = self._current_sigma()
            S = self._current_S()
            
            # Determinar régimen basado en sigma
            regimen = "Rough"
            if sigma < 2.5: regimen = "Spiral"
            elif sigma < 6.0: regimen = "Step"
            
            title = f"{regimen} Regime (σ={sigma:.2f}) | t={t_sel:.2f} s | conv={conv:.1f}% | S={S:.2f}"
        ax.set_title(title, pad=15)

        # Guardar imagen con fondo blanco para publicación académica
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor='white', transparent=False)
            print(f"💾 Imagen profesional guardada en: {save_path}")

        plt.show()

    def crystal_growth_gif(self, snapshots: List[Tuple[float, np.ndarray, float]],
                           save_path: str, mode: str = "voxel", elev: int = 45, azim: int = 45,
                           cmap: str = "hemozoina", fps: int = 10, interval_ms: int = 100, dpi: int = 150,
                           title_prefix: str = "Crystal Growth", repeat: bool = True, every_n: int = 1) -> None:
        """
        Genera un GIF animado del crecimiento del cristal con estilo académico.

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            save_path: ruta de salida del GIF (ej. 'growth.gif')
            mode: 'surface' o 'voxel'
            elev, azim: ángulos de cámara
            cmap: colormap para modo superficie
            fps: fotogramas por segundo del GIF
            interval_ms: intervalo entre frames en ms (visualización)
            dpi: resolución de guardado
            title_prefix: prefijo del título dinámico
            repeat: si la animación se repite al reproducirse
            every_n: usa 1 de cada N snapshots para reducir tamaño/costo
        """
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos para generar el GIF.")

        # Submuestreo opcional para acelerar y reducir tamaño de archivo
        frames_data = snapshots[::every_n]

        # Se infiere la geometría desde el primer snapshot
        _, h0, _ = frames_data[0]
        Lx, Ly = h0.shape

        # Escala global en Z para que no cambie entre frames (evita parpadeo)
        global_max_h = int(max(np.max(h) for _, h, _ in frames_data))
        if global_max_h < 1: global_max_h = 1

        # Preparación de figura y eje 3D con estilo académico
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        self._apply_academic_style(ax)

        def _draw_common_axes() -> None:
            """Dibuja configuración común del eje para cada frame."""
            ax.cla() # Limpiar eje
            self._apply_academic_style(ax) # Re-aplicar estilo académico
            ax.view_init(elev=elev, azim=azim)
            # Etiquetas y límites consistentes, más espaciadas para claridad
            ax.set_xlabel("x", labelpad=10)
            ax.set_ylabel("y", labelpad=10)
            ax.set_zlabel("height", labelpad=10)
            ax.set_xlim(0, max(Lx - 1, 1))
            ax.set_ylim(0, max(Ly - 1, 1))
            ax.set_zlim(0, global_max_h)

        def _update(frame_idx: int):
            """Actualiza un frame de la animación."""
            t_sel, heights, conv = frames_data[frame_idx]
            _draw_common_axes() # Configuración base de ejes

            # Vóxeles discretos con colores nítidos (cuerpo cian, tope rojo)
            if mode == "voxel":
                max_h_frame = int(np.max(heights))
                max_h_frame = max(max_h_frame, 1)

                voxels = np.zeros((Lx, Ly, max_h_frame), dtype=bool)
                for i in range(Lx):
                    for j in range(Ly):
                        h = int(heights[i, j])
                        if h > 0: voxels[i, j, :h] = True

                # Lógica de color nítida: Cuerpo cian, tope rojo
                bulk_color_hex, top_color_hex = self._get_crystal_colors()
                colors = np.zeros(voxels.shape + (4,), dtype=float)
                
                bulk_rgba = list(plt.cm.colors.to_rgba(bulk_color_hex))
                top_rgba = list(plt.cm.colors.to_rgba(top_color_hex))
                top_rgba[3] = 1.0 # Opacidad completa

                for i in range(Lx):
                    for j in range(Ly):
                        h = int(heights[i, j])
                        if h > 0:
                            # Color de las capas por debajo del tope (bulk)
                            for z in range(h - 1): colors[i, j, z, :] = bulk_rgba
                            # Color de la capa superior (tope)
                            colors[i, j, h - 1, :] = top_rgba

                # Dibujar vóxeles con bordes nítidos
                ax.voxels(voxels, facecolors=colors, edgecolor='#000000', linewidth=0.5)
                # Aspect ratio para percepción geométrica más estable
                ax.set_box_aspect((Lx, Ly, global_max_h * 0.55))

            # Título dinámico con tiempo y conversión, y régimen si es posible
            sigma = self._current_sigma()
            regimen = "Rough"
            if sigma < 2.5: regimen = "Spiral"
            elif sigma < 6.0: regimen = "Step"
            
            ax.set_title(f"{title_prefix} ({regimen} Regime)\nt={t_sel:.2f} s | conv={conv:.1f}%", pad=15)
            return (ax,)

        # Construcción de la animación
        anim = FuncAnimation(fig, _update, frames=len(frames_data), interval=interval_ms, blit=False, repeat=repeat)
        # Guardado a GIF
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"💾 GIF profesional guardado en: {save_path}")

    # # (Métodos de análisis no modificados)
    # def plot_growth_rate_analysis(self, stats: Dict, sigma: float, face: str,
    #                                 d_interplanar: float = 3.5e-3,
    #                                 time_unit_to_min: float = 1.0,
    #                                 save_path: Optional[str] = None): return None
    # def plot_adsorption_probabilities(self, stats: Dict, sigma: float, face: str,
    #                                      save_path: Optional[str] = None): return None
    # def plot_surface_morphology(self, heights: Optional[np.ndarray] = None,
    #                                cmap: str = "hemozoina",
    #                                show_histogram: bool = True,
    #                                save_path: Optional[str] = None): return None
    # def plot_multi_sigma_comparison(self, results_list: List[Dict],
    #                                  face: str,
    #                                  save_path: Optional[str] = None): return None
    # def plot_conversion(self, snapshots: List[Tuple[float, np.ndarray, float]], title: str,
    #                        figsize: Tuple[int, int] = (8, 5), save_path: Optional[str] = None): return None