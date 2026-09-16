import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
try:
    from .bkl import KMC_BKL
except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
    from bkl import KMC_BKL

class Plotter:
    def __init__(self, kmc: KMC_BKL):
        #self.lat = lattice
        self.kmc = kmc

    def plot_crystal_3d(self, mode: str = "voxel", elev: int = 45, azim: int = 45,
                        cmap: str = "viridis", save_path: Optional[str] = None,
                        title: Optional[str] = None, snapshots: Optional[List[Tuple[float, np.ndarray, float]]] = None,
                        t_snapshot: Optional[float] = None):
        """
        Visualiza el cristal 3D (superficie continua o cubos discretos).

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

        # ============================
        # Generar figura
        # ============================
        Lx, Ly = heights.shape
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        if mode == "surface":
            surf = ax.plot_surface(X, Y, heights, cmap=cmap, linewidth=0, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=10, label="Altura")

        elif mode == "voxel":
            max_h = int(np.max(heights))
            voxels = np.zeros((Lx, Ly, max_h + 1), dtype=bool)
            for i in range(Lx):
                for j in range(Ly):
                    h = int(heights[i, j])
                    if h > 0:
                        voxels[i, j, :h] = True

            # Colores tipo cristal (azul translúcido)
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            colors[..., :] = [0.2, 0.3, 0.8, 0.9]  # RGBA (azul translúcido)

            ax.voxels(voxels, facecolors=colors, edgecolor='black', linewidth=0.2)
            ax.set_box_aspect((Lx, Ly, max_h))  # asegura proporción cúbica

        else:
            raise ValueError("mode debe ser 'surface' o 'voxel'")

        # Etiquetas
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("height")

        # Título dinámico
        if title is None:
            title = f"Crystal at t={t_sel:.2f}, conv={conv:.1f}%"
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")

        plt.show()

    def crystal_growth_gif(self, snapshots: List[Tuple[float, np.ndarray, float]],
        save_path: str, mode: str = "voxel", elev: int = 45, azim: int = 45,
        cmap: str = "viridis", fps: int = 10, interval_ms: int = 100, dpi: int = 150,
        title_prefix: str = "Crystal growth", repeat: bool = True, every_n: int = 1) -> None:
        """
        Genera un GIF animado del crecimiento completo del cristal usando snapshots.

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
        # Validación básica de entrada
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos para generar el GIF.")
        if every_n < 1:
            raise ValueError("every_n debe ser >= 1.")
        if mode not in ("surface", "voxel"):
            raise ValueError("mode debe ser 'surface' o 'voxel'.")
        if not save_path.lower().endswith(".gif"):
            raise ValueError("save_path debe terminar en '.gif'.")

        # Submuestreo opcional para acelerar y reducir tamaño de archivo
        frames_data = snapshots[::every_n]

        # Se infiere la geometría desde el primer snapshot
        _, h0, _ = frames_data[0]
        Lx, Ly = h0.shape

        # Escala global en Z para que no cambie entre frames (evita parpadeo)
        global_max_h = int(max(np.max(h) for _, h, _ in frames_data))
        if global_max_h < 1:
            global_max_h = 1

        # Preparación de figura y eje 3D
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Rejilla base para modo superficie
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        def _draw_common_axes() -> None:
            """Dibuja configuración común del eje para cada frame."""
            # Limpiamos el eje para redibujar el estado actual
            ax.cla()
            # Mantener cámara fija entre frames
            ax.view_init(elev=elev, azim=azim)
            # Etiquetas y límites consistentes
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("height")
            ax.set_xlim(0, max(Lx - 1, 1))
            ax.set_ylim(0, max(Ly - 1, 1))
            ax.set_zlim(0, global_max_h)

        def _update(frame_idx: int):
            """Actualiza un frame de la animación."""
            t_sel, heights, conv = frames_data[frame_idx]

            # Configuración base de ejes en cada iteración
            _draw_common_axes()

            if mode == "surface":
                # Superficie continua con escala de color estable
                ax.plot_surface(
                    X, Y, heights,
                    cmap=cmap,
                    linewidth=0,
                    antialiased=True,
                    vmin=0,
                    vmax=global_max_h
                )
            else:
                # Vóxeles discretos (cubos apilados)
                max_h_frame = int(np.max(heights))
                max_h_frame = max(max_h_frame, 1)

                voxels = np.zeros((Lx, Ly, max_h_frame), dtype=bool)
                for i in range(Lx):
                    for j in range(Ly):
                        h = int(heights[i, j])
                        if h > 0:
                            voxels[i, j, :h] = True

                # Color RGBA uniforme estilo cristal
                colors = np.zeros(voxels.shape + (4,), dtype=float)
                colors[..., :] = [0.2, 0.3, 0.8, 0.9]

                ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                # Aspect ratio para percepción geométrica más estable
                ax.set_box_aspect((Lx, Ly, global_max_h))

            # Título dinámico con tiempo y conversión
            ax.set_title(f"{title_prefix} | t={t_sel:.2f}, conv={conv:.1f}%")

            # Retornamos eje para compatibilidad con FuncAnimation
            return (ax,)

        # Construcción de la animación
        anim = FuncAnimation(
            fig,
            _update,
            frames=len(frames_data),
            interval=interval_ms,
            blit=False,
            repeat=repeat
        )

        # Guardado a GIF (writer pillow provisto por matplotlib; requiere backend disponible)
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)

        print(f"💾 GIF guardado en: {save_path}")

    def plot_conversion(self, snapshots: List[Tuple[float, np.ndarray, float]], title: str,
                        figsize: Tuple[int, int] = (8, 5), save_path: Optional[str] = None):
        """
        Grafica la conversión vs tiempo usando los snapshots.

        Parámetros:
            snapshots: lista de tuplas (t, heights, conversion_percent)
            save_path: ruta opcional para guardar la imagen
        """
        times = [t for t, _, _ in snapshots]
        conversions = [conv for _, _, conv in snapshots]

        plt.figure(figsize=figsize)
        # plt.plot(times, conversions, marker='o', linestyle='-', color='blue')
        plt.plot(times, conversions, marker='o', linestyle='-', color='blue')
        plt.xlabel("Tiempo")
        plt.ylabel("Conversión (%)")
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")
        plt.show()
