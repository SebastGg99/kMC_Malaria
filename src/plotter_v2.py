import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
try:
    from .bkl_v2 import KMC_BKL_v2
except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
    from bkl_v2 import KMC_BKL_v2


class Plotter_v2:
    
    def __init__(self, kmc: KMC_BKL_v2):
        self.kmc = kmc

    def _current_sigma(self) -> float:
        """
        Sigma del paper: sigma = C/C_eq - 1.
        Si el motor no expone sigma explícitamente, se reconstruye desde S.
        """
        if hasattr(self.kmc, "sigma"):
            return float(self.kmc.sigma)
        if hasattr(self.kmc, "supersaturation"):
            return float(np.expm1(self.kmc.supersaturation))
        return 0.0

    def _current_S(self) -> float:
        """
        Sobresaturación logarítmica interna del motor:
        S = ln(1 + sigma)
        """
        if hasattr(self.kmc, "supersaturation"):
            return float(self.kmc.supersaturation)
        sigma = self._current_sigma()
        sigma = max(sigma, -1.0 + 1e-15)
        return float(np.log1p(sigma))

    def plot_crystal_3d(self, mode: str = "voxel", elev: int = 30, azim: int = 45,
                        cmap: str = "terrain", save_path: Optional[str] = None,
                        title: Optional[str] = None, 
                        snapshots: Optional[List[Tuple[float, np.ndarray, float]]] = None,
                        t_snapshot: Optional[float] = None):
        """
        Visualiza el cristal 3D (superficie continua o vóxeles discretos).
        
        Parámetros:
            mode: 'surface' (continuo) o 'voxel' (cubos discretos)
            elev, azim: ángulos de cámara
            cmap: colormap para modo superficie
            save_path: ruta opcional para guardar imagen
            title: título personalizado
            snapshots: lista de (t, heights, placeholder) del método run()
            t_snapshot: tiempo específico para extraer snapshot más cercano
        """
        
        # Seleccionar snapshot o estado actual
        if snapshots is not None and len(snapshots) > 0 and t_snapshot is not None:
            times = [abs(t - t_snapshot) for t, _, _ in snapshots]
            idx = int(np.argmin(times))
            t_sel, heights, _ = snapshots[idx]
            print(f"🧩 Snapshot seleccionado: t={t_sel:.3f}")
        elif snapshots is not None and len(snapshots) > 0:
            t_sel, heights, _ = snapshots[-1]
            print(f"🧩 Usando último snapshot: t={t_sel:.3f}")
        else:
            heights = self.kmc.lat.heights.copy()
            t_sel = self.kmc.t
            print(f"🧩 Usando estado actual: t={t_sel:.3f}")

        # Generar figura
        Lx, Ly = heights.shape
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        if mode == "surface":
            # Superficie continua con iluminación para relieve
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(heights.T, plt.cm.get_cmap(cmap), vert_exag=0.1)
            
            surf = ax.plot_surface(X, Y, heights, facecolors=rgb, 
                                  linewidth=0, antialiased=True, shade=False)
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(heights)
            fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Altura (layers)")

        elif mode == "voxel":
            max_h = int(np.max(heights))
            max_h = max(max_h, 1)
            
            voxels = np.zeros((Lx, Ly, max_h), dtype=bool)
            for i in range(Lx):
                for j in range(Ly):
                    h = int(heights[i, j])
                    if h > 0:
                        voxels[i, j, :h] = True

            # Colores tipo cristal con gradiente de altura
            colors = np.zeros(voxels.shape + (4,), dtype=float)
            for z in range(max_h):
                intensity = 0.3 + 0.7 * (z / max(max_h - 1, 1))
                colors[:, :, z, :] = [0.2 * intensity, 0.4 * intensity, 0.8, 0.9]

            # Resaltar la capa superior de cada columna en verde marino
            top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
            ii, jj = np.where(heights > 0)
            top_zz = heights[ii, jj].astype(int) - 1
            colors[ii, jj, top_zz, :] = top_layer_color

            ax.voxels(voxels, facecolors=colors, edgecolor='black', linewidth=0.15)
            ax.set_box_aspect((Lx, Ly, max_h * 0.5))  # Z comprimido para visibilidad

        else:
            raise ValueError("mode debe ser 'surface' o 'voxel'")

        ax.set_xlabel("X (lattice units)")
        ax.set_ylabel("Y (lattice units)")
        ax.set_zlabel("Altura (layers)")

        if title is None:
            sigma = self._current_sigma()
            S = self._current_S()
            title = f"Cristal ({self.kmc.p.T:.0f}K) | t={t_sel:.2f} | σ={sigma:.2f} | S={S:.2f}"
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight", transparent=True)
            print(f"💾 Imagen guardada en: {save_path}")

        plt.tight_layout()
        plt.show()

    def crystal_growth_gif(self, snapshots: List[Tuple[float, np.ndarray, float]],
                           save_path: str, mode: str = "surface", elev: int = 30, 
                           azim: int = 45, cmap: str = "terrain", fps: int = 10, 
                           interval_ms: int = 200, dpi: int = 150,
                           title_prefix: str = "Crecimiento cristalino", 
                           repeat: bool = True, every_n: int = 1) -> None:
        """
        Genera GIF animado del crecimiento cristalino.
        
        Parámetros:
            snapshots: lista de (t, heights, _) del método run()
            save_path: ruta de salida del GIF
            mode: 'surface' o 'voxel'
            every_n: usa 1 de cada N snapshots para reducir tamaño
        """
        if snapshots is None or len(snapshots) == 0:
            raise ValueError("Se requieren snapshots no vacíos.")
        if every_n < 1:
            raise ValueError("every_n debe ser >= 1.")
        if not save_path.lower().endswith(".gif"):
            raise ValueError("save_path debe terminar en '.gif'.")

        # Submuestreo opcional
        frames_data = snapshots[::every_n]
        _, h0, _ = frames_data[0]
        Lx, Ly = h0.shape
        global_max_h = int(max(np.max(h) for _, h, _ in frames_data))
        global_max_h = max(global_max_h, 1)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")

        def _draw_frame(frame_idx: int):
            """Dibuja un frame de la animación."""
            t_sel, heights, _ = frames_data[frame_idx]
            
            ax.cla()
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Altura")
            ax.set_xlim(0, max(Lx - 1, 1))
            ax.set_ylim(0, max(Ly - 1, 1))
            ax.set_zlim(0, global_max_h)

            if mode == "surface":
                ls = LightSource(azdeg=315, altdeg=45)
                rgb = ls.shade(heights.T, plt.cm.get_cmap(cmap), 
                              vmin=0, vmax=global_max_h, vert_exag=0.1)
                ax.plot_surface(X, Y, heights, facecolors=rgb, 
                               linewidth=0, antialiased=True, shade=False)
            else:
                max_h_frame = max(int(np.max(heights)), 1)
                voxels = np.zeros((Lx, Ly, max_h_frame), dtype=bool)
                for i in range(Lx):
                    for j in range(Ly):
                        h = int(heights[i, j])
                        if h > 0:
                            voxels[i, j, :h] = True
                
                colors = np.zeros(voxels.shape + (4,), dtype=float)
                colors[..., :] = [0.2, 0.3, 0.8, 0.85]

                # Resaltar la capa superior instantánea en cada frame
                top_layer_color = np.array([0.0, 0.35, 0.33, 0.98])
                ii, jj = np.where(heights > 0)
                top_zz = heights[ii, jj].astype(int) - 1
                colors[ii, jj, top_zz, :] = top_layer_color

                ax.voxels(voxels, facecolors=colors, edgecolor="black", linewidth=0.2)
                ax.set_box_aspect((Lx, Ly, global_max_h))

            sigma = self._current_sigma()
            S = self._current_S()
            ax.set_title(f"{title_prefix} | t={t_sel:.2f} | σ={sigma:.2f} | S={S:.2f}")
            return (ax,)

        anim = FuncAnimation(fig, _draw_frame, frames=len(frames_data),
                            interval=interval_ms, blit=False, repeat=repeat)
        anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"💾 GIF guardado en: {save_path}")

    # =========================================================================
    # NUEVOS MÉTODOS: Análisis específico para Figuras 7 y 8 del paper
    # =========================================================================

    def plot_growth_rate_analysis(self, stats: Dict, sigma: float, face: str,
                                   d_interplanar: float = 3.5e-3,
                                   time_unit_to_min: float = 1.0,
                                   save_path: Optional[str] = None):
        """
        Genera plot estilo Figura 7: Velocidad de crecimiento vs tiempo 
        con análisis de régimen.
        
        Parámetros:
            stats: diccionario retornado por run() con 'height_history'
            sigma: sigma del paper para la simulación
            face: cara cristalina ("110" o "101")
            d_interplanar: espaciado interplanar en μm (default 3.5e-3 para lisozima)
            time_unit_to_min: factor de conversión de unidades de tiempo a minutos
        """
        height_history = np.array(stats['height_history'])
        if len(height_history) == 0:
            print("⚠️ No hay datos de historia de altura")
            return

        t = height_history[:, 0]
        h = height_history[:, 1]
        sigma_actual = height_history[:, 2]

        # Calcular velocidad de crecimiento (pendiente)
        # Descartar transiente inicial (primeros 20% de datos)
        skip = max(1, int(0.2 * len(t)))
        if len(t) > skip + 10:
            coeffs = np.polyfit(t[skip:], h[skip:], 1)
            growth_rate_layers = coeffs[0]
            fit_values = np.polyval(coeffs, t[skip:])
        else:
            growth_rate_layers = (h[-1] - h[0]) / (t[-1] - t[0]) if len(t) > 1 else 0
            fit_values = h[skip:]

        # Conversión a unidades físicas (μm/min como en el paper)
        growth_rate_um_min = growth_rate_layers * d_interplanar / time_unit_to_min

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Altura vs tiempo (con ajuste lineal)
        ax1 = axes[0, 0]
        ax1.plot(t, h, 'b-', alpha=0.7, label='Simulación')
        ax1.plot(t[skip:], fit_values, 'r--', 
                linewidth=2, label=f'Ajuste lineal: v={growth_rate_layers:.3f} layers/u.t.')
        ax1.set_xlabel('Tiempo (u.t.)')
        ax1.set_ylabel('Altura promedio (layers)')
        ax1.set_title(f'Crecimiento en cara ({face}) | σ = {sigma:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Verificación de σ constante
        ax2 = axes[0, 1]
        ax2.plot(t, sigma_actual, 'g-', linewidth=1.5)
        ax2.axhline(y=sigma, color='r', linestyle='--', alpha=0.5, label='σ objetivo')
        ax2.set_xlabel('Tiempo (u.t.)')
        ax2.set_ylabel('Sigma')
        ax2.set_title('Verificación: σ constante (modo paper)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Velocidad instantánea (derivada numérica)
        ax3 = axes[1, 0]
        if len(t) > 10:
            # Derivada con suavizado (ventana móvil)
            window = max(3, len(t) // 20)
            h_smooth = np.convolve(h, np.ones(window)/window, mode='same')
            v_instant = np.gradient(h_smooth, t)
            ax3.plot(t, v_instant, 'purple', alpha=0.7, label='Velocidad instantánea')
            ax3.axhline(y=growth_rate_layers, color='r', linestyle='--', 
                       label=f'Velocidad media: {growth_rate_layers:.3f}')
        ax3.set_xlabel('Tiempo (u.t.)')
        ax3.set_ylabel('Velocidad (layers/u.t.)')
        ax3.set_title('Velocidad de crecimiento instantánea')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Resumen de parámetros y resultado
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Determinar régimen según sigma
        if sigma < 2.5:
            regimen = "SPIRAL (baja σ)"
            color_reg = 'yellow'
        elif sigma < 6.0:
            regimen = "STEP (media σ)"
            color_reg = 'orange'
        else:
            regimen = "ROUGH (alta σ)"
            color_reg = 'red'

        info_text = f"""
        PARÁMETROS DE SIMULACIÓN
        ─────────────────────────
        Cara cristalina: ({face})
        Sigma σ: {sigma:.2f}
        S = ln(1 + σ): {np.log1p(max(sigma, -1.0 + 1e-15)):.2f}
        Temperatura: {self.kmc.p.T:.1f} K
        
        PARÁMETROS DEL MODELO
        ─────────────────────
        E_pb/kT: {self.kmc.p.E_pb_over_kT:.2f}
        φ/kT: {self.kmc.p.phi_over_kT:.2f}
        δ (delta): {self.kmc.p.delta:.2f}
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
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color_reg, alpha=0.3, pad=1))

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Análisis guardado en: {save_path}")
        
        plt.show()
        
        return {
            'growth_rate_layers': growth_rate_layers,
            'growth_rate_um_min': growth_rate_um_min,
            'regimen': regimen,
            'final_height': h[-1],
            'final_roughness': np.std(self.kmc.lat.heights)
        }

    def plot_adsorption_probabilities(self, stats: Dict, sigma: float, face: str,
                                       save_path: Optional[str] = None):
        """
        Genera plot estilo Figura 8: Probabilidades de adsorción vs tiempo.
        Muestra la evolución de P(adatom), P(step), P(kink-like).
        """
        prob_hist = stats.get('adsorption_probs_history', [])
        if len(prob_hist) == 0:
            print("⚠️ No hay datos de probabilidades de adsorción")
            return

        # Extraer datos
        times = [p[0] for p in prob_hist]
        sigmas = [p[1] for p in prob_hist]
        
        # Probabilidades para cada tipo de sitio
        p_adatom = [p[2].get(0, 0) for p in prob_hist]
        p_step = [p[2].get(1, 0) for p in prob_hist]
        p_kink = [p[2].get(2, 0) for p in prob_hist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Izquierda: Probabilidades vs tiempo
        ax1.plot(times, p_adatom, 'b-', linewidth=2, label='Adatom (i=0)', marker='o', markersize=3)
        ax1.plot(times, p_step, 'r-', linewidth=2, label='Step (i=1)', marker='s', markersize=3)
        ax1.plot(times, p_kink, 'y-', linewidth=2, label='Kink-like (i≥2)', marker='^', markersize=3)
        ax1.set_xlabel('Tiempo (u.t.)')
        ax1.set_ylabel('Probabilidad relativa')
        ax1.set_title(f'Evolución de probabilidades de adsorción\nCara ({face}), σ={sigma:.2f}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Derecha: Probabilidades vs σ (comparación con Figura 8 del paper)
        ax2.plot(sigmas, p_adatom, 'b-', linewidth=2, label='Adatom (i=0)', alpha=0.7)
        ax2.plot(sigmas, p_step, 'r-', linewidth=2, label='Step (i=1)', alpha=0.7)
        ax2.plot(sigmas, p_kink, 'y-', linewidth=2, label='Kink-like (i≥2)', alpha=0.7)
        
        # Añadir líneas de referencia de la Figura 8 del paper (cualitativo)
        sigma_range = np.linspace(1, 8, 100)
        # Curvas teóricas aproximadas del paper para comparación visual
        p_kink_theory = np.exp(-sigma_range * 0.5) / (1 + np.exp(-sigma_range * 0.5))
        p_adatom_theory = 1 - p_kink_theory - 0.3
        
        ax2.plot(sigma_range, p_adatom_theory, 'b--', alpha=0.3, label='Tendencia esperada')
        ax2.plot(sigma_range, p_kink_theory, 'y--', alpha=0.3)
        
        ax2.set_xlabel('Sigma σ')
        ax2.set_ylabel('Probabilidad relativa')
        ax2.set_title('Probabilidades vs σ (Comparación Figura 8)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Probabilidades guardadas en: {save_path}")
        
        plt.show()

    def plot_surface_morphology(self, heights: Optional[np.ndarray] = None,
                                 cmap: str = "terrain", 
                                 show_histogram: bool = True,
                                 save_path: Optional[str] = None):
        """
        Visualización 2D de la morfología superficial con análisis de rugosidad.
        """
        if heights is None:
            heights = self.kmc.lat.heights

        fig = plt.figure(figsize=(14, 6) if show_histogram else (8, 7))

        # Panel 1: Mapa de alturas
        ax1 = fig.add_subplot(121 if show_histogram else 111)
        im = ax1.imshow(heights, cmap=cmap, origin='lower', 
                       interpolation='nearest')
        plt.colorbar(im, ax=ax1, label='Altura (layers)', fraction=0.046)
        ax1.set_title('Morfología Superficial 2D')
        ax1.set_xlabel('X (lattice units)')
        ax1.set_ylabel('Y (lattice units)')

        if show_histogram:
            # Panel 2: Histograma de alturas y estadísticas
            ax2 = fig.add_subplot(122)
            ax2.hist(heights.flatten(), bins=30, color='steelblue', 
                    edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(heights), color='r', linestyle='--', 
                       linewidth=2, label=f'Media: {np.mean(heights):.2f}')
            ax2.axvline(np.median(heights), color='g', linestyle='--',
                       linewidth=2, label=f'Mediana: {np.median(heights):.2f}')
            ax2.set_xlabel('Altura (layers)')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title(f'Distribución de Alturas\nσ_h = {np.std(heights):.2f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"💾 Morfología guardada en: {save_path}")
        
        plt.show()

    def plot_multi_sigma_comparison(self, results_list: List[Dict], 
                                     face: str,
                                     save_path: Optional[str] = None):
        """
        Compara múltiples simulaciones a diferentes σ (para construir Figura 7 completa).
        
        Parámetros:
            results_list: lista de dicts con 'sigma', 'growth_rate', 'stats'
        """
        if not results_list:
            print("⚠️ Lista de resultados vacía")
            return

        # Ordenar por sigma
        results_sorted = sorted(results_list, key=lambda x: x['sigma'])
        sigmas = [r['sigma'] for r in results_sorted]
        rates = [r['growth_rate_layers'] for r in results_sorted]
        rates_err = [r.get('growth_rate_std', 0) for r in results_sorted]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Velocidad vs sigma (estilo Figura 7 del paper)
        ax1.errorbar(sigmas, rates, yerr=rates_err, marker='o', markersize=8,
                    linewidth=2, capsize=5, color='darkblue', 
                    label=f'Cara ({face}) - Modelo kMC')
        
        # Colorear regiones de régimen
        ax1.axvspan(0, 2.5, alpha=0.2, color='yellow', label='Régimen Spiral')
        ax1.axvspan(2.5, 6.0, alpha=0.2, color='orange', label='Régimen Step')
        ax1.axvspan(6.0, 8.5, alpha=0.2, color='red', label='Régimen Rough')
        
        ax1.set_xlabel('Supersaturación σ', fontsize=12)
        ax1.set_ylabel('Velocidad de crecimiento (layers/u.t.)', fontsize=12)
        ax1.set_title(f'Figura 7 - Crecimiento en cara ({face})\n' + 
                     f'Lysozima HEW (Nagpal et al. 2024)', fontsize=13)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 8.5)

        # Panel 2: Curva log-log para identificar exponentes
        ax2.loglog(sigmas, rates, 'o-', color='darkgreen', linewidth=2, markersize=8)
        ax2.set_xlabel('Supersaturación σ (log)', fontsize=12)
        ax2.set_ylabel('Velocidad (log)', fontsize=12)
        ax2.set_title('Análisis de exponentes de crecimiento')
        ax2.grid(True, alpha=0.3, which='both')

        # Ajuste de potencia en régimen step (aproximado)
        step_mask = [(2.5 <= s <= 6.0) for s in sigmas]
        if sum(step_mask) > 2:
            step_sigmas = np.array([s for s, m in zip(sigmas, step_mask) if m])
            step_rates = np.array([r for r, m in zip(rates, step_mask) if m])
            log_fit = np.polyfit(np.log(step_sigmas), np.log(step_rates), 1)
            ax2.plot(step_sigmas, np.exp(log_fit[1]) * step_sigmas**log_fit[0],
                    'r--', label=f'Pendiente ≈ {log_fit[0]:.2f}')
            ax2.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Comparación guardada en: {save_path}")
        
        plt.show()
