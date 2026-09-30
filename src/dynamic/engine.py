"""Motor BKL de la línea DINÁMICA: `KMC_BKL_Dynamic`.

Origen: `src/bkl.py` (clases `KMC_BKL`, `SelectiveKMC`, `KMC_NoDesNoMig`).

Cambios respecto al original (ninguno altera la física ni el orden de consumo del
RNG; lo verifica `tests/test_golden.py` bit a bit):
- Imports relativos del paquete (antes: estilo script con fallback) y se elimina el
  `import matplotlib.pyplot` que no se usaba.
- Nombres canónicos (D3): `KMC_BKL` -> `KMC_BKL_Dynamic`, etc. Los nombres antiguos
  siguen disponibles como alias en `src/__init__.py`.
- `SelectiveKMC` y `KMC_NoDesNoMig` se unifican: `KMC_NoDesNoMig_Dynamic` es ahora
  `SelectiveKMC_Dynamic` con desorción y migración apagadas por defecto. El `step()`
  reescrito de `KMC_NoDesNoMig` se sustituye por un hook (`_process_enabled`) en el
  `step()` base que evita clasificar sitios de procesos apagados.
- Se elimina el bloque comentado `_adsorption_probabilities_3class` (queda en git).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .params import KMCParamsDynamic
from .lattice import LatticeSOSDynamic
from ..common.numerics import _safe_exp, _finite_or_zero

# C = Nbulk0 / V
# S = ln(C / C_eq)

# Nbulk0 

# =============================
# Adaptive BKL kMC with incorporation (robusto)
# =============================
class KMC_BKL_Dynamic:
    def __init__(self, lattice: LatticeSOSDynamic, params: KMCParamsDynamic,
                 N_bulk0: int, rng_seed: Optional[int] = None,
                 time_scale: float = 1.0, n_seeds: int = 0, 
                 debug: bool = False):
        self.lat = lattice
        self.p = params
        
        # [PRUEBA 4]: Garantía de Determinismo (Seed Check)
        if debug and rng_seed is None:
            raise ValueError("⛔ [DEBUG ERROR] Se requiere una semilla fija (rng_seed) para garantizar determinismo en modo debug.")

        self.rng = np.random.default_rng(rng_seed) # Instancia del generador de nums aleatorios moderno de numpy
        self.debug = debug

        # Reservas
        self.N0 = int(N_bulk0)   # total inicial para escalar adsorción y conversión
        self.N_bulk = int(N_bulk0) # Contador dinámico de materia en el bulk (reservorio)
        self.N_inc = 0

        # Estado temporal
        self.time_scale = float(time_scale) # Factor multiplicativo para el tiempo. Depende de las unidades físicas reales que se quieran simular
        self.t = 0.0 # reloj interno de la simulación. Comienza en 0.0 y avanza estocásticamente en cada paso del algoritmo BKL

        # Bookkeeping
        self.history = []  # lista que almacenará tuplas con (t, evt, site)
        self.counts = {"adsorption":0, "desorption":0, "migration":0, "incorporation":0}

        # Semillas iniciales
        for _ in range(max(0, int(n_seeds))):
            # Coordenadas aleatorias dentro de los límites de la red
            x, y = self.rng.integers(0, lattice.size, size=2)
            self.lat.inc_height((x,y), 1) # Método de LatticeSOS para incrementar la altura en el sitio (se deposita una partícula)
            self.N_inc += 1 # Contador de partículas incorporadas (convertidas) aumenta
            self.N_bulk = max(0, self.N_bulk - 1) # El reservorio se reduce por cada partícula que se deposita en la red

    # ---- Supersaturation ----
    @property
    # Calcula la sobresaturación
    def supersaturation(self) -> float:
        if getattr(self.p, "fixed_sigma", None) is not None:
            return float(np.clip(self.p.fixed_sigma, self.p.S_floor, self.p.S_ceil))
        C = self.N_bulk / max(self.p.V, 1e-12) # Concentración del reservorio (bulk)
        S = np.log((C + 1e-15) / max(self.p.C_eq, 1e-15)) # Sobresaturación logarítmica
        return float(np.clip(S, self.p.S_floor, self.p.S_ceil))

    @property
    # Calcula el porcentaje de conversión
    def conversion_percent(self) -> float:
        denom = self.N_bulk + self.N_inc
        return 100.0 * (self.N_inc / denom) if denom > 0 else 100.0


    # ---- Rate functions (con safe_exp y clamps) ----
    def r_a(self, i: int) -> float:
        fixed_sigma = getattr(self.p, "fixed_sigma", None)
        if self.N_bulk <= 0 and fixed_sigma is None:
            return 0.0
        S = self.supersaturation
        # evitar dividir por S~0: usar signo para no cambiar la física cualitativa
        eps = 1e-12 if S >= 0 else -1e-12
        arg = S + i * (self.p.delta / max(S, eps))
        base = self.p.K0_plus * _safe_exp(arg)
        # factor de reserva finita (empuja a meseta)
        if fixed_sigma is None:
            base *= (self.N_bulk / max(self.N0, 1))
        return _finite_or_zero(base)

    def r_d(self, i: int) -> float:
        arg = self.p.phi_over_kT - i * self.p.E_pb_over_kT
        val = self.p.K0_plus * _safe_exp(arg)
        return _finite_or_zero(val)

    def r_m(self, i: int) -> float:
        arg = self.p.phi_over_kT + 0.5*self.p.E_pb_over_kT - i*self.p.E_pb_over_kT
        val = self.p.K0_plus * _safe_exp(arg)
        return _finite_or_zero(val)

    def r_inc(self, i: int) -> float:
        arg = i * self.p.E_pb_over_kT
        val = self.p.K_inc_plus * _safe_exp(arg)
        return _finite_or_zero(val)

    # ---- Classify sites ----
    def _classify_adsorption_sites(self) -> Dict[int, List[Tuple[int,int]]]:
        bins = {0: [], 1: [], 2: [], 3: [], 4: []}
        for s in self.lat.get_sites():
            i = self.lat.adsorption_bonds(s)
            bins[min(max(i,0),4)].append(s)
        return bins

    def _classify_desorption_sites(self) -> Dict[int, List[Tuple[int,int]]]:
        bins = {0: [], 1: [], 2: [], 3: [], 4: []}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) > 0:
                i = self.lat.desorption_bonds(s)
                bins[min(max(i,0),4)].append(s)
        return bins

    def _classify_migration_sites(self) -> Dict[int, List[Tuple[int,int]]]:
        bins = {0: [], 1: [], 2: [], 3: []}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) > 0:
                targets = self.lat.migration_targets(s)
                if not targets:
                    continue
                i = self.lat.desorption_bonds(s)
                bins[min(max(i,0),3)].append(s)
        return bins

    def _classify_incorporation_sites(self) -> Dict[int, List[Tuple[int,int]]]:
        bins = {0: [], 1: [], 2: [], 3: [], 4: []}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) > 0:
                i = self.lat.desorption_bonds(s)
                bins[min(max(i,0),4)].append(s)
        return bins

    # ---- Event type selection ----
    def _choose_event_type(self, Wa, Wd, Wm, Wi) -> str:
        Wtot = Wa + Wd + Wm + Wi
        if not np.isfinite(Wtot) or Wtot <= 0.0:
            return "none"
        r = self.rng.random() * Wtot
        if r < Wa: return "adsorption"
        r -= Wa
        if r < Wd: return "desorption"
        r -= Wd
        if r < Wm: return "migration"
        return "incorporation"

    def _choose_class(self, weights: Dict[int, float]) -> int:
        total = sum(weights.values())
        if not np.isfinite(total) or total <= 0.0:
            return max(weights, key=weights.get)
        r = self.rng.random() * total
        cum = 0.0
        for i in sorted(weights.keys()):
            w = weights[i]
            cum += w
            if r <= cum:
                return i
        return max(weights, key=weights.get)

    def _choose_site_uniform(self, sites: List[Tuple[int,int]]) -> Tuple[int,int]:
        idx = self.rng.integers(0, len(sites))
        return sites[idx]

    # Método interno de validación exhaustiva
    def _validate_integrity(self, context_msg: str = ""):
        # [PRUEBA 3]: Sanidad de Tasas (Rate Sanity) exhaustive check
        for i in range(5):
            rates = [self.r_a(i), self.r_d(i), self.r_inc(i)]
            if i < 4: rates.append(self.r_m(i))
            
            if any(not np.isfinite(r) or r < 0 for r in rates):
                raise AssertionError(f"⛔ [RATE ERROR] Tasa inválida detectada para {i} vecinos: {rates}")

        if not np.isfinite(self.r_a(0)): raise AssertionError(f"Tasa r_a infinita o NaN. {context_msg}")
        
        total_pixels = self.lat.size[0] * self.lat.size[1]
        
        A_bins = self._classify_adsorption_sites()
        count_A = sum(len(lst) for lst in A_bins.values())
        assert count_A == total_pixels, f"Error en _classify_adsorption_sites: {count_A} != {total_pixels} ({context_msg})"

        # [PRUEBA 2]: Consistencia de Contenedores (Binning Integrity)
        D_bins = self._classify_desorption_sites()
        count_D = sum(len(lst) for lst in D_bins.values())
        empty_sites = len([s for s in self.lat.get_sites() if self.lat.get_height(s) == 0])
        
        if count_D + empty_sites != total_pixels:
             raise AssertionError(f"⛔ [BIN ERROR] Pérdida de sitios en desorción: {count_D} ocupados + {empty_sites} vacíos != {total_pixels}")

        print(f"✅ [DEBUG {self.t:.4f}] Integridad verificada: {context_msg}")

    # ---- Hook de procesos activos ----
    def _process_enabled(self, name: str) -> bool:
        """Indica si un proceso ("desorption"/"migration") está activo.

        En el motor base siempre es True. `SelectiveKMC_Dynamic` lo sobrescribe para
        no clasificar sitios de procesos apagados (su tasa ya es 0, así que el
        resultado es idéntico; solo se ahorra trabajo).
        """
        return True

    # ---- One kMC step (con defensas) ----
    def step(self) -> bool:
        if self.debug:
            self._validate_integrity("Pre-Step")

        A_bins = self._classify_adsorption_sites()
        # Clasificación selectiva: un proceso apagado no aporta sitios (tasa nula).
        D_bins = (self._classify_desorption_sites() if self._process_enabled("desorption")
                  else {0: [], 1: [], 2: [], 3: [], 4: []})
        M_bins = (self._classify_migration_sites() if self._process_enabled("migration")
                  else {0: [], 1: [], 2: [], 3: []})
        I_bins = self._classify_incorporation_sites()

        Wa = sum(len(A_bins[i]) * self.r_a(i) for i in A_bins)
        Wd = sum(len(D_bins[i]) * self.r_d(i) for i in D_bins if len(D_bins[i]) > 0)
        Wm = sum(len(M_bins[i]) * self.r_m(i) for i in M_bins if len(M_bins[i]) > 0)
        Wi = sum(len(I_bins[i]) * self.r_inc(i) for i in I_bins if len(I_bins[i]) > 0)

        # Sanitizar pesos
        Wa = _finite_or_zero(Wa); Wd = _finite_or_zero(Wd)
        Wm = _finite_or_zero(Wm); Wi = _finite_or_zero(Wi)
        Wtot = Wa + Wd + Wm + Wi
        
        # [PRUEBA 5]: Balance Detallado Instantáneo (Thermo Check)
        if self.debug:
            S = self.supersaturation
            if abs(S) < 0.05 and (Wa > 1e-20 or Wd > 1e-20):
                log_wa = np.log10(Wa + 1e-100)
                log_wd = np.log10(Wd + 1e-100)
                if abs(log_wa - log_wd) > 2.0:
                    print(f"⚠️ [THERMO WARNING] Posible violación de balance detallado a S={S:.4f}. Wa={Wa:.2e}, Wd={Wd:.2e}")

        if self.debug:
             assert Wtot >= 0, "Error físico: La tasa total Wtot es negativa."

        if not np.isfinite(Wtot) or Wtot <= 0.0:
            if self.debug:
                print(f"⚠️ Wtot inválido o cero: {Wtot}. Deteniendo simulación.")
            return False

        # tiempo
        z = max(self.rng.random(), 1e-15)
        dt = -np.log(z) / Wtot * self.time_scale
        if not np.isfinite(dt) or dt < 0:
            return False
        self.t += dt

        # evento
        etype = self._choose_event_type(Wa, Wd, Wm, Wi)
        
        if self.debug and etype == "none":
             print("⚠️ Evento seleccionado 'none' a pesar de Wtot > 0")

        if etype == "none":
            return False

        site = None
        if etype == "adsorption":
            weights = {i: (len(A_bins[i]) * self.r_a(i)) for i in A_bins if len(A_bins[i]) > 0}
            if not weights: return True
            i_sel = self._choose_class(weights); site = self._choose_site_uniform(A_bins[i_sel])
            self.lat.inc_height(site, 1)
            self.N_bulk = max(0, self.N_bulk - 1)

        elif etype == "desorption":
            weights = {i: (len(D_bins[i]) * self.r_d(i)) for i in D_bins if len(D_bins[i]) > 0}
            if not weights: return True
            i_sel = self._choose_class(weights); site = self._choose_site_uniform(D_bins[i_sel])
            if self.lat.get_height(site) > 0:
                self.lat.dec_height(site, 1)
                self.N_bulk += 1

        elif etype == "migration":
            weights = {i: (len(M_bins[i]) * self.r_m(i)) for i in M_bins if len(M_bins[i]) > 0}
            if not weights: return True
            i_sel = self._choose_class(weights); site = self._choose_site_uniform(M_bins[i_sel])
            targets = self.lat.migration_targets(site)
            if targets:
                tgt = targets[self.rng.integers(0, len(targets))]
                if self.lat.get_height(site) > 0 and self.lat.get_height(tgt) <= self.lat.get_height(site):
                    self.lat.dec_height(site, 1)
                    self.lat.inc_height(tgt, 1)

        elif etype == "incorporation":
            weights = {i: (len(I_bins[i]) * self.r_inc(i)) for i in I_bins if len(I_bins[i]) > 0}
            if not weights: return True
            i_sel = self._choose_class(weights); site = self._choose_site_uniform(I_bins[i_sel])
            self.N_inc += 1

        self.counts[etype] += 1
        self.history.append((self.t, etype, site))
        return True

    # ---- Run con cierre limpio y snapshots garantizados ----
    def run(self, t_end: float, snapshot_times: Optional[List[float]] = None, max_events: int = 2_000_000):
        snaps: List[Tuple[float, np.ndarray, float]] = []

        if snapshot_times is None:
            times_list: List[float] = []
        elif isinstance(snapshot_times, np.ndarray):
            times_list = snapshot_times.tolist()
        else:
            times_list = list(snapshot_times)
        times_list = sorted(times_list)

        next_snap_idx = 0
        n_events = 0
        try:
            while self.t < t_end and n_events < max_events:
                if self.debug and n_events % 100 == 0:
                     print(f"⏱️ t={self.t:.4e} | Events={n_events} | Conv={self.conversion_percent:.1f}%")

                progressed = self.step()
                if not progressed:
                    if self.debug: print("⏹️ Simulación detenida: step() devolvió False.")
                    break
                n_events += 1
                
                while next_snap_idx < len(times_list) and self.t >= times_list[next_snap_idx]:
                    snaps.append((times_list[next_snap_idx],
                                  self.lat.heights.copy(),
                                  self.conversion_percent))
                    next_snap_idx += 1
        except Exception as e:
            print(f"⚠️ Simulación detenida por excepción: {e}. Guardando estado parcial...")

        while next_snap_idx < len(times_list):
            snaps.append((times_list[next_snap_idx],
                          self.lat.heights.copy(),
                          self.conversion_percent))
            next_snap_idx += 1

        return snaps


# Clase para apagar/encender procesos selectivamente
class SelectiveKMC_Dynamic(KMC_BKL_Dynamic):
    """
    Versión de KMC_BKL_Dynamic que permite apagar selectivamente la desorción o la
    migración (su tasa pasa a ser exactamente cero y sus sitios no se clasifican).

    Unifica los antiguos `SelectiveKMC` (ambos procesos activos por defecto) y
    `KMC_NoDesNoMig` (ambos apagados por defecto), que eran equivalentes salvo por
    los valores por defecto.
    """
    def __init__(self, lattice, params, N_bulk0, rng_seed=None,
                 enable_desorption: bool = True, enable_migration: bool = True, **kwargs):
        # Flags de control selectivo
        self.enable_desorption = bool(enable_desorption)
        self.enable_migration = bool(enable_migration)
        # Inicialización base
        super().__init__(lattice, params, N_bulk0, rng_seed, **kwargs)

    def _process_enabled(self, name: str) -> bool:
        # Consulta el flag correspondiente al proceso
        if name == "desorption":
            return self.enable_desorption
        if name == "migration":
            return self.enable_migration
        return True

    # Sobrescribimos r_d (tasa de desorción)
    def r_d(self, i: int) -> float:
        if not self.enable_desorption:
            return 0.0
        return super().r_d(i)

    # Sobrescribimos r_m (tasa de migración)
    def r_m(self, i: int) -> float:
        if not self.enable_migration:
            return 0.0
        return super().r_m(i)


class KMC_NoDesNoMig_Dynamic(SelectiveKMC_Dynamic):
    """
    Retrocompatible con `KMC_NoDesNoMig`: por defecto desorción OFF y migración OFF.
    Se pueden reactivar con enable_desorption=True / enable_migration=True.
    """
    def __init__(self, lattice, params, N_bulk0, rng_seed=None,
                 enable_desorption: bool = False, enable_migration: bool = False, **kwargs):
        super().__init__(lattice, params, N_bulk0, rng_seed,
                         enable_desorption=enable_desorption,
                         enable_migration=enable_migration, **kwargs)
