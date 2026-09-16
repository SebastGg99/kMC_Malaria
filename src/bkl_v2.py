import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from .params_v2 import KMCParams_v2
    from .lattice_v2 import LatticeSOS_v2
    from .utils import _safe_exp, _finite_or_zero
except ImportError:  # pragma: no cover - compatibility with legacy script-style imports
    from params_v2 import KMCParams_v2
    from lattice_v2 import LatticeSOS_v2
    from utils import _safe_exp, _finite_or_zero


# =============================
# Adaptive BKL kMC - Fidelidad incremental al paper
# Chemical Engineering Science 299 (2024) 120472
# =============================
class KMC_BKL_v2:
    def __init__(
        self,
        lattice: LatticeSOS_v2,
        params: KMCParams_v2,
        N_bulk0: int,
        rng_seed: Optional[int] = None,
        time_scale: float = 1.0,
        n_seeds: int = 0,
        debug: bool = False,
        constant_concentration: bool = True,
    ):
        """
        constant_concentration:
            True  -> reservorio efectivo constante
            False -> modo batch con N_bulk dinámico

        Si params.fixed_sigma está definido, domina sobre ambos modos. En ese caso
        el input estático se interpreta como sigma = C/C_eq - 1 y el motor convierte
        internamente a S = ln(1 + sigma) para las ecuaciones de tasa.
        """
        self.lat = lattice
        self.p = params

        if debug and rng_seed is None:
            raise ValueError("⛔ [DEBUG ERROR] Se requiere una semilla fija para garantizar determinismo.")

        self.rng = np.random.default_rng(rng_seed)
        self.debug = debug

        # Reservorio
        self.N0 = int(N_bulk0)
        self.N_bulk = int(N_bulk0)
        self.constant_concentration = bool(constant_concentration)
        self.C_target = self.N0 / max(self.p.V, 1e-12)

        self.sigma_const = None
        if self.p.fixed_sigma is None and self.constant_concentration:
            raw_sigma = self._sigma_from_concentration(self.C_target)
            self.sigma_const = self._clip_sigma_via_S(raw_sigma)

        # Estado temporal
        self.time_scale = float(time_scale)
        self.t = 0.0

        # Bookkeeping
        self.history: List[Tuple[float, str, Optional[Tuple[int, int]]]] = []
        self.counts = {"adsorption": 0, "desorption": 0, "migration": 0}

        # Tracking para análisis
        self.height_history: List[Tuple[float, float, float]] = []
        self.adsorption_probs_history: List[Tuple[float, float, Dict[int, float]]] = []

        self._seed_initial_surface_if_requested(n_seeds)

    def _seed_initial_surface_if_requested(self, n_seeds: int) -> None:
        """
        Si se solicita y la red está completamente plana, siembra una población
        mínima de sitios mono-capa aleatorios. Nunca sobrescribe una superficie
        ya inicializada por el usuario.
        """
        n_pick = int(max(0, n_seeds))
        if n_pick == 0:
            return
        if np.any(self.lat.heights):
            return

        total_sites = self.lat.nx * self.lat.ny
        n_pick = min(n_pick, total_sites)
        flat_indices = self.rng.choice(total_sites, size=n_pick, replace=False)
        for flat_idx in np.atleast_1d(flat_indices):
            i = int(flat_idx) // self.lat.ny
            j = int(flat_idx) % self.lat.ny
            self.lat.inc_height((i, j), 1)
            if self._reservoir_is_dynamic():
                self.N_bulk = max(0, self.N_bulk - 1)

    def _reservoir_is_dynamic(self) -> bool:
        return self.p.fixed_sigma is None and not self.constant_concentration

    def _sigma_from_concentration(self, C: float) -> float:
        return (C / max(self.p.C_eq, 1e-15)) - 1.0

    def _clip_sigma_via_S(self, sigma: float) -> float:
        if sigma <= -1.0:
            sigma = -1.0 + 1e-15
        S = np.log1p(sigma)
        S = float(np.clip(S, self.p.S_floor, self.p.S_ceil))
        return float(np.expm1(S))

    def _S_from_sigma(self, sigma: float) -> float:
        sigma_eff = self._clip_sigma_via_S(sigma)
        return float(np.log1p(sigma_eff))

    @property
    def sigma(self) -> float:
        """
        Sigma físico/adimensional del paper:
        sigma = C/C_eq - 1
        """
        if self.p.fixed_sigma is not None:
            return self._clip_sigma_via_S(self.p.fixed_sigma)

        if self.constant_concentration:
            return float(self.sigma_const)

        C = self.N_bulk / max(self.p.V, 1e-12)
        return self._clip_sigma_via_S(self._sigma_from_concentration(C))

    # ---- Supersaturation ----
    @property
    def supersaturation(self) -> float:
        """
        Devuelve la cantidad S usada en las ecuaciones de tasa:
        S = ln(1 + sigma)
        """
        # S = np.log1p(self.p.fixed_sigma)
        S = np.log1p(self.sigma)
        # return self._S_from_sigma(self.sigma)
        return S

    # # ---- Rate functions ----
    def r_a(self, i: int) -> float:
        """Tasa de adsorción adaptativa basada en la forma actual del modelo."""
        if self._reservoir_is_dynamic() and self.N_bulk <= 0:
            return 0.0

        S = self.supersaturation
        s_eps = 1e-12
        sign = 1.0 if S >= 0 else -1.0
        S_eff = sign * max(abs(S), s_eps)
        arg = S + i * (self.p.delta / S_eff)
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    def r_d(self, i: int) -> float:
        """Desorción controlada por phi_over_kT y la coordinación local."""
        arg = self.p.phi_over_kT - i * self.p.E_pb_over_kT
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    def r_m(self, i: int) -> float:
        """Migración superficial con término adicional de movilidad."""
        arg = self.p.phi_over_kT + 0.5 * self.p.E_pb_over_kT - i * self.p.E_pb_over_kT
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    # ---- Classify sites ----
    def _classify_adsorption_sites(self) -> Dict[int, List[Tuple[int, int]]]:
        bins = {0: [], 1: [], 2: [], 3: [], 4: []}
        for s in self.lat.get_sites():
            i = self.lat.adsorption_bonds(s)
            bins[min(max(i, 0), 4)].append(s)
        return bins

    def _classify_desorption_sites(self) -> Dict[int, List[Tuple[int, int]]]:
        bins = {0: [], 1: [], 2: [], 3: [], 4: []}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) > 0:
                i = self.lat.desorption_bonds(s)
                bins[min(max(i, 0), 4)].append(s)
        return bins

    def _classify_migration_sites(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Para seguir el esquema BKL del paper, la elegibilidad se define por el
        estado ocupado y la clase de coordinación i=0..3. La factibilidad del
        salto se resuelve recién después de escoger una dirección aleatoria.
        """
        bins = {0: [], 1: [], 2: [], 3: []}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) <= 0:
                continue
            i = self.lat.desorption_bonds(s)
            if 0 <= i <= 3:
                bins[i].append(s)
        return bins

    # ---- Selection helpers ----
    def _choose_event_type(self, Wa: float, Wd: float, Wm: float) -> str:
        Wtot = Wa + Wd + Wm
        if not np.isfinite(Wtot) or Wtot <= 0.0:
            return "none"
        r = self.rng.random() * Wtot
        if r < Wa:
            return "adsorption"
        r -= Wa
        if r < Wd:
            return "desorption"
        return "migration"

    def _choose_class(self, weights: Dict[int, float]) -> int:
        total = sum(weights.values())
        if not np.isfinite(total) or total <= 0.0:
            return max(weights, key=weights.get)
        r = self.rng.random() * total
        cum = 0.0
        for i in sorted(weights.keys()):
            cum += weights[i]
            if r <= cum:
                return i
        return max(weights, key=weights.get)

    def _choose_site_uniform(self, sites: List[Tuple[int, int]]) -> Tuple[int, int]:
        idx = self.rng.integers(0, len(sites))
        return sites[idx]

    def _adsorption_probabilities_3class(self) -> Dict[int, float]:
        A_bins = self._classify_adsorption_sites()
        rates_a = {i: self.r_a(i) for i in range(3)}
        #rates_a = {i: self.r_a(i) for i in range(5)}
        #Wa = sum(len(A_bins[i]) * rates_a[i] for i in range(5))
        Wa = sum(len(A_bins[i]) * rates_a[i] for i in range(3))

        if Wa <= 0.0:
            return {0: 0.0, 1: 0.0, 2: 0.0}

        #probs_raw = {i: (len(A_bins[i]) * rates_a[i]) / Wa for i in range(5)}
        probs_raw = {i: (len(A_bins[i]) * rates_a[i]) / Wa for i in range(3)}

        probs_merged = {
            0: probs_raw.get(0, 0.0),
            1: probs_raw.get(1, 0.0),
            #2: sum(probs_raw.get(i, 0.0) for i in (2, 3, 4)),
            2: probs_raw.get(2, 0.0),
        }
        total_prob = sum(probs_merged.values())
        if total_prob > 0.0:
            probs_merged = {i: p / total_prob for i, p in probs_merged.items()}
        return probs_merged

    def _record_observables(self) -> None:
        self.height_history.append((self.t, float(np.mean(self.lat.heights)), self.sigma))
        self.adsorption_probs_history.append(
            (self.t, self.sigma, self._adsorption_probabilities_3class())
        )

    def _validate_integrity(self, context_msg: str = "") -> None:
        for i in range(5):
            rates = [self.r_a(i), self.r_d(i)]
            if i < 4:
                rates.append(self.r_m(i))
            if any(not np.isfinite(r) or r < 0 for r in rates):
                raise AssertionError(f"⛔ [RATE ERROR] Tasa inválida para {i} vecinos: {rates}")

        if self.debug:
            print(f"✅ [DEBUG {self.t:.4f}] Integridad verificada: {context_msg}")

    # ---- One kMC step ----
    def step(self) -> bool:
        if self.debug:
            self._validate_integrity("Pre-Step")

        A_bins = self._classify_adsorption_sites()
        D_bins = self._classify_desorption_sites()
        M_bins = self._classify_migration_sites()

        rates_a = {i: self.r_a(i) for i in range(5)}
        rates_d = {i: self.r_d(i) for i in range(5)}
        rates_m = {i: self.r_m(i) for i in range(4)}

        Wa = sum(len(A_bins[i]) * rates_a[i] for i in range(5))
        Wd = sum(len(D_bins[i]) * rates_d[i] for i in range(5) if D_bins[i])
        Wm = sum(len(M_bins[i]) * rates_m[i] for i in range(4) if M_bins[i])

        Wa = _finite_or_zero(Wa)
        Wd = _finite_or_zero(Wd)
        Wm = _finite_or_zero(Wm)
        Wtot = Wa + Wd + Wm

        if self.debug:
            S = self.supersaturation
            if abs(S) < 0.05 and (Wa > 1e-20 or Wd > 1e-20):
                log_wa = np.log10(Wa + 1e-100)
                log_wd = np.log10(Wd + 1e-100)
                if abs(log_wa - log_wd) > 2.0:
                    print(f"⚠️ [THERMO WARNING] Desbalance detallado a S={S:.4f}. Wa={Wa:.2e}, Wd={Wd:.2e}")

        if not np.isfinite(Wtot) or Wtot <= 0.0:
            if self.debug:
                print(f"⚠️ Wtot inválido: {Wtot}. Deteniendo.")
            return False

        z = max(self.rng.random(), 1e-15)
        dt = -np.log(z) / Wtot * self.time_scale
        if not np.isfinite(dt) or dt < 0.0:
            return False
        self.t += dt

        etype = self._choose_event_type(Wa, Wd, Wm)
        if etype == "none":
            return False

        site: Optional[Tuple[int, int]] = None

        if etype == "adsorption":
            weights = {i: len(A_bins[i]) * rates_a[i] for i in range(5) if A_bins[i]}
            if not weights:
                return True
            i_sel = self._choose_class(weights)
            site = self._choose_site_uniform(A_bins[i_sel])
            self.lat.inc_height(site, 1)
            if self._reservoir_is_dynamic():
                self.N_bulk = max(0, self.N_bulk - 1)

        elif etype == "desorption":
            weights = {i: len(D_bins[i]) * rates_d[i] for i in range(5) if D_bins[i]}
            if not weights:
                return True
            i_sel = self._choose_class(weights)
            site = self._choose_site_uniform(D_bins[i_sel])
            if self.lat.get_height(site) > 0:
                self.lat.dec_height(site, 1)
                if self._reservoir_is_dynamic():
                    self.N_bulk += 1

        elif etype == "migration":
            weights = {i: len(M_bins[i]) * rates_m[i] for i in range(4) if M_bins[i]}
            if not weights:
                return True
            i_sel = self._choose_class(weights)
            site = self._choose_site_uniform(M_bins[i_sel])
            direction_idx = int(self.rng.integers(0, 4))
            tgt = self.lat.neighbor_in_direction(site, direction_idx)
            if self.lat.get_height(site) > 0 and self.lat.get_height(tgt) < self.lat.get_height(site):
                self.lat.dec_height(site, 1)
                self.lat.inc_height(tgt, 1)

        self.counts[etype] += 1
        self.history.append((self.t, etype, site))

        if len(self.history) % 1 == 0:
            self._record_observables()

        return True

    # ---- Run con tracking completo ----
    def run(
        self,
        t_end: float,
        snapshot_times: Optional[List[float]] = None,
        max_events: int = 2_000_000,
    ):
        """
        Ejecuta la simulación hasta t_end o max_events.

        Returns:
            snaps: Lista de (tiempo, alturas, placeholder)
            stats: Dict con historiales y conteos
        """
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
                    print(
                        f"⏱️ t={self.t:.4e} | Events={n_events} | "
                        f"sigma={self.sigma:.2f} | S={self.supersaturation:.2f}"
                    )

                progressed = self.step()
                if not progressed:
                    if self.debug:
                        print("⏹️ Simulación detenida: step() devolvió False.")
                    break
                n_events += 1

                while next_snap_idx < len(times_list) and self.t >= times_list[next_snap_idx]:
                    snaps.append((times_list[next_snap_idx], self.lat.heights.copy(), 0.0))
                    next_snap_idx += 1

        except Exception as e:
            print(f"⚠️ Excepción: {e}. Guardando estado parcial...")

        while next_snap_idx < len(times_list):
            snaps.append((times_list[next_snap_idx], self.lat.heights.copy(), 0.0))
            next_snap_idx += 1

        stats = {
            "height_history": self.height_history,
            "adsorption_probs_history": self.adsorption_probs_history,
            "event_counts": self.counts,
            "total_time": self.t,
            "total_events": len(self.history),
        }

        return snaps, stats


class SelectiveKMC_v2(KMC_BKL_v2):
    """
    Versión modificada que permite apagar selectivamente desorción o migración.
    """

    def __init__(
        self,
        lattice,
        params,
        N_bulk0,
        rng_seed=None,
        time_scale=1.0,
        n_seeds=0,
        debug=False,
        constant_concentration=True,
        enable_desorption=True,
        enable_migration=True,
    ):
        self.enable_desorption = enable_desorption
        self.enable_migration = enable_migration

        super().__init__(
            lattice=lattice,
            params=params,
            N_bulk0=N_bulk0,
            rng_seed=rng_seed,
            time_scale=time_scale,
            n_seeds=n_seeds,
            debug=debug,
            constant_concentration=constant_concentration,
        )

    def r_d(self, i: int) -> float:
        if not self.enable_desorption:
            return 0.0
        return super().r_d(i)

    def r_m(self, i: int) -> float:
        if not self.enable_migration:
            return 0.0
        return super().r_m(i)


class KMC_NoDesNoMig_v2(KMC_BKL_v2):
    """
    Clase retrocompatible: por defecto desactiva desorción y migración.
    """

    def __init__(
        self,
        lattice,
        params,
        N_bulk0,
        rng_seed=None,
        time_scale=1.0,
        n_seeds=0,
        debug=False,
        constant_concentration=True,
        enable_desorption: bool = False,
        enable_migration: bool = False,
    ):
        self.enable_desorption = enable_desorption
        self.enable_migration = enable_migration

        super().__init__(
            lattice=lattice,
            params=params,
            N_bulk0=N_bulk0,
            rng_seed=rng_seed,
            time_scale=time_scale,
            n_seeds=n_seeds,
            debug=debug,
            constant_concentration=constant_concentration,
        )

    def r_d(self, i: int) -> float:
        if not self.enable_desorption:
            return 0.0
        return super().r_d(i)

    def r_m(self, i: int) -> float:
        if not self.enable_migration:
            return 0.0
        return super().r_m(i)
