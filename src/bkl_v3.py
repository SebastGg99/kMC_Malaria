import numpy as np
from typing import Dict, List, Tuple, Optional
from params_v3 import KMCParams_v3
from lattice_v3 import LatticeSOS_v3
from utils import _safe_exp, _finite_or_zero


# =============================
# Adaptive BKL kMC with shape anisotropy (x/y)
# =============================
class KMC_BKL_v3:
    def __init__(
        self,
        lattice: LatticeSOS_v3,
        params: KMCParams_v3,
        N_bulk0: int,
        rng_seed: Optional[int] = None,
        time_scale: float = 1.0,
        n_seeds: int = 0,
        debug: bool = False,
        constant_concentration: bool = True,
    ):
        """Initialize the KMC engine with optional constant concentration."""
        self.lat = lattice
        self.p = params

        if debug and rng_seed is None:
            raise ValueError("[DEBUG ERROR] A fixed seed is required in debug mode.")

        self.rng = np.random.default_rng(rng_seed)
        self.debug = debug

        # Reservoir
        self.N0 = int(N_bulk0)
        self.N_bulk = int(N_bulk0)
        self.constant_concentration = bool(constant_concentration)
        self.C_target = self.N0 / max(self.p.V, 1e-12)

        self.sigma_const = None
        if self.p.fixed_sigma is None and self.constant_concentration:
            raw_sigma = self._sigma_from_concentration(self.C_target)
            self.sigma_const = self._clip_sigma_via_S(raw_sigma)

        # Time state
        self.time_scale = float(time_scale)
        self.t = 0.0

        # Bookkeeping
        self.history: List[Tuple[float, str, Optional[Tuple[int, int]]]] = []
        self.counts = {"adsorption": 0, "desorption": 0, "migration": 0}

        # Tracking for analysis
        self.height_history: List[Tuple[float, float, float]] = []
        self.adsorption_probs_history: List[Tuple[float, float, Dict[int, float]]] = []

        self._seed_initial_surface_if_requested(n_seeds)

    def _seed_initial_surface_if_requested(self, n_seeds: int) -> None:
        """Seed a minimal population if requested and surface is flat."""
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

    @property
    def sigma(self) -> float:
        """Sigma = C/C_eq - 1 (clipped via S range)."""
        if self.p.fixed_sigma is not None:
            return self._clip_sigma_via_S(self.p.fixed_sigma)

        if self.constant_concentration:
            return float(self.sigma_const)

        C = self.N_bulk / max(self.p.V, 1e-12)
        return self._clip_sigma_via_S(self._sigma_from_concentration(C))

    @property
    def supersaturation(self) -> float:
        """S = ln(1 + sigma) used in rate equations."""
        return float(np.log1p(self.sigma))

    def r_a(self, ix: int, iy: int) -> float:
        """Adsorption rate with directional contributions (x/y)."""
        if self._reservoir_is_dynamic() and self.N_bulk <= 0:
            return 0.0

        S = self.supersaturation
        s_eps = 1e-12
        sign = 1.0 if S >= 0 else -1.0
        S_eff = sign * max(abs(S), s_eps)
        arg = S + (ix * self.p.delta_x + iy * self.p.delta_y) / S_eff
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    def r_d(self, ix: int, iy: int) -> float:
        """Desorption rate with directional bond energies."""
        arg = self.p.phi_over_kT - (ix * self.p.E_pb_over_kT_x + iy * self.p.E_pb_over_kT_y)
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    def r_m(self, ix: int, iy: int) -> float:
        """Migration rate using directional bond energies.

        The 0.5*E_pb term is approximated by the mean of x/y contributions.
        """
        e_pb_mean = 0.5 * (self.p.E_pb_over_kT_x + self.p.E_pb_over_kT_y)
        arg = self.p.phi_over_kT + 0.5 * e_pb_mean - (
            ix * self.p.E_pb_over_kT_x + iy * self.p.E_pb_over_kT_y
        )
        return _finite_or_zero(self.p.K0_plus * _safe_exp(arg))

    def _clip_bond_pair(self, ix: int, iy: int) -> Tuple[int, int]:
        """Clamp bond counts to a safe range for binning."""
        ix_c = min(max(int(ix), 0), 4)
        iy_c = min(max(int(iy), 0), 4)
        return ix_c, iy_c

    def _classify_adsorption_sites(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        bins: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for s in self.lat.get_sites():
            ix, iy = self.lat.adsorption_bonds_xy(s)
            key = self._clip_bond_pair(ix, iy)
            bins.setdefault(key, []).append(s)
        return bins

    def _classify_desorption_sites(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        bins: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) > 0:
                ix, iy = self.lat.desorption_bonds_xy(s)
                key = self._clip_bond_pair(ix, iy)
                bins.setdefault(key, []).append(s)
        return bins

    def _classify_migration_sites(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Classify occupied sites by directional coordination (ix, iy)."""
        bins: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for s in self.lat.get_sites():
            if self.lat.get_height(s) <= 0:
                continue
            ix, iy = self.lat.desorption_bonds_xy(s)
            if 0 <= ix + iy <= 3:
                key = self._clip_bond_pair(ix, iy)
                bins.setdefault(key, []).append(s)
        return bins

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

    def _choose_class_key(self, weights: Dict[Tuple[int, int], float]) -> Tuple[int, int]:
        total = sum(weights.values())
        if not np.isfinite(total) or total <= 0.0:
            return max(weights, key=weights.get)
        r = self.rng.random() * total
        cum = 0.0
        for key in weights:
            cum += weights[key]
            if r <= cum:
                return key
        return max(weights, key=weights.get)

    def _choose_site_uniform(self, sites: List[Tuple[int, int]]) -> Tuple[int, int]:
        idx = self.rng.integers(0, len(sites))
        return sites[idx]

    def _adsorption_probabilities_3class(self) -> Dict[int, float]:
        """Aggregate adsorption probabilities by total bond count (0,1,>=2)."""
        A_bins = self._classify_adsorption_sites()
        weights = {0: 0.0, 1: 0.0, 2: 0.0}

        for (ix, iy), sites in A_bins.items():
            total = min(ix + iy, 2)
            weights[total] += len(sites) * self.r_a(ix, iy)

        Wa = sum(weights.values())
        if Wa <= 0.0:
            return {0: 0.0, 1: 0.0, 2: 0.0}

        return {k: v / Wa for k, v in weights.items()}

    def _record_observables(self) -> None:
        self.height_history.append((self.t, float(np.mean(self.lat.heights)), self.sigma))
        self.adsorption_probs_history.append(
            (self.t, self.sigma, self._adsorption_probabilities_3class())
        )

    def _validate_integrity(self, context_msg: str = "") -> None:
        for ix in range(5):
            for iy in range(5):
                if ix + iy > 4:
                    continue
                rates = [self.r_a(ix, iy), self.r_d(ix, iy)]
                if ix + iy < 4:
                    rates.append(self.r_m(ix, iy))
                if any(not np.isfinite(r) or r < 0 for r in rates):
                    raise AssertionError(
                        f"[RATE ERROR] Invalid rate for bonds ({ix},{iy}): {rates}. {context_msg}"
                    )

        if self.debug:
            print(f"[DEBUG {self.t:.4f}] Integrity OK: {context_msg}")

    def step(self) -> bool:
        if self.debug:
            self._validate_integrity("Pre-Step")

        A_bins = self._classify_adsorption_sites()
        D_bins = self._classify_desorption_sites()
        M_bins = self._classify_migration_sites()

        rates_a = {key: self.r_a(*key) for key in A_bins}
        rates_d = {key: self.r_d(*key) for key in D_bins}
        rates_m = {key: self.r_m(*key) for key in M_bins}

        Wa = sum(len(A_bins[key]) * rates_a[key] for key in A_bins)
        Wd = sum(len(D_bins[key]) * rates_d[key] for key in D_bins)
        Wm = sum(len(M_bins[key]) * rates_m[key] for key in M_bins)

        Wa = _finite_or_zero(Wa)
        Wd = _finite_or_zero(Wd)
        Wm = _finite_or_zero(Wm)
        Wtot = Wa + Wd + Wm

        if not np.isfinite(Wtot) or Wtot <= 0.0:
            if self.debug:
                print(f"[DEBUG] Invalid Wtot: {Wtot}. Stopping.")
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
            weights = {key: len(A_bins[key]) * rates_a[key] for key in A_bins}
            if not weights:
                return True
            key_sel = self._choose_class_key(weights)
            site = self._choose_site_uniform(A_bins[key_sel])
            self.lat.inc_height(site, 1)
            if self._reservoir_is_dynamic():
                self.N_bulk = max(0, self.N_bulk - 1)

        elif etype == "desorption":
            weights = {key: len(D_bins[key]) * rates_d[key] for key in D_bins}
            if not weights:
                return True
            key_sel = self._choose_class_key(weights)
            site = self._choose_site_uniform(D_bins[key_sel])
            if self.lat.get_height(site) > 0:
                self.lat.dec_height(site, 1)
                if self._reservoir_is_dynamic():
                    self.N_bulk += 1

        elif etype == "migration":
            weights = {key: len(M_bins[key]) * rates_m[key] for key in M_bins}
            if not weights:
                return True
            key_sel = self._choose_class_key(weights)
            site = self._choose_site_uniform(M_bins[key_sel])
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

    def run(
        self,
        t_end: float,
        snapshot_times: Optional[List[float]] = None,
        max_events: int = 2_000_000,
    ):
        """Run simulation until t_end or max_events.

        Returns:
            snaps: list of (time, heights, placeholder)
            stats: dict with histories and counts
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
                        f"t={self.t:.4e} | Events={n_events} | "
                        f"sigma={self.sigma:.2f} | S={self.supersaturation:.2f}"
                    )

                progressed = self.step()
                if not progressed:
                    if self.debug:
                        print("Simulation stopped: step() returned False.")
                    break
                n_events += 1

                while next_snap_idx < len(times_list) and self.t >= times_list[next_snap_idx]:
                    snaps.append((times_list[next_snap_idx], self.lat.heights.copy(), 0.0))
                    next_snap_idx += 1

        except Exception as exc:
            print(f"Exception: {exc}. Saving partial state...")

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


class SelectiveKMC_v3(KMC_BKL_v3):
    """Modified version that can disable desorption or migration."""

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

    def r_d(self, ix: int, iy: int) -> float:
        if not self.enable_desorption:
            return 0.0
        return super().r_d(ix, iy)

    def r_m(self, ix: int, iy: int) -> float:
        if not self.enable_migration:
            return 0.0
        return super().r_m(ix, iy)


class KMC_NoDesNoMig_v3(KMC_BKL_v3):
    """Retro-compatible class that disables desorption and migration by default."""

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

    def r_d(self, ix: int, iy: int) -> float:
        if not self.enable_desorption:
            return 0.0
        return super().r_d(ix, iy)

    def r_m(self, ix: int, iy: int) -> float:
        if not self.enable_migration:
            return 0.0
        return super().r_m(ix, iy)
