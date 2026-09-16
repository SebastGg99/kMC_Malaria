import numpy as np
from typing import Dict, List, Tuple, Optional

from params_v4 import KMCParams_v4
from lattice_v4 import LatticeSOS_v4
from utils import _safe_exp, _finite_or_zero


# =============================
# Adaptive BKL kMC with anisotropy and solvent factors
# =============================
class KMC_BKL_v5:
    """Adaptive BKL engine with anisotropy and simplified solvent scaling."""

    def __init__(
        self,
        lattice: LatticeSOS_v4,
        params: KMCParams_v4,
        N_bulk0: int,
        rng_seed: Optional[int] = None,
        time_scale: float = 1.0,
        n_seeds: int = 0,
        debug: bool = False,
        constant_concentration: bool = True,
        use_solvent: bool = True,
        record_adsorption_probs: bool = False,
    ):
        """Initialize the KMC engine with optional constant concentration."""
        self.lat = lattice
        self.p = params

        if debug and rng_seed is None:
            raise ValueError("[DEBUG ERROR] A fixed seed is required in debug mode.")

        self.rng = np.random.default_rng(rng_seed)
        self.debug = debug
        self.use_solvent = bool(use_solvent)

        # Reservoir
        self.N0 = int(N_bulk0)
        self.N_bulk = int(N_bulk0)

        # Incorporation bookkeeping (converted particles).
        self.N_inc = 0
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
        self.counts = {
            "adsorption": 0,
            "desorption": 0,
            "migration": 0,
            "incorporation": 0,
        }

        # Tracking for analysis
        self.height_history: List[Tuple[float, float, float]] = []
        self.adsorption_probs_history: List[Tuple[float, float, Dict[int, float]]] = []
        self.conversion_history: List[Tuple[float, float]] = []

        # Recording control
        self.record_adsorption_probs = bool(record_adsorption_probs)

        # Internal counters for fast observables
        self._step_count = 0

        self._seed_initial_surface_if_requested(n_seeds)

        # Reference amount of crystalline material already present in the lattice
        # right after initialization/seeding.
        self.N_seed0 = int(np.sum(self.lat.heights))
        self.N_total0 = self.N0 + self.N_seed0

        # Cached crystalline mass, updated incrementally.
        self._crystal_mass = float(self.N_seed0)
        self._n_sites = int(self.lat.nx * self.lat.ny)

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

    def _conversion_reservoir(self) -> float:
        # Kept for backward compatibility.
        if self.constant_concentration or self.p.fixed_sigma is not None:
            return float(self.N0)
        return float(self.N_bulk)

    def crystal_mass(self) -> float:
        """Fast O(1) crystalline mass/occupancy stored in the lattice."""
        return float(self._crystal_mass)

    @property
    def conversion_percent(self) -> float:
        """
        Percent conversion measured from the crystalline mass in the lattice.
        This avoids a full scan of the lattice at every event.
        """
        denom = float(self.N_total0)
        if denom <= 0.0:
            return 0.0
        return 100.0 * self.crystal_mass() / denom

    def _solvent_factor(self, name: str) -> float:
        if not self.use_solvent:
            return 1.0
        return float(getattr(self.p, name, 1.0))

    def _apply_solvent_factor(self, base_rate: float, factor_name: str) -> float:
        scaled = base_rate * self._solvent_factor(factor_name)
        if not np.isfinite(scaled) or scaled < 0.0:
            return 0.0
        return float(scaled)

    def r_a(self, ix: int, iy: int) -> float:
        """Adsorption rate with directional contributions (x/y)."""
        if self._reservoir_is_dynamic() and self.N_bulk <= 0:
            return 0.0

        S = self.supersaturation
        s_eps = 1e-12
        sign = 1.0 if S >= 0 else -1.0
        S_eff = sign * max(abs(S), s_eps)
        arg = S + (ix * self.p.delta_x + iy * self.p.delta_y) / S_eff
        rate = _finite_or_zero(self.p.K0_plus * _safe_exp(arg))
        return self._apply_solvent_factor(rate, "solvent_ads_factor")

    def r_d(self, ix: int, iy: int) -> float:
        """Desorption rate with directional bond energies."""
        arg = self.p.phi_over_kT - (ix * self.p.E_pb_over_kT_x + iy * self.p.E_pb_over_kT_y)
        rate = _finite_or_zero(self.p.K0_plus * _safe_exp(arg))
        return self._apply_solvent_factor(rate, "solvent_des_factor")

    def r_m(self, ix: int, iy: int) -> float:
        """Migration rate using directional bond energies."""
        e_pb_mean = 0.5 * (self.p.E_pb_over_kT_x + self.p.E_pb_over_kT_y)
        arg = self.p.phi_over_kT + 0.5 * e_pb_mean - (
            ix * self.p.E_pb_over_kT_x + iy * self.p.E_pb_over_kT_y
        )
        rate = _finite_or_zero(self.p.K0_plus * _safe_exp(arg))
        return self._apply_solvent_factor(rate, "solvent_mig_factor")

    def r_inc(self, ix: int, iy: int) -> float:
        """Incorporation rate using directional bond energies."""
        arg = ix * self.p.E_pb_over_kT_x + iy * self.p.E_pb_over_kT_y
        rate = _finite_or_zero(self.p.K_inc_plus * _safe_exp(arg))
        return self._apply_solvent_factor(rate, "solvent_inc_factor")

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

    def _classify_incorporation_sites(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Classify occupied sites for incorporation (same bins as desorption)."""
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

    def _choose_event_type(self, Wa: float, Wd: float, Wm: float, Wi: float) -> str:
        Wtot = Wa + Wd + Wm + Wi
        if not np.isfinite(Wtot) or Wtot <= 0.0:
            return "none"
        r = self.rng.random() * Wtot
        if r < Wa:
            return "adsorption"
        r -= Wa
        if r < Wd:
            return "desorption"
        r -= Wd
        if r < Wm:
            return "migration"
        return "incorporation"

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
        # O(1) observables.
        mean_height = self._crystal_mass / max(self._n_sites, 1)
        self.height_history.append((self.t, float(mean_height), self.sigma))

        # This is the expensive one; keep it optional.
        if self.record_adsorption_probs:
            self.adsorption_probs_history.append(
                (self.t, self.sigma, self._adsorption_probabilities_3class())
            )

        self.conversion_history.append((self.t, self.conversion_percent))

    def _validate_integrity(self, context_msg: str = "") -> None:
        for ix in range(5):
            for iy in range(5):
                if ix + iy > 4:
                    continue
                rates = [self.r_a(ix, iy), self.r_d(ix, iy), self.r_inc(ix, iy)]
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
        I_bins = self._classify_incorporation_sites()

        rates_a = {key: self.r_a(*key) for key in A_bins}
        rates_d = {key: self.r_d(*key) for key in D_bins}
        rates_m = {key: self.r_m(*key) for key in M_bins}
        rates_i = {key: self.r_inc(*key) for key in I_bins}

        Wa = sum(len(A_bins[key]) * rates_a[key] for key in A_bins)
        Wd = sum(len(D_bins[key]) * rates_d[key] for key in D_bins)
        Wm = sum(len(M_bins[key]) * rates_m[key] for key in M_bins)
        Wi = sum(len(I_bins[key]) * rates_i[key] for key in I_bins)

        Wa = _finite_or_zero(Wa)
        Wd = _finite_or_zero(Wd)
        Wm = _finite_or_zero(Wm)
        Wi = _finite_or_zero(Wi)
        Wtot = Wa + Wd + Wm + Wi

        if not np.isfinite(Wtot) or Wtot <= 0.0:
            if self.debug:
                print(f"[DEBUG] Invalid Wtot: {Wtot}. Stopping.")
            return False

        z = max(self.rng.random(), 1e-15)
        dt = -np.log(z) / Wtot * self.time_scale
        if not np.isfinite(dt) or dt < 0.0:
            return False
        self.t += dt

        etype = self._choose_event_type(Wa, Wd, Wm, Wi)
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

            # Fast incremental bookkeeping for crystalline mass.
            self._crystal_mass += 1.0

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

                # Fast incremental bookkeeping for crystalline mass.
                self._crystal_mass = max(0.0, self._crystal_mass - 1.0)

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

                # Migration conserves total crystalline mass.
                # No update to self._crystal_mass required.

        elif etype == "incorporation":
            weights = {key: len(I_bins[key]) * rates_i[key] for key in I_bins}
            if not weights:
                return True
            key_sel = self._choose_class_key(weights)
            site = self._choose_site_uniform(I_bins[key_sel])
            # Incorporation only updates chemical bookkeeping.
            self.N_inc += 1

        self.counts[etype] += 1
        self.history.append((self.t, etype, site))

        self._step_count += 1
        self._record_observables()
        return True

    def run(
        self,
        t_end: float,
        snapshot_times: Optional[List[float]] = None,
        max_events: int = 2_000_000,
    ):
        """Run simulation until t_end or max_events."""
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
                    snaps.append(
                        (
                            times_list[next_snap_idx],
                            self.lat.heights.copy(),
                            self.conversion_percent,
                        )
                    )
                    next_snap_idx += 1

        except Exception as exc:
            print(f"Exception: {exc}. Saving partial state...")

        while next_snap_idx < len(times_list):
            snaps.append(
                (
                    times_list[next_snap_idx],
                    self.lat.heights.copy(),
                    self.conversion_percent,
                )
            )
            next_snap_idx += 1

        stats = {
            "height_history": self.height_history,
            "adsorption_probs_history": self.adsorption_probs_history,
            "event_counts": self.counts,
            "incorporated_count": self.N_inc,
            "conversion_percent": self.conversion_percent,
            "conversion_history": self.conversion_history,
            "total_time": self.t,
            "total_events": len(self.history),
        }

        return snaps, stats