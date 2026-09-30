"""Motor BKL ESTÁTICO (NUEVO: antes no había tests de esta línea)."""

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.static import (  # noqa: E402
    KMCParamsStatic, LatticeSOSStatic, KMC_BKL_Static, KMC_NoDesNoMig_Static,
    SelectiveKMC_Static,
)


def make_params(**overrides):
    """Parámetros isotrópicos de isotropicRun (results/outputs_iso)."""
    base = dict(K0_plus=1.16718, K_inc_plus=0.5069325183371898,
                E_pb_over_kT_x=1.2704636027368605, E_pb_over_kT_y=1.2704636027368605,
                phi_over_kT=1.4792478012079329, delta_x=1.7789686274068774,
                delta_y=1.7789686274068774, V=1, C_eq=15.0, fixed_sigma=1.0,
                S_floor=-5.0, S_ceil=9.0)
    base.update(overrides)
    return KMCParamsStatic(**base)


class TestKMCBKLStatic(unittest.TestCase):

    def setUp(self):
        self.lat = LatticeSOSStatic(size=8, seed=42)
        self.kmc = KMC_BKL_Static(lattice=self.lat, params=make_params(), N_bulk0=2000,
                                  rng_seed=123, n_seeds=5)

    def test_sigma_and_S(self):
        # fixed_sigma es σ; S = ln(1+σ); con σ enorme se recorta a S_ceil
        self.assertAlmostEqual(self.kmc.sigma, 1.0)
        self.assertAlmostEqual(self.kmc.supersaturation, np.log(2.0))
        big = KMC_BKL_Static(lattice=LatticeSOSStatic(8), params=make_params(fixed_sigma=1e9),
                             N_bulk0=10, rng_seed=1)
        self.assertAlmostEqual(big.supersaturation, 9.0)

    def test_seeding_only_on_flat(self):
        self.assertEqual(int(self.lat.heights.sum()), 5)
        self.assertEqual(self.kmc.N_seed0, 5)
        lat = LatticeSOSStatic(size=8, seed=1)
        lat.initialize(mode="random", max_height=1)
        before = lat.heights.copy()
        KMC_BKL_Static(lattice=lat, params=make_params(), N_bulk0=10, rng_seed=1, n_seeds=5)
        np.testing.assert_array_equal(lat.heights, before)  # red no plana: no siembra

    def test_classification_completeness(self):
        total = 64
        self.assertEqual(sum(len(v) for v in self.kmc._classify_adsorption_sites().values()), total)
        self.lat.heights[:] = 1
        self.assertEqual(sum(len(v) for v in self.kmc._classify_desorption_sites().values()), total)
        self.assertEqual(self.kmc._classify_incorporation_sites(), self.kmc._classify_desorption_sites())

    def test_mass_bookkeeping_constant_concentration(self):
        # En modo σ fijo el reservorio no cambia; la masa de la red sí
        self.kmc.run(t_end=1e9, max_events=300)
        c = self.kmc.counts
        self.assertEqual(self.kmc.N_bulk, 2000)
        self.assertEqual(self.kmc.crystal_mass(), 5 + c["adsorption"] - c["desorption"])
        self.assertEqual(self.kmc.N_inc, c["incorporation"])

    def test_conversion_definitions(self):
        self.kmc.run(t_end=1e9, max_events=200)
        n_inc, n0 = self.kmc.N_inc, self.kmc.N0
        # v4 (se conserva): eventos de incorporación
        self.assertAlmostEqual(self.kmc.conversion_percent, 100.0 * n_inc / (n0 + n_inc))
        # v5 (opcional): masa en la red / (N0 + N_seed0)
        self.assertAlmostEqual(self.kmc.crystal_fraction_percent,
                               100.0 * self.kmc.crystal_mass() / (n0 + 5))

    def test_record_adsorption_probs_flag(self):
        fast = KMC_BKL_Static(lattice=LatticeSOSStatic(size=8, seed=42), params=make_params(),
                              N_bulk0=2000, rng_seed=123, n_seeds=5, record_adsorption_probs=False)
        _, stats_fast = fast.run(t_end=1e9, max_events=100)
        _, stats_full = self.kmc.run(t_end=1e9, max_events=100)
        self.assertEqual(len(stats_fast["adsorption_probs_history"]), 0)
        self.assertEqual(len(stats_full["adsorption_probs_history"]), 100)
        self.assertEqual(fast.history, self.kmc.history)

    def test_solvent_factors(self):
        p = make_params(solvent_des_factor=3.0)
        k_on = KMC_BKL_Static(lattice=LatticeSOSStatic(8), params=p, N_bulk0=10, rng_seed=1)
        k_off = KMC_BKL_Static(lattice=LatticeSOSStatic(8), params=p, N_bulk0=10, rng_seed=1,
                               use_solvent=False)
        self.assertAlmostEqual(k_on.r_d(1, 0), 3.0 * k_off.r_d(1, 0))

    def test_selective_variants(self):
        nd = KMC_NoDesNoMig_Static(LatticeSOSStatic(8), make_params(), 10, 1)
        self.assertEqual((nd.r_d(0, 0), nd.r_m(0, 0)), (0.0, 0.0))
        sel = SelectiveKMC_Static(LatticeSOSStatic(8), make_params(), 10, 1, enable_migration=False)
        self.assertGreater(sel.r_d(0, 0), 0.0)
        self.assertEqual(sel.r_m(0, 0), 0.0)

    def test_pickle_roundtrip_and_legacy_state(self):
        self.kmc.run(t_end=1e9, max_events=50)
        clone = pickle.loads(pickle.dumps(self.kmc))
        np.testing.assert_array_equal(clone.lat.heights, self.kmc.lat.heights)
        # Estado de un pickle antiguo (sin atributos nuevos) -> __setstate__ los rellena
        state = dict(self.kmc.__dict__)
        del state["record_adsorption_probs"], state["N_seed0"]
        legacy = KMC_BKL_Static.__new__(KMC_BKL_Static)
        legacy.__setstate__(state)
        self.assertTrue(legacy.record_adsorption_probs)
        self.assertTrue(np.isnan(legacy.crystal_fraction_percent))


if __name__ == "__main__":
    unittest.main()
