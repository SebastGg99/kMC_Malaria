"""Motor BKL DINÁMICO (antes tests/test_bkl.py, misma lógica)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dynamic import (  # noqa: E402
    KMCParamsDynamic, LatticeSOSDynamic, KMC_BKL_Dynamic, KMC_NoDesNoMig_Dynamic,
    SelectiveKMC_Dynamic,
)


class TestKMCBKLDynamic(unittest.TestCase):
    """Consistencia de tasas, conservación de masa y actualización."""

    def setUp(self):
        self.params = KMCParamsDynamic(
            T=300, K0_plus=1e12, K_inc_plus=1e2,
            E_pb_over_kT=2.0, phi_over_kT=5.0, delta=0.0,
            V=1.0, C_eq=1e5, S_floor=-5, S_ceil=5,
        )
        self.lat = LatticeSOSDynamic(size=5, seed=123)
        self.kmc = KMC_BKL_Dynamic(self.lat, self.params, N_bulk0=1000, rng_seed=999)

    def test_supersaturation_clamping(self):
        self.kmc.N_bulk = 1e20
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_ceil)
        self.kmc.N_bulk = 0
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_floor)

    def test_event_classification_completeness(self):
        # Todos los sitios deben quedar en algún bin
        self.lat.heights[:] = 1
        total = self.lat.size ** 2
        self.assertEqual(sum(len(v) for v in self.kmc._classify_adsorption_sites().values()), total)
        self.assertEqual(sum(len(v) for v in self.kmc._classify_desorption_sites().values()), total)

    def test_mass_conservation_single_step(self):
        # Si ocurre una adsorción: +1 en el cristal y -1 en el reservorio
        self.params.E_pb_over_kT = 50.0
        h0 = np.sum(self.lat.heights)
        n0 = self.kmc.N_bulk
        ok = self.kmc.step()
        if ok and self.kmc.history and self.kmc.history[-1][1] == "adsorption":
            self.assertEqual(np.sum(self.lat.heights), h0 + 1)
            self.assertEqual(self.kmc.N_bulk, n0 - 1)

    def test_conversion_percent_definition(self):
        # Definición REAL de bkl.py: N_inc / (N_bulk + N_inc). (El test antiguo
        # comprobaba (N0 - N_bulk)/N0 y pasaba solo por coincidencia numérica.)
        self.kmc.N_bulk, self.kmc.N_inc, self.kmc.N0 = 30, 10, 100
        self.assertAlmostEqual(self.kmc.conversion_percent, 100.0 * 10 / 40)

    def test_selective_flags(self):
        sel = SelectiveKMC_Dynamic(self.lat, self.params, 100, 1, enable_desorption=False)
        self.assertEqual(sel.r_d(0), 0.0)
        self.assertGreater(sel.r_m(0), 0.0)
        nd = KMC_NoDesNoMig_Dynamic(self.lat, self.params, 100, 1)
        self.assertEqual((nd.r_d(0), nd.r_m(0)), (0.0, 0.0))
        self.assertIsInstance(nd, SelectiveKMC_Dynamic)


if __name__ == "__main__":
    unittest.main()
