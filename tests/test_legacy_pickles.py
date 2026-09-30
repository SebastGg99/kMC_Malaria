"""Los kmc.pkl de `results/` (generados con bkl_v4/lattice_v3/params_v4 en estilo
script) deben cargar con el código refactorizado vía `load_legacy_pickle`.

Comprobaciones por corrida:
- el objeto es un KMC_BKL_Static con red LatticeSOSStatic y params KMCParamsStatic
- las alturas del objeto coinciden con el último snapshot de snaps.pkl
- conversion_percent coincide con el valor final guardado en stats.pkl
- los atributos nuevos (__setstate__) tienen sus valores de compatibilidad

Variable de entorno opcional KMC_LEGACY_MAX=N para limitar el número de archivos
(cada kmc.pkl ocupa ~23 MB).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

from tests._paths import RESULTS_DIR  # noqa: E402  (añade la carpeta de src/ al path)
from src.common.io import load_pickle  # noqa: E402
from src.static import KMC_BKL_Static, KMCParamsStatic, LatticeSOSStatic  # noqa: E402



class TestLegacyPickles(unittest.TestCase):

    def test_results_kmc_pickles_load(self):
        files = sorted(RESULTS_DIR.glob("outputs_*/*/kmc.pkl"))
        if not files:
            self.skipTest(f"No hay kmc.pkl en {RESULTS_DIR}")
        limit = int(os.environ.get("KMC_LEGACY_MAX", "0"))
        if limit > 0:
            files = files[:limit]

        for path in files:
            run_dir = path.parent
            with self.subTest(run=f"{run_dir.parent.name}/{run_dir.name}"):
                kmc = load_pickle(path)
                snaps = load_pickle(run_dir / "snaps.pkl")
                stats = load_pickle(run_dir / "stats.pkl")

                self.assertIsInstance(kmc, KMC_BKL_Static)
                self.assertIsInstance(kmc.lat, LatticeSOSStatic)
                self.assertIsInstance(kmc.p, KMCParamsStatic)
                np.testing.assert_array_equal(kmc.lat.heights, snaps[-1][1])
                self.assertEqual(kmc.conversion_percent, stats["conversion_percent"])
                # Atributos rellenados por __setstate__ (no existían en bkl_v4)
                self.assertTrue(kmc.record_adsorption_probs)
                self.assertIsNone(kmc.N_seed0)
                self.assertTrue(np.isnan(kmc.crystal_fraction_percent))
                # Los métodos del motor nuevo funcionan sobre el objeto cargado
                self.assertGreaterEqual(kmc.crystal_mass(), 0.0)


if __name__ == "__main__":
    unittest.main()
