"""Las 38 corridas de results/ se reproducen con el código refactorizado.

Para cada corrida de scripts/configs/results/*.json se ejecutan los primeros N
eventos (variable de entorno KMC_REPRO_EVENTS, por defecto 150) y se comparan con el
historial guardado en su kmc.pkl: tipo de evento y sitio exactos, tiempo con
tolerancia relativa 1e-12.

Esto valida a la vez: el motor refactorizado, el cargador de pickles antiguos y la
deducción del modo de inicio efectivo (build_results_configs.py).
"""

import os
import sys
import unittest
from pathlib import Path

from tests._paths import PKG_ROOT, RESULTS_DIR  # noqa: E402
sys.path.insert(0, str(PKG_ROOT / "scripts"))

from reproduce_results import CONFIG_DIR, check_run  # noqa: E402
from src.common.io import load_json, load_pickle  # noqa: E402



class TestResultsReproduction(unittest.TestCase):

    def test_first_events_match_stored_history(self):
        specs = sorted(CONFIG_DIR.glob("outputs_*.json"))
        if not specs or not RESULTS_DIR.exists():
            self.skipTest("No hay configs de results/ o no existe la carpeta results/")
        n_events = int(os.environ.get("KMC_REPRO_EVENTS", "150"))
        for spec_path in specs:
            spec = load_json(spec_path)
            for run in spec["runs"]:
                with self.subTest(run=f"{spec['folder']}/{run['run_id']}"):
                    kmc_pkl = RESULTS_DIR / spec["folder"] / run["run_id"] / "kmc.pkl"
                    stored = load_pickle(kmc_pkl)
                    self.assertEqual(check_run(run, stored.history, n_events), "OK")


if __name__ == "__main__":
    unittest.main()
