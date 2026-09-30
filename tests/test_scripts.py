"""Pruebas de humo y de paridad de los scripts unificados (scripts/*.py).

- run_sigma_scan.py produce los mismos datos que el flujo de anisotropicRun.py
  ejecutado con los módulos ORIGINALES (se omite si ya no están disponibles).
- rebuild_from_metadata reconstruye una corrida nueva y la reproduce.
- run_single.py funciona con configs estáticos (un tamaño y varios) y dinámico.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tests._paths import ORIGINAL_SRC, PKG_ROOT as ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))

import run_sigma_scan  # noqa: E402
import run_single  # noqa: E402
from src.common.io import load_json, load_pickle, rebuild_from_metadata  # noqa: E402

CONFIGS = ROOT / "scripts" / "configs"


def _run_main(module, argv):
    """Ejecuta module.main() con sys.argv simulado."""
    with mock.patch.object(sys, "argv", ["script"] + [str(a) for a in argv]):
        module.main()


class TestSigmaScan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "scan"
        _run_main(run_sigma_scan, [
            "--config", CONFIGS / "aniso_flat.json", "--times", "0:0.03:0.01",
            "--size", 10, 10, "--n-seeds", 7, "--fixed-sigma", 0.3, 0.4,
            "--sigma-step", 0.1, "--time-scale", 70, "--output-dir", cls.out,
        ])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_exist(self):
        runs = sorted(self.out.glob("size_*"))
        self.assertEqual(len(runs), 2)
        for r in runs:
            for f in ("conversion_history.csv", "conversion_history.npy", "snaps.pkl",
                      "stats.pkl", "kmc.pkl", "metadata.json"):
                self.assertTrue((r / f).exists(), f"{r.name}/{f}")
        self.assertTrue((self.out / "all_conversion_histories.csv").exists())
        meta = load_json(runs[0] / "metadata.json")
        for key in ("engine", "init_mode", "init_kwargs", "N_bulk0", "git_commit", "numpy_version"):
            self.assertIn(key, meta)

    def test_parity_with_original_anisotropicRun_flow(self):
        """Mismo flujo que run_single_sigma_simulation de anisotropicRun.py, pero con
        los módulos originales (bkl_v4, lattice_v4, params_v4)."""
        src = ORIGINAL_SRC
        if src is None:
            self.skipTest("Módulos originales no disponibles")
        sys.path.insert(0, str(src))
        try:
            import bkl_v4, lattice_v4, params_v4  # noqa: E401,E402
        finally:
            sys.path.remove(str(src))
        cfg = load_json(CONFIGS / "aniso_flat.json")
        times = np.arange(0, 0.03, 0.01)
        for sigma in (0.3, 0.4):
            with self.subTest(sigma=sigma):
                lat = lattice_v4.LatticeSOS_v4(size=(10, 10), seed=42)
                lat.initialize(mode="flat", max_height=1, n_seeds=7)
                params = params_v4.KMCParams_v4(**cfg["params"], fixed_sigma=sigma)
                kmc = bkl_v4.KMC_BKL_v4(lattice=lat, params=params, N_bulk0=2000, rng_seed=123,
                                        time_scale=70, n_seeds=7, constant_concentration=True)
                _, stats = kmc.run(t_end=times[-1], snapshot_times=times)
                run_dir = self.out / f"size_10x10_sigma_{sigma:.6g}_seed_123"
                ref = np.array(stats["conversion_history"], dtype=float)
                # El .npy conserva la precisión completa: igualdad exacta.
                np.testing.assert_array_equal(np.load(run_dir / "conversion_history.npy"), ref)
                # El CSV (pandas.to_csv, igual que en los scripts antiguos) escribe
                # ~15 cifras significativas: solo se compara con tolerancia.
                got = pd.read_csv(run_dir / "conversion_history.csv")
                np.testing.assert_allclose(got[["time", "conversion"]].to_numpy(), ref,
                                           rtol=1e-13, atol=0)

    def test_rebuild_from_metadata_reproduces(self):
        run_dir = sorted(self.out.glob("size_*"))[0]
        meta = load_json(run_dir / "metadata.json")
        kmc = rebuild_from_metadata(meta, verbose=False)
        times = np.array(meta["times"])
        snaps, _ = kmc.run(t_end=times[-1], snapshot_times=times)
        stored = load_pickle(run_dir / "snaps.pkl")
        for (t1, h1, c1), (t2, h2, c2) in zip(snaps, stored):
            np.testing.assert_array_equal(h1, h2)
            self.assertEqual(c1, c2)


class TestRunSingle(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_static_single_with_gif(self):
        _run_main(run_single, ["--config", CONFIGS / "single_iso.json", "--times", "0:0.02:0.01",
                               "--size", 6, 6, "--gif", "--output-dir", self.out])
        run_dir = next(self.out.glob("size_6x6_*"))
        for f in ("conversion_history.csv", "crystal_growth_prueba_run.png",
                  "crystal_growth_prueba_run.gif", "metadata.json"):
            self.assertTrue((run_dir / f).exists(), f)

    def test_static_sizes_density(self):
        _run_main(run_single, ["--config", CONFIGS / "sizes_density_seeds.json",
                               "--times", "0:0.02:0.01", "--size", 10, 10, "--n-seeds", 20,
                               "--sizes", "6x6", "8x8", "--output-dir", self.out])
        info = pd.read_csv(self.out / "runs_info.csv")
        # densidad 20/100 = 0.2 -> 7 semillas en 6x6, 13 en 8x8 (redondeo de pRun_v3)
        self.assertEqual(info["n_seeds"].tolist(), [7, 13])
        self.assertTrue((self.out / "conversion_overlay.png").exists())

    def test_dynamic_engine(self):
        _run_main(run_single, ["--config", CONFIGS / "dynamic_final_results.json",
                               "--times", "0:0.002:0.001", "--size", 6, 6, "--n-seeds", 3,
                               "--output-dir", self.out])
        run_dir = next(self.out.glob("size_6x6_*"))
        self.assertEqual(load_json(run_dir / "metadata.json")["engine"], "dynamic")
        self.assertTrue((run_dir / "conversion_history.csv").exists())


if __name__ == "__main__":
    unittest.main()
