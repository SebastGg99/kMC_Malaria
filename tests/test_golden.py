"""Paridad bit a bit entre el código refactorizado y el original (Fase 0).

Compara cada caso de `golden_cases.CASES` ejecutado con `src.dynamic` / `src.static`
contra las referencias `golden/*.npz` generadas por `make_golden.py` con el código
original (bkl.py, bkl_v4.py, ...). La igualdad es EXACTA (sin tolerancia): cualquier
cambio en el orden de consumo del RNG o en la aritmética hace fallar el test.

Las referencias solo son válidas con la versión de numpy indicada en golden/ENV.txt.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tests._paths import ORIGINAL_SRC  # noqa: E402  (añade la carpeta de src/ al path)

from src.dynamic import (  # noqa: E402
    KMCParamsDynamic, LatticeSOSDynamic, KMC_BKL_Dynamic,
    SelectiveKMC_Dynamic, KMC_NoDesNoMig_Dynamic,
)
from src.static import (  # noqa: E402
    KMCParamsStatic, LatticeSOSStatic, KMC_BKL_Static,
    SelectiveKMC_Static, KMC_NoDesNoMig_Static,
)
from tests.golden_cases import CASES, run_case  # noqa: E402

GOLDEN_DIR = HERE / "golden"

REFAC_API = {
    "dyn_params": KMCParamsDynamic,
    "dyn_lattice": LatticeSOSDynamic,
    "dyn_engine": KMC_BKL_Dynamic,
    "dyn_selective": SelectiveKMC_Dynamic,
    "dyn_nodesnomig": KMC_NoDesNoMig_Dynamic,
    "sta_params": KMCParamsStatic,
    "sta_lattice": LatticeSOSStatic,
    "sta_engine": KMC_BKL_Static,
    "sta_selective": SelectiveKMC_Static,
    "sta_nodesnomig": KMC_NoDesNoMig_Static,
}


def _expected_numpy_version() -> str:
    """Versión de numpy con la que se generaron las referencias."""
    for line in (GOLDEN_DIR / "ENV.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("numpy=="):
            return line.split("==", 1)[1]
    return ""


class TestGoldenParity(unittest.TestCase):
    """Un subtest por caso; falla si cualquier array difiere en un solo bit."""

    @classmethod
    def setUpClass(cls):
        if not GOLDEN_DIR.exists():
            raise unittest.SkipTest("No hay referencias: ejecutar tests/make_golden.py")
        expected = _expected_numpy_version()
        if expected and expected != np.__version__:
            raise unittest.SkipTest(
                f"Referencias generadas con numpy {expected}; entorno actual {np.__version__}")

    def test_all_cases_bit_identical(self):
        for name in CASES:
            with self.subTest(case=name):
                ref = np.load(GOLDEN_DIR / f"{name}.npz")
                got = run_case(name, REFAC_API)
                self.assertEqual(set(ref.files), set(got.keys()), "Distintos observables")
                for key in ref.files:
                    np.testing.assert_array_equal(
                        got[key], ref[key], err_msg=f"{name}: '{key}' difiere")


class TestStaticV5Options(unittest.TestCase):
    """Opciones de bkl_v5 integradas en KMC_BKL_Static (decisión D2)."""

    def test_no_adsorption_probs_same_trajectory(self):
        # Con record_adsorption_probs=False la trayectoria debe coincidir con la
        # referencia de v4 y solo debe faltar adsorption_probs_history.
        api = dict(REFAC_API)
        api["sta_engine"] = lambda **kw: KMC_BKL_Static(record_adsorption_probs=False, **kw)
        ref = np.load(GOLDEN_DIR / "sta_iso_fixed_sigma.npz")
        got = run_case("sta_iso_fixed_sigma", api)
        for key in ("heights", "t", "hist_t", "hist_code", "hist_site", "counts",
                    "height_history", "conversion_history", "snap_h"):
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
        self.assertEqual(got["ads_probs"].shape[0], 0)

    def test_crystal_fraction_matches_bkl_v5(self):
        # crystal_fraction_percent debe dar lo mismo que conversion_percent de bkl_v5.
        # Se omite si el código original ya no está en <raíz>/src.
        repo_src = ORIGINAL_SRC
        if repo_src is None or not (repo_src / "bkl_v5.py").exists():
            self.skipTest("src/bkl_v5.py original no disponible")
        sys.path.insert(0, str(repo_src))
        try:
            import bkl_v5  # noqa: E402  (original, estilo script)
            import lattice_v4 as orig_lattice  # noqa: E402
            import params_v4 as orig_params  # noqa: E402
        finally:
            sys.path.remove(str(repo_src))

        from tests.golden_cases import _sta_generic
        orig_api = dict(REFAC_API, sta_engine=bkl_v5.KMC_BKL_v5,
                        sta_lattice=orig_lattice.LatticeSOS_v4,
                        sta_params=orig_params.KMCParams_v4)
        for case_kwargs in (dict(), dict(iso=False),
                            dict(params_overrides=dict(fixed_sigma=None),
                                 engine_kwargs=dict(N_bulk0=30))):
            with self.subTest(**{k: str(v) for k, v in case_kwargs.items()}):
                v5 = _sta_generic(orig_api, **case_kwargs)
                new = _sta_generic(REFAC_API, **case_kwargs)
                v5.run(t_end=1e9, max_events=800)
                new.run(t_end=1e9, max_events=800)
                self.assertEqual(new.crystal_fraction_percent, v5.conversion_percent)


if __name__ == "__main__":
    unittest.main()
