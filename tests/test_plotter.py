"""El Plotter unificado debe producir las mismas figuras que los plotters originales.

Se compara, para cada estilo, la imagen PNG de `plot_crystal_3d` (voxel y surface) y
el GIF de `crystal_growth_gif` generados por:
- original: src/plotter.py (Plotter), .descartables/.../plotter_v2.py (Plotter_v2),
            src/plotter_v3.py (Plotter_v3)
- refactorizado: src.plotting.Plotter con style="classic" / "v2" / "academic"

Se omite si el código original ya no está disponible.

Nota: el plotter_v2 original usa `plt.cm.get_cmap`, eliminado en matplotlib >= 3.9;
para poder ejecutarlo como referencia se le inyecta el equivalente
`matplotlib.colormaps.get_cmap` solo durante el test.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.testing.compare import compare_images  # noqa: E402

from tests._paths import DESCARTABLES, ORIGINAL_SRC  # noqa: E402

from src.plotting import Plotter  # noqa: E402
from tests.golden_cases import SNAP_TIMES, _sta_generic  # noqa: E402
from tests.test_golden import REFAC_API  # noqa: E402


def _load_original_plotters():
    """Importa los tres plotters originales (estilo script) o devuelve None."""
    if ORIGINAL_SRC is None or not DESCARTABLES.exists():
        return None
    paths = [ORIGINAL_SRC, DESCARTABLES]
    for p in paths:
        sys.path.insert(0, str(p))
    try:
        import plotter as p1  # noqa: E402
        import plotter_v2 as p2  # noqa: E402
        import plotter_v3 as p3  # noqa: E402
    finally:
        for p in paths:
            sys.path.remove(str(p))
    if not hasattr(plt.cm, "get_cmap"):
        # Shim solo para ejecutar el original en matplotlib moderno
        plt.cm.get_cmap = matplotlib.colormaps.get_cmap
    return {"classic": p1.Plotter, "v2": p2.Plotter_v2, "academic": p3.Plotter_v3}


class TestPlotterParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.originals = _load_original_plotters()
        if cls.originals is None:
            raise unittest.SkipTest("Plotters originales no disponibles")
        # Simulación pequeña de la línea estática con snapshots
        cls.kmc = _sta_generic(REFAC_API)
        cls.snaps, _ = cls.kmc.run(t_end=1e9, snapshot_times=SNAP_TIMES, max_events=400)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _png_pair(self, style, mode):
        """Genera la misma figura con el plotter original y con el refactorizado."""
        a = self.out / f"orig_{style}_{mode}.png"
        b = self.out / f"new_{style}_{mode}.png"
        kwargs = dict(mode=mode, snapshots=self.snaps, t_snapshot=SNAP_TIMES[-1])
        self.originals[style](self.kmc).plot_crystal_3d(save_path=str(a), **kwargs)
        plt.close("all")
        Plotter(self.kmc, style=style).plot_crystal_3d(save_path=str(b), **kwargs)
        plt.close("all")
        return a, b

    def test_crystal_3d_png_identical(self):
        cases = [("classic", "voxel"), ("classic", "surface"),
                 ("v2", "voxel"), ("v2", "surface"), ("academic", "voxel")]
        for style, mode in cases:
            with self.subTest(style=style, mode=mode):
                a, b = self._png_pair(style, mode)
                # tol=0: comparación exacta de píxeles
                self.assertIsNone(compare_images(str(a), str(b), tol=0))

    def test_gif_identical(self):
        for style, mode in [("classic", "voxel"), ("v2", "voxel"), ("v2", "surface"),
                            ("academic", "voxel")]:
            with self.subTest(style=style, mode=mode):
                a = self.out / f"orig_{style}_{mode}.gif"
                b = self.out / f"new_{style}_{mode}.gif"
                kwargs = dict(snapshots=self.snaps[:3], mode=mode, dpi=40)
                self.originals[style](self.kmc).crystal_growth_gif(save_path=str(a), **kwargs)
                Plotter(self.kmc, style=style).crystal_growth_gif(save_path=str(b), **kwargs)
                self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_academic_surface_raises(self):
        # Antes producía una figura vacía sin avisar.
        with self.assertRaises(ValueError):
            Plotter(self.kmc, style="academic").plot_crystal_3d(mode="surface")

    def test_legacy_aliases(self):
        from src.plotting import Plotter_v2, Plotter_v3
        self.assertEqual(Plotter_v2(self.kmc).style, "v2")
        self.assertEqual(Plotter_v3(self.kmc).style, "academic")
        self.assertIsInstance(Plotter_v2(self.kmc), Plotter)


class TestPaperStyle(unittest.TestCase):
    """Estilo "paper" (sin original de referencia): clasificación de capas y humo."""

    @classmethod
    def setUpClass(cls):
        cls.kmc = _sta_generic(REFAC_API)
        cls.snaps, _ = cls.kmc.run(t_end=1e9, snapshot_times=SNAP_TIMES, max_events=400)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_layer_classification(self):
        import numpy as np
        from src.plotting.plotter import _paper_voxels

        # Capa completa en h=1; una columna sube a 2 y otra a 3
        heights = np.array([[1, 2], [3, 1]])
        filled, face, _, h_full = _paper_voxels(heights, base_layers=1)
        self.assertEqual(h_full, 1)
        # Ventana: niveles z = 0, 1, 2 (una capa azul bajo el frente + 2 por encima)
        self.assertEqual(filled.shape, (2, 2, 3))
        np.testing.assert_array_equal(filled.sum(axis=2), heights)  # h_full-1 = 0 -> ventana en z=0
        blue = np.all(face[..., :3] == matplotlib.colors.to_rgb("#2B8A96"), axis=-1)
        red = np.all(face[..., :3] == matplotlib.colors.to_rgb("#8B1212"), axis=-1)
        # Solo el nivel z=0 (lleno en todas las columnas) es azul; lo de encima, rojo
        np.testing.assert_array_equal(blue, filled & (np.arange(3) < 1))
        np.testing.assert_array_equal(red, filled & (np.arange(3) >= 1))

    def test_flat_surface_has_blue_base(self):
        import numpy as np
        from src.plotting.plotter import _paper_voxels

        # Superficie plana en h=0: la base azul es el sustrato (z < 0)
        filled, face, _, h_full = _paper_voxels(np.zeros((3, 3), dtype=int), base_layers=2)
        self.assertEqual(h_full, 0)
        self.assertEqual(filled.shape, (3, 3, 2))
        self.assertTrue(filled.all())
        with self.assertRaises(ValueError):
            _paper_voxels(np.zeros((3, 3), dtype=int), base_layers=0)

    def test_png_gif_and_sequence(self):
        plotter = Plotter(self.kmc, style="paper")
        png = self.out / "paper.png"
        plotter.plot_crystal_3d(snapshots=self.snaps, save_path=str(png))
        plt.close("all")
        gif = self.out / "paper.gif"
        plotter.crystal_growth_gif(snapshots=self.snaps[:3], save_path=str(gif), dpi=40)
        seq = self.out / "paper_seq.png"
        plotter.plot_morphology_sequence(self.snaps, n_panels=3, save_path=str(seq))
        plt.close("all")
        for f in (png, gif, seq):
            self.assertGreater(f.stat().st_size, 0)
        with self.assertRaises(ValueError):
            plotter.plot_crystal_3d(mode="surface")


if __name__ == "__main__":
    unittest.main()
