"""Geometría de la red ESTÁTICA (NUEVO: antes no había tests de esta línea)."""

import pickle
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.static import LatticeSOSStatic  # noqa: E402


class TestLatticeSOSStaticGeometry(unittest.TestCase):

    def setUp(self):
        self.L = 10
        self.lat = LatticeSOSStatic(size=self.L, seed=42)

    def test_height_limits(self):
        site = (5, 5)
        self.lat.heights[site] = 0
        self.lat.dec_height(site)
        self.assertEqual(self.lat.get_height(site), 0)

    def test_pbc_and_directions(self):
        self.assertIn((0, 0), self.lat.neighbors4((0, self.L - 1)))
        self.assertIn((0, 0), self.lat.neighbors4((self.L - 1, 0)))
        # 0=N, 1=S, 2=W, 3=E
        self.assertEqual(self.lat.neighbor_in_direction((0, 0), 0), (self.L - 1, 0))
        self.assertEqual(self.lat.neighbor_in_direction((0, 0), 3), (0, 1))

    def test_directional_bond_counts(self):
        # Vecinos en x (cambia la fila i) e y (cambia la columna j) se cuentan aparte
        c = (5, 5)
        self.lat.heights[:] = 0
        self.lat.heights[4, 5] = 1  # vecino x
        self.lat.heights[6, 5] = 1  # vecino x
        self.lat.heights[5, 4] = 1  # vecino y
        self.assertEqual(self.lat.adsorption_bonds_xy(c), (2, 1))
        self.assertEqual(self.lat.adsorption_bonds(c), 3)

    def test_static_lattice_counts_only_equal_height_neighbors(self):
        """DOCUMENTA la convención actual (auditoria.md §5.1), no la valida físicamente.

        La red estática cuenta SOLO vecinos con altura exactamente h0+1. Un sitio en
        el fondo de un pozo de profundidad 2 tiene 0 enlaces de adsorción (en SOS
        estándar serían 4). Si alguien cambia la convención, este test falla.
        """
        c = (5, 5)
        self.lat.heights[:] = 2
        self.lat.heights[c] = 0
        self.assertEqual(self.lat.adsorption_bonds(c), 0)
        self.lat.heights[:] = 1
        self.lat.heights[c] = 0
        self.assertEqual(self.lat.adsorption_bonds(c), 4)

    def test_migration_targets_strictly_lower(self):
        c = (5, 5)
        self.lat.heights[:] = 0
        self.lat.heights[c] = 2
        self.lat.heights[4, 5] = 2  # igual: NO es destino en la red estática
        self.lat.heights[6, 5] = 3  # más alto: no es destino
        targets = self.lat.migration_targets(c)
        self.assertNotIn((4, 5), targets)
        self.assertNotIn((6, 5), targets)
        self.assertIn((5, 4), targets)

    def test_init_modes(self):
        self.lat.initialize(mode="seeds", n_seeds=13)
        self.assertEqual(int(self.lat.heights.sum()), 13)
        self.lat.initialize(mode="random", max_height=1)
        self.assertTrue(set(self.lat.heights.ravel()) <= {0, 1})
        self.lat.initialize(mode="screw", screw_pitch=3)
        self.assertEqual(int(self.lat.heights.min()), 0)
        with self.assertRaises(ValueError):
            self.lat.initialize(mode="desconocido")

    def test_pickle_roundtrip(self):
        # Antes fallaba (_LatticeSize.__new__ no aceptaba la tupla que pasa pickle)
        clone = pickle.loads(pickle.dumps(self.lat))
        self.assertEqual((clone.nx, clone.ny), (self.L, self.L))
        self.assertEqual(clone.size ** 2, self.L ** 2)


if __name__ == "__main__":
    unittest.main()
