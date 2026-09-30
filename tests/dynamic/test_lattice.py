"""Geometría de la red DINÁMICA (antes tests/test_lattice.py, misma lógica)."""

import pickle
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dynamic import LatticeSOSDynamic  # noqa: E402


class TestLatticeSOSDynamicGeometry(unittest.TestCase):
    """Condiciones de contorno periódicas y conteo de enlaces."""

    def setUp(self):
        self.L = 10
        self.lat = LatticeSOSDynamic(size=self.L, seed=42)

    def test_height_limits(self):
        # La desorción no puede crear alturas negativas (suelo duro)
        site = (5, 5)
        self.lat.heights[site] = 0
        self.lat.dec_height(site)
        self.assertEqual(self.lat.get_height(site), 0, "No se respetó el suelo h=0")

    def test_pbc_contiguity(self):
        # Vecino este de (0, L-1) es (0, 0); vecino sur de (L-1, 0) es (0, 0)
        self.assertIn((0, 0), self.lat.neighbors4((0, self.L - 1)))
        self.assertIn((0, 0), self.lat.neighbors4((self.L - 1, 0)))

    def test_bond_counting_manual_setup(self):
        center = (5, 5)
        # Pilar solitario de altura 2: sin enlaces laterales al desorber
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2
        self.assertEqual(self.lat.desorption_bonds(center), 0)
        # Hueco de profundidad 2: 4 enlaces al adsorber
        self.lat.heights[:] = 2
        self.lat.heights[center] = 0
        self.assertEqual(self.lat.adsorption_bonds(center), 4)

    def test_dynamic_lattice_counts_neighbors_at_or_above_h0(self):
        """DOCUMENTA la convención actual (auditoria.md §5.1), no la valida físicamente.

        La red dinámica cuenta vecinos con h >= h0 para la adsorción, así que en una
        superficie plana TODOS los sitios tienen 4 enlaces de adsorción. En SOS
        estándar (h >= h0+1) serían 0. Si alguien cambia la convención, este test
        falla y obliga a hacerlo a propósito.
        """
        self.lat.heights[:] = 0
        self.assertEqual(self.lat.adsorption_bonds((5, 5)), 4)

    def test_migration_targets_validity(self):
        # Solo se migra a vecinos de altura menor o IGUAL; nunca hacia arriba
        center = (5, 5)
        neigh_low_eq = [(4, 5), (5, 4), (5, 6)]
        neigh_high = (6, 5)
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2
        for n in neigh_low_eq:
            self.lat.heights[n] = 1
        self.lat.heights[neigh_high] = 3
        targets = self.lat.migration_targets(center)
        self.assertNotIn(neigh_high, targets)
        for n in neigh_low_eq:
            self.assertIn(n, targets)

    def test_pickle_roundtrip(self):
        # Antes fallaba (_LatticeSize.__new__ no aceptaba la tupla que pasa pickle)
        self.lat.heights[3, 3] = 2
        clone = pickle.loads(pickle.dumps(self.lat))
        self.assertEqual(tuple(clone.size), (self.L, self.L))
        self.assertEqual(clone.size ** 2, self.L ** 2)
        self.assertTrue((clone.heights == self.lat.heights).all())


if __name__ == "__main__":
    unittest.main()
