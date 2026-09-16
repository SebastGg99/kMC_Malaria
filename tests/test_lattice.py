import unittest
import numpy as np
import sys
import os

# Ajuste de path para encontrar src
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.lattice import LatticeSOS

class TestLatticeSOSGeometry(unittest.TestCase):
    """
    Auditoría estricta de la geometría. 
    Prueba condiciones de contorno (PBC) y conteo de vecinos (Bond Counting).
    """
    def setUp(self):
        # Se ejecuta antes de cada test individual
        self.L = 10
        self.lat = LatticeSOS(size=self.L, seed=42)

    def test_height_limits(self):
        # Desorción no puede crear alturas negativas (suelo duro)
        site = (5, 5)
        self.lat.heights[site] = 0
        self.lat.dec_height(site) # Intentar bajar de 0
        self.assertEqual(self.lat.get_height(site), 0, "No se respetó el suelo h=0")

    def test_pbc_contiguity(self):
        # Verificar que la red es un toroide perfecto (condiciones periódicas)
        
        # Vecino este (derecha) de (x, L-1) debe ser (x, 0)
        vecinos_y = self.lat.neighbors4((0, self.L-1))
        self.assertIn((0, 0), vecinos_y, "Fallo en condición periódica eje Y (Este-Oeste)")
        
        # Vecino sur (abajo) de (L-1, 0) debe ser (0, 0)
        vecinos_x = self.lat.neighbors4((self.L-1, 0))
        self.assertIn((0, 0), vecinos_x, "Fallo en condición periódica eje X (Norte-Sur)")

    def test_bond_counting_manual_setup(self):
        """
        Prueba CRÍTICA: Construimos configuraciones específicas a mano y verificamos
        el conteo de enlaces.
        """
        # --- Caso 1: Pilar Solitario ---
        #   0
        # 0 2 0
        #   0
        center = (5, 5)
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2 
        
        # Test Desorción: Quitar partícula a altura h=2.
        # Vecinos tienen h=0. Necesitamos vecinos con h >= 2. Resultado esperado: 0.
        self.assertEqual(self.lat.desorption_bonds(center), 0, 
                         "Pilar solitario no debería tener enlaces laterales al desorber")

        # --- Caso 2: Agujero ---
        #   2
        # 2 0 2
        #   2
        self.lat.heights[:] = 2
        self.lat.heights[center] = 0
        
        # Test Adsorción: Poner partícula a altura h=0+1=1.
        # Vecinos tienen h=2. Necesitamos vecinos con h >= 1. Resultado esperado: 4.
        self.assertEqual(self.lat.adsorption_bonds(center), 4, 
                         "Adsorción en un hueco profundo debería tener 4 enlaces")

    def test_migration_targets_validity(self):
        """
        Verifica que una partícula solo pueda migrar hacia abajo o igual nivel, 
        NUNCA trepar hacia un nivel superior (restricción física típica SOS).
        """
        # Configuración:
        #   1
        # 1 2 3
        #   1
        center = (5, 5)
        neigh_low_eq = [(4,5), (5,4), (5,6)] # h=1
        neigh_high = (6,5) # h=3
        
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2
        for n in neigh_low_eq: self.lat.heights[n] = 1
        self.lat.heights[neigh_high] = 3 # Muro alto
        
        targets = self.lat.migration_targets(center)
        
        # Verificar restricciones
        self.assertNotIn(neigh_high, targets, "Migración ilegal hacia arriba detectada")
        for n in neigh_low_eq:
            self.assertIn(n, targets, "Debería poder migrar a sitios de menor o igual altura")

if __name__ == '__main__':
    unittest.main()