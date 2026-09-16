import unittest
import numpy as np
import sys
import os
# Ajuste de path para importar desde src independientemente de dónde se ejecute el test
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(r'c:/Users/sebas/MalariaProject')

from src import KMCParams, LatticeSOS, KMC_BKL, _safe_exp, _finite_or_zero

class TestNumerics(unittest.TestCase):
    """
    Control de calidad de las funciones numéricas de bajo nivel en utils.py.
    Fundamental para evitar NaNs y Overflows que arruinen simulaciones largas.
    """
    def test_safe_exp_limits(self):
        # Caso normal
        self.assertAlmostEqual(_safe_exp(1.0), np.exp(1.0))
        # Caso Overflow positivo (debe clippear, no lanzar error)
        self.assertLess(_safe_exp(800), np.inf)
        self.assertEqual(_safe_exp(800), _safe_exp(1000)) # Deben saturar igual
        # Caso Underflow (debe ser 0)
        self.assertAlmostEqual(_safe_exp(-800), 0.0)

    def test_finite_or_zero_robustness(self):
        self.assertEqual(_finite_or_zero(np.inf), 0.0)
        self.assertEqual(_finite_or_zero(np.nan), 0.0)
        self.assertEqual(_finite_or_zero(-np.inf), 0.0)
        self.assertEqual(_finite_or_zero(5.5), 5.5)

class TestLatticeSOSGeometry(unittest.TestCase):
    """
    Auditoría estricta de la geometría. Si esto falla, toda la física es incorrecta.
    Prueba condiciones de contorno (PBC) y conteo de vecinos (Bond Counting).
    """
    def setUp(self):
        self.L = 10
        self.lat = LatticeSOS(size=self.L, seed=42)

    def test_height_limits(self):
        # Desorción no puede crear alturas negativas
        site = (5, 5)
        self.lat.heights[site] = 0
        self.lat.dec_height(site)
        self.assertEqual(self.lat.get_height(site), 0, "No se respetó el suelo h=0")

    def test_pbc_contiguity(self):
        # Verificar que la red es un toroide perfecto
        # Vecino este de (x, L-1) debe ser (x, 0)
        vecinos = self.lat.neighbors4((0, self.L-1))
        self.assertIn((0, 0), vecinos, "Fallo en condición periódica eje Y")
        # Vecino sur de (L-1, 0) debe ser (0, 0)
        vecinos = self.lat.neighbors4((self.L-1, 0))
        self.assertIn((0, 0), vecinos, "Fallo en condición periódica eje X")

    def test_bond_counting_manual_setup(self):
        """
        Prueba CRÍTICA: Construimos configuraciones específicas a mano y verificamos
        el conteo de enlaces.
        """
        # Configuración "Pilar Solitario"
        #   0
        # 0 2 0
        #   0
        center = (5, 5)
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2 
        
        # Test Desorción (quitar partícula a altura h=2)
        # Vecinos tienen h=0. Nivel a evaluar = 2.
        # Vecinos con h>=2? R: 0.
        self.assertEqual(self.lat.desorption_bonds(center), 0, "Pilar solitario no debería tener enlaces laterales al desorber")

        # Configuración "Agujero"
        #   2
        # 2 0 2
        #   2
        self.lat.heights[:] = 2
        self.lat.heights[center] = 0
        
        # Test Adsorción (poner partícula a altura h=0+1=1)
        # Vecinos tienen h=2. Nivel a evaluar = 1.
        # Vecinos con h>=1? R: 4 de 4.
        self.assertEqual(self.lat.adsorption_bonds(center), 4, "Adsorción en agujero debería tener 4 enlaces")

    def test_migration_targets_validity(self):
        """
        Verifica que una partícula solo pueda migrar hacia abajo o lateralmente, NUNCA hacia arriba.
        """
        #   1
        # 1 2 3
        #   1
        center = (5, 5)
        neigh_low_eq = [(4,5), (5,4), (5,6)] # h=1
        neigh_high = (6,5) # h=3
        
        self.lat.heights[:] = 0
        self.lat.heights[center] = 2
        for n in neigh_low_eq: self.lat.heights[n] = 1
        self.lat.heights[neigh_high] = 3
        
        targets = self.lat.migration_targets(center)
        
        # Debe poder ir a los sitios de h=1, pero NO al de h=3
        self.assertNotIn(neigh_high, targets, "Migración ilegal hacia arriba detectada")
        for n in neigh_low_eq:
            self.assertIn(n, targets)

class TestKMCLogic(unittest.TestCase):
    """
    Pruebas del motor BKL. Verifica la consistencia de tasas, conservación de masa
    y lógica de actualización.
    """
    def setUp(self):
        # Parámetros controlados
        self.params = KMCParams(
            T=300, K0_plus=1e12, K_inc_plus=1e2,
            E_pb_over_kT=2.0, phi_over_kT=5.0, delta=0.0,
            V=1.0, C_eq=1e5, S_floor=-5, S_ceil=5
        )
        self.lat = LatticeSOS(size=5, seed=123)
        self.N_bulk_init = 1000
        # Inicializar KMC con semilla fija para determinismo en tests
        self.kmc = KMC_BKL(self.lat, self.params, N_bulk0=self.N_bulk_init, rng_seed=999)

    def test_supersaturation_clamping(self):
        # Forzar N_bulk muy alto -> S debería ser S_ceil
        self.kmc.N_bulk = 1e20
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_ceil)
        
        # Forzar N_bulk 0 -> S debería ser S_floor
        self.kmc.N_bulk = 0
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_floor)

    def test_event_classification_completeness(self):
        """
        Verifica que todos los sitios de la red estén clasificados en algún bin.
        Si se pierden sitios, el kMC se detiene o es incorrecto.
        """
        self.lat.heights[:] = 1 # Superficie plana
        
        # Adsorción: todos los sitios deberían estar en bin 0 (no hay vecinos a h+1=2)
        bins_ads = self.kmc._classify_adsorption_sites()
        total_sites_ads = sum(len(lst) for lst in bins_ads.values())
        self.assertEqual(total_sites_ads, self.lat.size**2, "Perdidos sitios en clasificación de adsorción")
        
        # Desorción: todos los sitios (h=1) en bin 4 (todos vecinos a h=1 tienen h=1)
        bins_des = self.kmc._classify_desorption_sites()
        total_sites_des = sum(len(lst) for lst in bins_des.values())
        self.assertEqual(total_sites_des, self.lat.size**2, "Perdidos sitios en clasificación de desorción")

    def test_mass_conservation_single_step(self):
        """
        Verifica que una adsorción reduzca N_bulk y aumente la altura media.
        """
        # Configurar para favorecer adsorción masiva (tasa desorción ~ 0)
        self.params.E_pb_over_kT = 50.0 # Enlaces fortísimos, nadie se va
        
        h_initial_sum = np.sum(self.lat.heights)
        n_bulk_initial = self.kmc.N_bulk
        
        # Forzar un evento de adsorción manualmente manipulando tasas es difícil en BKL,
        # pero podemos correr 1 paso y ver qué pasó.
        # Dado que inicializamos plano h=0, solo adsorción es posible.
        
        success = self.kmc.step()
        
        if success and self.kmc.history[-1][1] == "adsorption":
            # Verificar conservación
            h_final_sum = np.sum(self.lat.heights)
            n_bulk_final = self.kmc.N_bulk
            
            self.assertEqual(h_final_sum, h_initial_sum + 1, "Adsorción no incrementó la masa del cristal")
            self.assertEqual(n_bulk_final, n_bulk_initial - 1, "Adsorción no consumió del bulk")
    
    def test_conversion_percent(self):
        self.kmc.N_bulk = 50
        self.kmc.N_inc = 50
        self.assertEqual(self.kmc.conversion_percent, 50.0)
        
        self.kmc.N_bulk = 0
        self.kmc.N_inc = 100
        self.assertEqual(self.kmc.conversion_percent, 100.0)

if __name__ == '__main__':
    # Verbosity 2 da detalle de cada test individual
    unittest.main(verbosity=2)