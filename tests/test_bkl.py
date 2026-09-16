import unittest
import numpy as np
import sys
import os

# Ajuste de path para encontrar src
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.params import KMCParams
from src.lattice import LatticeSOS
from src.bkl import KMC_BKL

class TestKMCLogic(unittest.TestCase):
    """
    Pruebas del motor BKL. Verifica la consistencia de tasas, conservación de masa
    y lógica de actualización.
    """
    def setUp(self):
        # Parámetros controlados para testeo
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
        # Testear límites de saturación (S)
        
        # Caso extremo alto: N_bulk infinito -> S debería ser S_ceil
        self.kmc.N_bulk = 1e20
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_ceil)
        
        # Caso extremo bajo: N_bulk cero -> S debería ser S_floor
        self.kmc.N_bulk = 0
        self.assertAlmostEqual(self.kmc.supersaturation, self.params.S_floor)

    def test_event_classification_completeness(self):
        """
        Verifica que todos los sitios de la red estén clasificados en algún 'bin' de eventos.
        Si la suma de sitios en las listas no es igual al total de la red, hay un bug lógico.
        """
        self.lat.heights[:] = 1 # Superficie plana
        total_lattice_sites = self.lat.size**2
        
        # Check Adsorción
        bins_ads = self.kmc._classify_adsorption_sites()
        total_sites_ads = sum(len(lst) for lst in bins_ads.values())
        self.assertEqual(total_sites_ads, total_lattice_sites, 
                         "Perdidos sitios en clasificación de adsorción")
        
        # Check Desorción
        bins_des = self.kmc._classify_desorption_sites()
        total_sites_des = sum(len(lst) for lst in bins_des.values())
        self.assertEqual(total_sites_des, total_lattice_sites, 
                         "Perdidos sitios en clasificación de desorción")

    def test_mass_conservation_single_step(self):
        """
        Verifica el balance de materia: si algo se adhiere, debe desaparecer del bulk.
        """
        # Configurar física para favorecer adsorción masiva (tasa desorción ~ 0)
        # Hacemos los enlaces extremadamente fuertes
        self.params.E_pb_over_kT = 50.0 
        
        h_initial_sum = np.sum(self.lat.heights)
        n_bulk_initial = self.kmc.N_bulk
        
        # Ejecutar un único paso
        success = self.kmc.step()
        
        # Si ocurrió una adsorción (lo más probable con estos params)
        if success and self.kmc.history and self.kmc.history[-1][1] == "adsorption":
            h_final_sum = np.sum(self.lat.heights)
            n_bulk_final = self.kmc.N_bulk
            
            # Verificar conservación
            self.assertEqual(h_final_sum, h_initial_sum + 1, 
                             "Adsorción no incrementó la masa del cristal")
            self.assertEqual(n_bulk_final, n_bulk_initial - 1, 
                             "Adsorción no consumió masa del bulk")
    
    def test_conversion_percent(self):
        # Test calculo porcentual simple
        self.kmc.N_bulk = 50
        self.kmc.N_inc = 50
        # (50 consumidos / (50 + 50) iniciales) * 100 = 50% ? Depende de la definición de N_bulk0
        # Asumiendo que N_bulk0 fue 100
        self.kmc.N0 = 100
        # Cálculo esperado: ((N0 - N_bulk) / N0) * 100 = ((100-50)/100)*100 = 50%
        # Nota: Ajusta esto según tu fórmula exacta en bkl.py si es diferente.
        
        # Si usas N_inc directamente:
        self.assertEqual(self.kmc.conversion_percent, (self.kmc.N0 - self.kmc.N_bulk)/self.kmc.N0 * 100)

if __name__ == '__main__':
    unittest.main()