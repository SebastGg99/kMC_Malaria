import unittest
import numpy as np
import sys
import os

# Ajuste de path para encontrar la carpeta src desde la carpeta tests
# Subimos un nivel (..) y entramos en src
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.utils import _safe_exp, _finite_or_zero

class TestNumerics(unittest.TestCase):
    """
    Control de calidad de las funciones numéricas de bajo nivel en utils.py.
    Fundamental para evitar NaNs y Overflows que arruinen simulaciones largas.
    """
    def test_safe_exp_limits(self):
        # Caso normal: e^1
        self.assertAlmostEqual(_safe_exp(1.0), np.exp(1.0))
        
        # Caso Overflow positivo (debe clippear al máximo seguro, no lanzar error)
        # Verificamos que sea finito y positivo
        val_overflow = _safe_exp(800)
        self.assertLess(val_overflow, np.inf)
        self.assertGreater(val_overflow, 0)
        
        # Verificamos que saturen igual por encima del límite
        self.assertEqual(_safe_exp(800), _safe_exp(1000))

        # Caso Underflow (e^-infinito debe ser 0)
        self.assertAlmostEqual(_safe_exp(-800), 0.0)

    def test_finite_or_zero_robustness(self):
        # Pruebas de robustez contra valores no numéricos o infinitos
        self.assertEqual(_finite_or_zero(np.inf), 0.0)
        self.assertEqual(_finite_or_zero(np.nan), 0.0)
        self.assertEqual(_finite_or_zero(-np.inf), 0.0)
        self.assertEqual(_finite_or_zero(5.5), 5.5)

if __name__ == '__main__':
    unittest.main()