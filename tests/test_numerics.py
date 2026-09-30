"""Funciones numéricas de bajo nivel (antes tests/test_utils.py, sin cambios de lógica)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.numerics import _safe_exp, _finite_or_zero  # noqa: E402


class TestNumerics(unittest.TestCase):
    """Evitan NaNs y overflows que arruinen simulaciones largas."""

    def test_safe_exp_limits(self):
        # Caso normal: e^1
        self.assertAlmostEqual(_safe_exp(1.0), np.exp(1.0))
        # Overflow positivo: se recorta, no lanza error ni devuelve inf
        val_overflow = _safe_exp(800)
        self.assertLess(val_overflow, np.inf)
        self.assertGreater(val_overflow, 0)
        # Satura igual por encima del límite
        self.assertEqual(_safe_exp(800), _safe_exp(1000))
        # Underflow: e^-inf ~ 0
        self.assertAlmostEqual(_safe_exp(-800), 0.0)

    def test_finite_or_zero_robustness(self):
        self.assertEqual(_finite_or_zero(np.inf), 0.0)
        self.assertEqual(_finite_or_zero(np.nan), 0.0)
        self.assertEqual(_finite_or_zero(-np.inf), 0.0)
        self.assertEqual(_finite_or_zero(5.5), 5.5)


if __name__ == "__main__":
    unittest.main()
