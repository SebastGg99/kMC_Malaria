# tests - Estrategia de validacion

La carpeta `tests/` combina pruebas del nucleo estable con pruebas de lineas experimentales.

## Archivos de prueba actuales

- `test_utils.py`: robustez numerica (`_safe_exp`, `_finite_or_zero`).
- `test_lattice.py`: geometria SOS, PBC y reglas de migracion.
- `test_bkl.py`: consistencia del motor BKL base (tasas, bins, balance de masa).
- `tests_suite.py`: suite agregada para ejecucion integral.
- `test_adsorption_probability_analysis.py`: pruebas de analisis de probabilidad de adsorcion y sigma fijo.
- `test_paper_kmc.py`: pruebas para una API `paper_kmc` (cara 110/101 y regimenes).

## Alcance cubierto

- Estabilidad numerica de utilidades criticas.
- Integridad geometrica de la red SOS.
- Consistencia de seleccion de eventos y conservacion de materia en BKL.
- Comportamiento del nucleo base con sobresaturacion dinamica (deplecion de reservorio).

## Estado importante de compatibilidad

En el estado actual de `src/`, no existen `analysis.py` ni `paper_kmc.py`.

Por ello:

- `test_adsorption_probability_analysis.py` y `test_paper_kmc.py` requieren completar/ajustar implementaciones antes de formar parte de la bateria estable de CI.
- Los escenarios de sigma fijo se consideran cobertura experimental hasta cerrar esa integracion.

## Ejecucion recomendada

Para correr solo el nucleo estable:

- `python -m unittest tests.test_utils tests.test_lattice tests.test_bkl`

Para ejecutar todo el directorio y detectar faltantes de integracion:

- `python -m unittest discover -s tests -p "test_*.py"`

## Criterios minimos antes de publicar documentacion

- Sin fallos en pruebas estables del nucleo.
- Sin errores de import en modulos documentados de `src/`.
- Sin warnings criticos al construir Sphinx en modo estricto.
