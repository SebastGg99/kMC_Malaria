# MalariaProject

## Descripción General

MalariaProject implementa simulaciones **Kinetic Monte Carlo (kMC)** para crecimiento cristalino sobre superficies **Solid-On-Solid (SOS)** con condiciones periódicas y selección de eventos tipo **BKL (n-fold way)**.

Actualmente el proyecto tiene dos líneas principales de modelado en `src/`:

- Línea base (`params.py`, `lattice.py`, `bkl.py`, `plotter.py`).
- Línea v2, orientada a reproducción de resultados de referencia principal (`params_v2.py`, `lattice_v2.py`, `bkl_v2.py`, `plotter_v2.py`).

Además, se mantiene una variante alternativa de un modelo en 2D (`model2D.py`) con las mismas condiciones periódicas y el mismo algoritmo de selección de eventos.

## Actualizaciones relevantes del núcleo

### 1. Red SOS base (`lattice.py`)

La clase `LatticeSOS` mantiene vecindad de 4 vecinos y contorno periódico, e incorpora:

- Inicialización reproducible por semilla.
- Método `seed_surface_with_islands(...)` para sembrar islas mono-capa en sitios únicos.
- Reglas de migración con restricción geométrica (no migrar a columnas más altas).

### 2. Parámetros base (`params.py`)

La `dataclass` `KMCParams` incluye los parámetros cinéticos y termodinámicos del modelo clásico:

- Variables termodinámicas de depleción (`V`, `C_eq`) para sobresaturación dinámica.
- Clamps `S_floor` y `S_ceil` para estabilidad numérica.

### 3. Motor BKL base (`bkl.py`)

La clase `KMC_BKL` implementa:

- Tasas de adsorción, desorción, migración e incorporación con protección numérica.
- Sobresaturación dinámica por depleción del reservorio (`N_bulk`) en el flujo base.
- Clasificación por coordinación local para selección rejection-free.
- Validaciones internas en modo `debug` (consistencia de bins, tasas y chequeos termodinámicos y de consistencia física).

También se mantienen variantes de control selectivo de procesos:

- `SelectiveKMC`
- `KMC_NoDesNoMig`

### 4. Línea v2 (`*_v2.py`)

La variante v2 introduce una formulación alineada con el artículo de referencia principal:

- `KMCParams_v2` con parámetros adimensionales y `fixed_sigma` opcional.
- `LatticeSOS_v2` con modos de inicialización `flat`, `random`, `seeds` y `screw`.
- `KMC_BKL_v2` con manejo explícito de:
  - $\sigma = C/C_{eq} - 1$ (magnitud física de entrada).
  - $S = \ln(1+\sigma)$ (magnitud interna de las tasas).
  - Opción de sobresaturación fija (`fixed_sigma`).
  - Si `fixed_sigma` no está activo, operación con concentración constante o reservorio dinámico.
- `Plotter_v2` con visualización 3D, GIF y utilidades de análisis de velocidad/probabilidades.

### 5. Diferencia principal entre líneas

- Línea base: sobresaturación diseñada para evolucionar dinámicamente con depleción de soluto.
- Línea v2: mantiene esa posibilidad y además habilita sobresaturación fija mediante `fixed_sigma`.

### 6. Utilidades y API del paquete

- `utils.py` centraliza utilidades robustas (`_safe_exp`, `_finite_or_zero`) y helpers de vecindad para `model2D.py`.
- `src/__init__.py` exporta las clases y funciones principales de ambas líneas (base y v2).

## Flujo recomendado de simulación

1. Definir parámetros (`KMCParams` o `KMCParams_v2`).
2. Construir e inicializar la red (`LatticeSOS` o `LatticeSOS_v2`).
3. Instanciar el motor (`KMC_BKL` o `KMC_BKL_v2`).
4. Ejecutar `run(...)` y recolectar snapshots/estadísticas.
5. Visualizar con `Plotter`/`Plotter_v2` y guardar artefactos en `results/`.
