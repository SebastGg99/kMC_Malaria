# src - Documentacion tecnica del nucleo kMC

Este documento describe el estado actual del codigo en `src/`.

## Modulos base

### params.py

Define `KMCParams` como contenedor de parametros del motor base:

- Termicos/cineticos: `T`, `K0_plus`, `K_inc_plus`.
- Energeticos: `E_pb_over_kT`, `phi_over_kT`, `delta`.
- Termodinamicos para deplecion dinamica: `V`, `C_eq`.
- Control numerico: `S_floor`, `S_ceil`.
- `fixed_sigma` permanece en la API por compatibilidad, pero la linea base se documenta y valida en modo dinamico.

### lattice.py

Define `LatticeSOS` (SOS discreto con 4 vecinos y contorno periodico).

Capacidades principales:

- Inicializacion de superficie (`flat`, `random_surface`).
- Siembra controlada de islas con `seed_surface_with_islands`.
- Conteo de enlaces para adsorcion/desorcion.
- Lista de destinos de migracion permitidos por geometria local.

### bkl.py

Define `KMC_BKL` y las variantes `SelectiveKMC` y `KMC_NoDesNoMig`.

Responsabilidades:

- Tasas `r_a`, `r_d`, `r_m`, `r_inc` con defensas numericas.
- Sobresaturacion dinamica por deplecion de reservorio (`N_bulk`) en el flujo base.
- Clasificacion de sitios por coordinacion local.
- Seleccion de eventos tipo n-fold way.
- Integridad en modo debug y seguimiento de conversion/historial.

### plotter.py

`Plotter` para la linea base:

- Visualizacion 3D (`surface` o `voxel`).
- Generacion de GIF a partir de snapshots.
- Curvas de conversion temporal.

## Modulos v2

### params_v2.py

Incluye:

- `KMCParams_v2` (parametros adimensionales/efectivos).
- `fixed_sigma` opcional (sigma del paper) y propiedad `T` de solo lectura para compatibilidad de visualizacion.

### lattice_v2.py

`LatticeSOS_v2` extiende la inicializacion con modos:

- `flat`
- `random`
- `seeds`
- `screw` (rampa helicoidal discreta)

Tambien incorpora metricas auxiliares como `mean_height`, `roughness` y `step_density`.

### bkl_v2.py

`KMC_BKL_v2` implementa una formulacion con:

- Entrada fisica en sigma: $\sigma = C/C_{eq} - 1$.
- Variable interna para tasas: $S = \ln(1+\sigma)$.
- Opcion de sobresaturacion fija via `fixed_sigma`.
- Si `fixed_sigma` no se usa, modo de concentracion constante o reservorio dinamico.
- Registro de observables (`height_history`, `adsorption_probs_history`).
- Probabilidades de adsorcion por clase (`_adsorption_probabilities_3class`).

Incluye variantes:

- `SelectiveKMC_v2`
- `KMC_NoDesNoMig_v2`

### plotter_v2.py

`Plotter_v2` agrega:

- Visualizacion 3D y GIF para la linea v2.
- Analisis de velocidad de crecimiento.
- Analisis de probabilidades de adsorcion por clase.

## Modulos auxiliares

### model2D.py

Motor alternativo 2D con eventos de adsorcion, desorcion, difusion e incorporacion, util para pruebas de sensibilidad fuera del flujo BKL principal.

### utils.py

Utilidades numericas y geometricas compartidas:

- `_safe_exp`
- `_finite_or_zero`
- Funciones de vecindad y barreras para `model2D.py`.

## Flujo recomendado

1. Elegir linea de trabajo (base o v2).
2. Instanciar parametros y red correspondientes.
3. Inicializar superficie.
4. Instanciar motor kMC.
5. Ejecutar `run(...)`.
6. Analizar snapshots/estadisticas con `Plotter` o `Plotter_v2`.

## Nota de consistencia

En el estado actual del repositorio no existen `src/paper_kmc.py` ni `src/analysis.py`. Si se requiere documentar o ejecutar esa linea, primero debe incorporarse al arbol de `src/`.