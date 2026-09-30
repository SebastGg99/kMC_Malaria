# kMC_Malaria: crecimiento de cristales de hemozoína con kMC-SOS

Simulación **Kinetic Monte Carlo (kMC)** del crecimiento de cristales de hemozoína
(biomineral relacionado con la malaria). La superficie se modela con la aproximación
**Solid-On-Solid (SOS)** sobre una red cuadrada con condiciones de contorno periódicas,
y la evolución temporal usa el algoritmo **BKL** (Bortz-Kalos-Lebowitz, "n-fold way",
sin rechazo). Hay dos líneas de motor:

- **Línea dinámica** (`src.dynamic`): el reservorio se agota y la sobresaturación
  S = ln(C/C_eq) baja con el tiempo. Isotrópica.
- **Línea estática** (`src.static`): σ fija o concentración constante, con
  anisotropía x/y y factores de solvente. Generó todas las corridas de `results/`.

Para empezar, abre `notebooks/ejemplo_linea_estatica.ipynb` y
`notebooks/ejemplo_linea_dinamica.ipynb`: recorren el uso completo de cada línea.

**Estado de la refactorización:** hecha y verificada en `refac/` (2026-09-26) y
promovida a la raíz del repositorio (2026-09-29, §7). Nada se borró: el código, los
tests y los documentos anteriores están en `.descartables/pre_refactor/`.
**Base:** `auditoria.md` y `propuesta_refac.md`, con las decisiones del autor:

| # | Decisión | Aplicada como |
|---|---|---|
| D1 | Líneas según la tabla recomendada | Dinámica = `bkl.py` + `lattice.py` + `params.py`; estática = `bkl_v4.py` + `lattice_v3.py` + `params_v4.py` |
| D2 | Integrar v5 en el motor estático | Opciones de `bkl_v5` dentro de `KMC_BKL_Static`, con valores por defecto iguales a v4 |
| D3 | Nombres canónicos | `KMC_BKL_Dynamic`/`KMC_BKL_Static`, etc., más alias heredados (§3) |
| D4 | Estructura en subpaquetes | `src/{common,dynamic,static,plotting}` |
| D5 | Modo dinámico del motor estático tal cual, "no validado" | Sin cambios de código; documentado en `src/static/engine.py` |
| D6 | Descartar `model2D` | No está en `src/`; sigue en `.descartables/scripts/` (ver el efecto en §6) |
| D7 | Archivar `growthRate.py` y `probabilityAnalysis.py` | Copias sin modificar en `archive/` |
| D8 | Dejar `.descartables/` y mover lo propuesto | Todo lo reemplazado se **movió** a `.descartables/pre_refactor/` (§7); nada se borró |

---

## 1. Estructura

```
kMC_Malaria/
├── src/
│   ├── __init__.py            API pública: nombres canónicos + alias heredados
│   ├── common/
│   │   ├── numerics.py        _safe_exp, _finite_or_zero                  ← utils.py
│   │   ├── reference_data.py  FACE_DATA (única copia)                     ← utils.py
│   │   ├── observables.py     mean_height, roughness, step_density,
│   │   │                      count_by_coordination                       ← lattice_v2.py
│   │   ├── io.py              save/load, conversion_history, load_run,
│   │   │                      rebuild_from_metadata, git_commit           ← *Run.py, notebooks
│   │   ├── cli.py             parse_times, parse_sigma_range, ...         ← 7 scripts
│   │   └── legacy.py          load_legacy_pickle (kmc.pkl antiguos)
│   ├── dynamic/               params.py ← params.py (.descartables) · lattice.py ← lattice.py
│   │                          engine.py ← bkl.py
│   ├── static/                params.py ← params_v4.py · lattice.py ← lattice_v3.py
│   │                          engine.py ← bkl_v4.py (+ opciones de bkl_v5.py)
│   └── plotting/plotter.py    Plotter(style=classic|v2|academic|paper) ← plotter.py + plotter_v2.py + plotter_v3.py
│                              (paper: nuevo, paleta azul/rojo de la Fig. 9 de Nagpal et al. 2024)
├── scripts/
│   ├── run_sigma_scan.py      ← isotropicRun.py, isotropicRun_v2.py, anisotropicRun.py
│   ├── run_single.py          ← pRun.py, pRunAni.py, pRun_v2.py, pRun_v3.py (+ línea dinámica)
│   ├── reproduce_results.py   reproduce o verifica las 38 corridas de results/
│   ├── build_results_configs.py  genera configs/results/*.json
│   └── configs/               un JSON por variante que antes era un script clonado
│       ├── iso_flat.json, iso_random.json, aniso_flat.json            (barridos de σ)
│       ├── single_iso.json, single_aniso.json                          (pRun, pRunAni)
│       ├── sizes_fixed_seeds.json, sizes_density_seeds.json            (pRun_v2, pRun_v3)
│       ├── dynamic_final_results.json                                  (línea dinámica)
│       └── results/outputs_*.json   especificación reconstruida de cada carpeta de results/
├── notebooks/                 ejemplo_linea_estatica.ipynb, ejemplo_linea_dinamica.ipynb
│                              (uso completo de cada línea + comprobaciones de invariantes)
│                              + notebooks de análisis: final_results, morphologies, results_1 (§6)
├── tests/                     ver §2
├── archive/                   growthRate.py, probabilityAnalysis.py (D7) + README
├── requirements.txt           numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2
├── results/, outputs/         corridas históricas (versionadas en git)
├── references/                artículos de referencia (Nagpal et al. 2024, ...)
├── docs/                      documentación Sphinx (describe aún la estructura antigua)
├── auditoria.md, propuesta_refac.md, Plan_Paper.md, Plan_kMC_SOS.md
└── .descartables/             código y notebooks descartados (ignorado por git, §7)
    ├── pre_refactor/          src/, tests/ y documentos raíz anteriores a la promoción
    ├── scripts/               model2D*.py y versions/ (bkl_v2, params.py, plotter_v2, ...)
    ├── notebooks/             notebooks antiguos
    └── figures/               figuras sueltas que estaban en la raíz
```

Tamaño:
- motor + plotters + utilidades: 13 archivos y 3379 líneas → 17 archivos pequeños y
  2930 líneas;
- scripts de ejecución: 7 archivos y 2554 líneas → 2 scripts y 414 líneas, más
  2 utilidades nuevas de reproducción.

---

## 2. Verificación

Resultados de la ejecución del 2026-09-26 en `refac/` (Python 3.12.3, numpy 2.5.3, en
un entorno temporal). La promoción a la raíz (§7) solo movió archivos y ajustó rutas;
**la batería no se ha vuelto a ejecutar desde la raíz** (ver §7.3).

| Test | Qué comprueba | Resultado |
|---|---|---|
| `test_golden.py` | 19 casos (8 dinámicos, 11 estáticos: σ fija, concentración constante, modo dinámico, inicio `random`/`seeds`/`screw`, solvente, subclases) ejecutados con el código **original** (`make_golden.py`) y con el refactorizado | **Idénticos bit a bit**: alturas, tiempos, historial, contadores, historias y snapshots |
| `test_golden.py` (v5) | `record_adsorption_probs=False` da la misma trayectoria; `crystal_fraction_percent` == `conversion_percent` de `bkl_v5` original | OK |
| `test_legacy_pickles.py` | Los **38** `kmc.pkl` de `results/` cargan y coinciden con su `snaps.pkl`/`stats.pkl` | OK (antes era imposible, ver §4.1) |
| `test_results_reproduction.py` | Las **38** corridas de `results/` se regeneran: los primeros 150 eventos de cada una coinciden con el historial guardado (evento y sitio exactos; tiempo con rtol 1e-12) | OK |
| `test_plotter.py` | PNG de `plot_crystal_3d` (5 combinaciones estilo/modo) y GIF (4) frente a los 3 plotters originales | **PNG idénticos píxel a píxel; GIF idénticos byte a byte** |
| `test_scripts.py` | `run_sigma_scan` frente al flujo de `anisotropicRun.py` con módulos originales; `rebuild_from_metadata`; `run_single` (estático, varios tamaños, dinámico) | OK |
| `dynamic/`, `static/`, `test_numerics.py` | Tests antiguos portados y tests **nuevos** de la línea estática (antes no tenía) | 29 tests OK |

Cómo ejecutarlos (desde la raíz del repositorio):

```bash
python -m unittest discover -s tests -t .          # todo (~4 min)
KMC_REPRO_EVENTS=50 python -m unittest tests.test_results_reproduction   # más rápido
```

- Las referencias golden solo valen con numpy 2.5.3 (`tests/golden/ENV.txt`). Con
  otra versión, `test_golden` se omite y hay que regenerarlas con
  `python tests/make_golden.py`, que ejecuta el código original de
  `.descartables/pre_refactor/src/`.
- `test_plotter`, `test_golden` (parte de v5) y `test_scripts` (paridad) usan los
  módulos originales de `.descartables/pre_refactor/src/` y
  `.descartables/scripts/versions/` (`tests/_paths.py`); se omiten solos si no están.
- El `.env/` local es Python 3.14.4 con numpy 2.5.3 y matplotlib 3.11.2, pero **sin
  pandas** (lo usan `test_scripts`, `scripts/` y los notebooks de análisis).

---

## 3. Uso

```python
import sys; sys.path.insert(0, "/home/sggphysics/kMC_Malaria")   # la carpeta que CONTIENE src/
from src import *

# Línea estática (nombres canónicos)
lat = LatticeSOSStatic(size=(80, 80), seed=42); lat.initialize(mode="flat", max_height=1)
kmc = KMC_BKL_Static(lattice=lat, params=KMCParamsStatic(...), N_bulk0=2000, rng_seed=123,
                     time_scale=70, record_adsorption_probs=False)   # 1.8-2x más rápido (medido)
snaps, stats = kmc.run(t_end=5, snapshot_times=np.arange(0, 5.5, 0.5))
Plotter(kmc, style="v2").plot_crystal_3d(snapshots=snaps)

# Cargar una corrida antigua (ahora sí funciona kmc.pkl)
from src.common.io import load_run
run = load_run("results/outputs_aniso_3/size_80x80_sigma_0.3_seed_123", load_kmc=True)
```

**Alias heredados**, para que los notebooks no tengan que cambiar:
- `KMC_BKL`, `KMCParams`, `LatticeSOS`, `SelectiveKMC`, `KMC_NoDesNoMig` → línea
  dinámica;
- `KMC_BKL_v4`, `KMCParams_v4`, `LatticeSOS_v4`, `LatticeSOS_v3`, `SelectiveKMC_v4`,
  `KMC_NoDesNoMig_v4` → línea estática;
- `Plotter`, `Plotter_v2`, `Plotter_v3` → `Plotter` con estilo `classic`/`v2`/`academic`.
  El estilo `paper` (azul = capa completa, rojo = capa en crecimiento) es nuevo.

`KMC_BKL_v5` no se exporta porque no tenía consumidores. Su equivalente es
`KMC_BKL_Static(..., record_adsorption_probs=False)` junto con
`kmc.crystal_fraction_percent`.

Scripts (desde la raíz del repositorio):

```bash
python scripts/run_sigma_scan.py --config scripts/configs/aniso_flat.json --times 0:5.5:0.5 \
    --size 80 80 --n-seeds 100 --fixed-sigma 0.3 0.7 --sigma-step 0.1 --time-scale 70 \
    --output-dir outputs_aniso_new [--no-adsorption-probs] [--no-kmc-pickle]
python scripts/run_single.py --config scripts/configs/single_iso.json --times 0:8:1 --size 10 10 --gif
python scripts/reproduce_results.py --folder outputs_aniso_3 --check 300
```

---

## 4. Hallazgos nuevos durante la refactorización (no estaban en la auditoría)

### 4.1 Los 38 `kmc.pkl` nunca se pudieron cargar

`_LatticeSize` (en `lattice.py` y `lattice_v3.py`) es una subclase de `tuple` con
`__new__(cls, nx, ny)`. Pickle la reconstruye llamando a `__new__(cls, (nx, ny))` con
un solo argumento, así que `pickle.load` falla con `TypeError` **incluso con el código
original**; lo comprobé con `src/` en `sys.path`. Son unos 870 MB de archivos que no se
podían abrir. Esto probablemente explica por qué los notebooks reconstruyen el objeto
desde `metadata.json`.

**Corrección** en las dos redes refactorizadas: `__new__` acepta también la tupla. No
afecta a la simulación: los tests golden siguen idénticos.

### 4.2 `n_seeds` de metadata.json no se aplicó en la mayoría de corridas

El primer snapshot de 30 de las 38 corridas tiene **una sola partícula**, aunque su
metadata dice `n_seeds=100` o `500`. Las corridas se reproducen evento a evento **solo
si se simulan sin semillas**. El código que las generó no sembraba, aunque el
`bkl_v4.py` actual sí lo haría.

La metadata tampoco guardaba el modo de inicio. Deducido de los datos y verificado por
reproducción:

| Carpeta | Corridas | Inicio efectivo | Semillas efectivas | time_scale | K0 |
|---|---|---|---|---|---|
| `outputs_aniso` | σ 0.4–0.7 / σ 1–7 | flat | 0 (metadata: 500) | 70 / **100** | 1.16718 / **11.6718** |
| `outputs_aniso_2` | 4 | flat | 0 (metadata: 500) | 70 | 11.6718 |
| `outputs_aniso_3` | 5 | flat | 0 (metadata: 100) | 70 | 0.516718 |
| `outputs_iso` | σ 0.4–0.7 / σ 1–6 | flat / **seeds (isla compacta de 500)** | 0 / 500 | 70 / **80** | 1.16718 / **0.2116718** |
| `outputs_iso_2` | 4 | flat | 0 (metadata: 500) | 100 | 11.6718 |
| `outputs_iso_3` | 5 | flat | 0 (metadata: 100) | 70 | 0.5116718 |
| `outputs_iso_4` | 2 | **random** | — | 110 | 11.6718 |

Consecuencias:
- `outputs_aniso` y `outputs_iso` **mezclan corridas con distinto K0, `time_scale` e
  inicio** en la misma carpeta. Al compararlas como una sola serie se mezclan
  condiciones distintas.
- La tabla de `auditoria.md` §6 solo miraba la primera corrida de cada carpeta; esta
  tabla la corrige.
- `rebuild_from_metadata` con metadata antigua reconstruye mal estas corridas, porque
  sembraría `n_seeds`. Para reproducirlas hay que usar
  `scripts/configs/results/*.json` con `reproduce_results.py`.

### 4.3 Otros

- **Conversión de v5 frente a v4:** las trayectorias son idénticas (verificado), pero
  la conversión de v5 llega a **1412 %** con concentración constante (caso
  `sta_constant_conc`). Se expone como `crystal_fraction_percent`, aparte
  de `conversion_percent`.
- **`SelectiveKMC` y `KMC_NoDesNoMig`** (en las dos líneas) producen exactamente las
  mismas trayectorias con los mismos flags. Por eso se unificaron sin riesgo.
- **Precisión de los CSV:** `conversion_history.csv` se escribía con `pandas.to_csv`
  (unas 15 cifras significativas). El `.npy` guarda la precisión completa. Así se
  mantiene.
- **matplotlib moderno:** el `plotter_v2` original usa `plt.cm.get_cmap`, eliminado en
  matplotlib 3.9, y fallaba en modo `surface`. Corregido en el `Plotter` unificado.

### 4.4 Rendimiento (medido, red 80x80, σ=0.3, 300 eventos)

| Motor | ms/evento |
|---|---:|
| `bkl_v4` original | 36.0 |
| `KMC_BKL_Static`, valores por defecto (misma trayectoria) | 32.1 (reutiliza la clasificación de desorción para la incorporación) |
| `KMC_BKL_Static(record_adsorption_probs=False)` | 17.9 (x1.8; en 40x40, x2.05) |

Una corrida típica de `results/` (unos 50 000–210 000 eventos en 80x80) tarda por tanto entre 0.5 y 2 h. El cuello de botella sigue siendo reclasificar toda la red en cada evento (§8.5).

---

## 5. Diferencias deliberadas respecto a los originales

Ninguna cambia la física ni la trayectoria:

1. `_LatticeSize.__new__` acepta una tupla (§4.1).
2. `Plotter(style="academic", mode="surface")` lanza `ValueError`. Antes dibujaba una
   figura vacía.
3. `plot_growth_rate_analysis` muestra los parámetros x/y de la línea estática. Antes
   fallaba con `AttributeError` porque buscaba `p.E_pb_over_kT` y `p.delta`.
4. En `run_single.py`, `--n-seeds` sí tiene efecto. En `pRun.py`/`pRunAni.py` no lo
   tenía: el motor recibía 10 fijo. El valor por defecto del config es 10, el
   efectivo de antes.
5. `run_single.py` escribe en `--output-dir`. `pRun.py` escribía en el directorio
   actual, y `pRun_v3` usaba el sufijo `_1` en los agregados.
6. `metadata.json` guarda además `engine`, `init_mode`, `init_kwargs`, `N_bulk0`,
   `constant_concentration`, `use_solvent`, `record_adsorption_probs`, `N_seed0`,
   `git_commit` y `numpy_version`.
7. Respecto a `propuesta_refac.md` §4: la masa cristalina de v5 se calcula bajo demanda
   (`np.sum`) y no de forma incremental. El coste es despreciable frente a las 4
   clasificaciones O(N) por paso, y no puede desincronizarse si alguien modifica
   `lat.heights`.
8. Estilo nuevo `Plotter(style="paper")` y método `plot_morphology_sequence`: azul =
   capa completa (niveles < min(heights)), rojo = capa en crecimiento, como la Fig. 9
   de `references/modern_kMC/`. No tiene original: lo cubre `TestPaperStyle`.

---

## 6. Qué se rompe en los notebooks de análisis (y cómo se arregla)

| Consumidor | Qué deja de funcionar | Arreglo |
|---|---|---|
| `notebooks/final_results.ipynb` celda 1 | `from scipy.optimize import curve_fit` | `scipy` no está en `requirements.txt` ni en la lista de librerías permitidas |
| `notebooks/final_results.ipynb` celdas 8–20 | `from model2D_v3 import ...` (D6: descartado) | Ninguno dentro de `src/`. Si se necesitan esas figuras, ejecutar esas celdas con `.descartables/scripts/` en `sys.path` |
| `notebooks/results_1.ipynb` | `from plotter_v3 import Plotter_v3` | `from src import Plotter_v3` |
| Todos los notebooks | `sys.path.append('/home/sgaviria/MalariaProject/')` | `sys.path.append('/home/sggphysics/kMC_Malaria')` (la raíz) |
| Rutas de datos `/home/sgaviria/MalariaProject/results/...` (los 3 notebooks) | No existen en esta máquina | Cambiar por rutas relativas a la raíz (`results/...`) |
| `reconstruct_kmc_from_metadata` (2 notebooks) | Sigue funcionando, pero reconstruye mal las corridas de §4.2 | `src.common.io.rebuild_from_metadata`, o los configs de `scripts/configs/results/` |

---

## 7. Promoción a la raíz (2026-09-29)

### 7.1 Qué se movió

Nada se borró. Se comprobó con un manifiesto de hashes SHA-256 que los 158 archivos
afectados siguen existiendo, en su sitio nuevo.

| Antes | Ahora |
|---|---|
| `refac/src/`, `refac/tests/`, `refac/scripts/`, `refac/archive/` | `src/`, `tests/`, `scripts/`, `archive/` |
| `refac/requirements.txt` | `requirements.txt` |
| `refac/notebooks/ejemplo_linea_*.ipynb` | `notebooks/` (junto a los de análisis) |
| `refac/README.md` | este `README.md` (adaptado); el original, en `.descartables/pre_refactor/refac_README.md` |
| `src/*` (21 `.py`, incluido `__init__.py` con cambios sin commitear) | `.descartables/pre_refactor/src/` |
| `tests/*` (`test_bkl.py`, `test_lattice.py`, `test_utils.py`, `tests_suite.py`) | `.descartables/pre_refactor/tests/` (portados a `tests/dynamic/` y `tests/test_numerics.py`) |
| `README.md`, `requirements.txt` | `.descartables/pre_refactor/raiz/` |
| `CLAUDE.md`, `.github/copilot-instructions.md` | actualizados en su sitio; copia previa en `.descartables/pre_refactor/raiz/` |

El detalle de cada archivo movido está en `.descartables/pre_refactor/README.md`.

### 7.2 Git

- `.gitignore` excluye `.descartables/*`. Para git, lo movido allí aparece como
  **borrado**, aunque sigue en disco y en el historial (commit `6109612` y
  anteriores).
- No se ha hecho commit ni tag. Para dejar un punto de retorno explícito:

```bash
git tag pre-refactor 6109612              # último commit con el código original
git switch -c refactor/dos-lineas
# -A registra también la salida de los archivos antiguos de src/ y tests/
git add -A src tests scripts archive requirements.txt README.md .github
git add notebooks/ejemplo_linea_estatica.ipynb notebooks/ejemplo_linea_dinamica.ipynb
git commit -m "Refactorización: dos líneas de motor (dinámica y estática)"
```

### 7.3 Pendiente de la promoción

- Volver a ejecutar `python -m unittest discover -s tests -t .` desde la raíz. Solo se
  cambiaron rutas: `tests/_paths.py` busca ahora el código original en
  `.descartables/pre_refactor/src/`, así que los tests de paridad siguen activos.
- `docs/source/*.md` (Sphinx) sigue describiendo la estructura antigua.

---

## 8. Pendiente (física, fuera del alcance de la refactorización)

Estos puntos requieren decisión del autor y validación por separado (auditoría §5):
1. Convenciones de conteo de enlaces. Los tests
   `test_dynamic_lattice_counts_neighbors_at_or_above_h0` y
   `test_static_lattice_counts_only_equal_height_neighbors` las **congelan** a
   propósito.
2. Factor `N_bulk/N0` duplicado en modo dinámico.
3. Incorporación sin límite.
4. Factor 2 de δ y significado de `fixed_sigma`.
5. Rendimiento: clasificación incremental. Rompería la paridad bit a bit, así que
   necesitaría validación estadística.
