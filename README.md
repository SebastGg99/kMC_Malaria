# MalariaProject

## Descripción General

Este repositorio contiene una implementación robusta de una simulación **Kinetic Monte Carlo (kMC)** aplicada al estudio de la cinética de crecimiento de cristales de Hemozoína. El software modela la superficie del cristal utilizando una aproximación **Solid-On-Solid (SOS)** sobre una red cuadrada con condiciones de contorno periódicas.

El núcleo de la simulación utiliza el **algoritmo BKL (Bortz-Kalos-Lebowitz)**, también conocido como "n-fold way" o algoritmo libre de rechazo, para garantizar una evolución temporal eficiente y exacta del sistema estocástico.

## Estructura del Código Fuente (`src/`)

El código está estructurado de manera modular para separar la física del sistema, la topología de la red y la lógica del algoritmo estocástico.

### 1. `lattice.py`: Topología y Estado de la Superficie
Contiene la clase [`LatticeSOS`](src/lattice.py#L4). Esta clase gestiona el estado geométrico del cristal representando la superficie como una matriz de alturas enteras bajo el modelo **Solid-On-Solid (SOS)**.

**Inicialización:**
*   [`__init__(size, seed, debug)`](src/lattice.py#L10): Constructor que crea una red cuadrada vacía de tamaño `size` y establece el generador de números aleatorios.
*   [`initialize(init_mode, max_roughness)`](src/lattice.py#L19): Configura la topografía inicial de la superficie (plana o rugosa aleatoria).

**Topología:**
*   [`wrap(idx)`](src/lattice.py#L31): Aplica condiciones de contorno periódicas para asegurar la continuidad de la red en los bordes.
*   [`neighbors4(site)`](src/lattice.py#L36): Retorna las coordenadas de los 4 vecinos más cercanos (vecindad de von Neumann) respetando las condiciones periódicas.
*   [`get_sites()`](src/lattice.py#L129): Devuelve una lista con las coordenadas de todos los sitios de la red.

**Manipulación de Estado:**
*   [`get_height(site)`](src/lattice.py#L44): Retorna la altura actual de una columna.
*   [`inc_height(site, dh)`](src/lattice.py#L47): Incrementa la altura de un sitio, simulando un evento de deposición o adsorción.
*   [`dec_height(site, dh)`](src/lattice.py#L53): Decrementa la altura de un sitio, simulando un evento de desorción. Incluye validaciones para evitar alturas negativas.

**Cálculo de Enlaces y Coordinación:**
*   [`lateral_neighbors_at_level(site, level)`](src/lattice.py#L62): Método auxiliar que cuenta cuántos vecinos alcanzan al menos cierta altura `level`. Usado para calcular energías.
*   [`adsorption_bonds(site)`](src/lattice.py#L76): Calcula cuántos vecinos laterales tendría una partícula si se adsorbiera en el sitio. Determina la estabilidad de llegada.
*   [`desorption_bonds(site)`](src/lattice.py#L89): Calcula cuántos vecinos laterales vinculan a la partícula en el tope de la columna. Determina la barrera energética para la desorción.
*   [`migration_targets(site)`](src/lattice.py#L105): Identifica a qué sitios vecinos puede moverse una partícula (donde la altura destino $\le$ altura origen), gobernando la difusión superficial.

### 2. `params.py`: Parámetros Fisicoquímicos
Define la `dataclass` [`KMCParams`](src/params.py#L4), que actúa como contenedor inmutable para los parámetros de la simulación. Su diseño facilita la configuración centralizada de la física del experimento.

**Parámetros Termodinámicos y Ambientales:**
*   [`T`](src/params.py#L5): Temperatura absoluta del sistema.
*   [`V`](src/params.py#L12): Volumen de la solución circundante. Fundamental para la variación dinámica de la concentración.
*   [`C_eq`](src/params.py#L13): Concentración de equilibrio del soluto. Base para el cálculo de la supersaturación ($S$).

**Parámetros Cinéticos (Arrhenius):**
*   [`K0_plus`](src/params.py#L6): Factor pre-exponencial cinético base para eventos reversibles (adsorción, desorción, migración).
*   [`K_inc_plus`](src/params.py#L7): Factor pre-exponencial específico para el evento de incorporación irreversible.
*   [`E_pb_over_kT`](src/params.py#L8): Energía por enlace lateral (partícula-partícula) normalizada por $k_B T$. Controla la cooperatividad.
*   [`phi_over_kT`](src/params.py#L9): Energía de activación de difusión base (potencial de superficie) normalizada por $k_B T$.
*   [`delta`](src/params.py#L10): Parámetro de ajuste que modula la dependencia de la tasa de adsorción respecto a la supersaturación.

**Límites de Seguridad Numérica:**
*   [`S_floor`](src/params.py#L14): Límite inferior (suelo) para la supersaturación, evitando inestabilidades en regímenes de disolución rápida.
*   [`S_ceil`](src/params.py#L15): Límite superior (techo) para la supersaturación, previniendo tasas de adsorción numéricamente explosivas.

### 3. `bkl.py`: Motor de Simulación (Algoritmo BKL)
Contiene la clase [`KMC_BKL`](src/bkl.py#L11), que orquesta la evolución temporal del sistema.

**Configuración y Estado:**
*   [`__init__(lattice, params, N_bulk0, ...)`](src/bkl.py#L12): Inicializa la simulación inyectando la red, los parámetros físicos y la cantidad de soluto inicial. Configura las semillas del cristal si es necesario.
*   [`supersaturación`](src/bkl.py#L48): Propiedad que calcula dinámicamente la fuerza impulsora termodinámica ($S$) basada en la concentración actual de soluto.
*   [`conversion_percent`](src/bkl.py#L55): Propiedad que monitorea el progreso de la cristalización (porcentaje de soluto convertido en cristal).

**Cálculo de Tasas (Rates):**
Implementación de las ecuaciones de Arrhenius para cada proceso elemental:
*   [`r_a(i)`](src/bkl.py#L63): Tasa de **Adsorción**. Depende de $S$ y se ajusta por la depleción del soluto en el 'bulk'.
*   [`r_d(i)`](src/bkl.py#L75): Tasa de **Desorción**. Depende fuertemente del número de vecinos laterales `i`.
*   [`r_m(i)`](src/bkl.py#L80): Tasa de **Migración**. Difusión superficial térmica.
*   [`r_inc(i)`](src/bkl.py#L85): Tasa de **Incorporación**. Evento de crecimiento irreversible.

**Algoritmo BKL (Clasificación y Selección):**
El corazón del método "rejection-free" o n-fold way:
*   **Clasificadores**: Agrupan los sitios de la red según su entorno local para evitar recalcular tasas idénticas.
    *   [`_classify_adsorption_sites()`](src/bkl.py#L91): Clasifica todos los sitios donde puede ocurrir adsorción.
    *   [`_classify_desorption_sites()`](src/bkl.py#L98): Clasifica sitios ocupados susceptibles a desorber.
    *   [`_classify_migration_sites()`](src/bkl.py#L106): Clasifica sitios móviles que tienen destinos válidos.
    *   [`_classify_incorporation_sites()`](src/bkl.py#L117): Clasifica sitios candidatos para incorporación permanente.
*   **Selectores Monte Carlo**:
    *   [`_choose_event_type(...)`](src/bkl.py#L127): Elige qué tipo de evento físico ocurre (ads/des/mig/inc) proporcional a sus tasas totales $W$.
    *   [`_choose_class(weights)`](src/bkl.py#L139): Elige la clase de coordinación específica dentro del evento seleccionado.
    *   [`_choose_site_uniform(sites)`](src/bkl.py#L153): Elige al azar un sitio específico dentro de la clase ganadora.

**Ejecución y Control:**
*   [`step()`](src/bkl.py#L189): Ejecuta un único paso de Monte Carlo: calcula tasas totales, avanza el tiempo estocásticamente, selecciona y ejecuta el evento, y actualiza la red. Incluye verificaciones de integridad si `debug=True`.
*   [`run(t_end, snapshot_times, max_events)`](src/bkl.py#L292): Bucle principal que itera llamadas a `step()` hasta cumplir la condición de parada. Gestiona la grabación de "snapshots" del estado del sistema en tiempos específicos.
*   [`_validate_integrity()`](src/bkl.py#L158): (Modo Debug) Auditoría exhaustiva que verifica consistencia matemática y física (sin tasas negativas, conservación de sitios, termodinámica).

**Visualización:**
*   [`plot_crystal_3d(...)`](src/bkl.py#L327): Genera visualizaciones 3D del cristal utilizando `matplotlib`. Soporta modo superficie continua (`mode="surface"`) o visualización de voxeles (`mode="voxel"`).

### 4. `utils.py`: Estabilidad Numérica
Provee funciones auxiliares para el manejo robusto de operaciones de punto flotante:
*   [`_safe_exp()`](src/utils.py#L9): Evita desbordamientos (*overflow*) en cálculos exponenciales de Arrhenius clamping de argumentos.
*   [`_finite_or_zero()`](src/utils.py#L16): Sanitiza los resultados para evitar la propagación de valores `NaN` o `Inf` en las tasas de reacción.

### 5. Sistema de Auditoría y Modo Debug (`debug=True`)
El código implementa un sistema de **Programación Defensiva** activable mediante el flag `debug=True` en los constructores de `LatticeSOS` y `KMC_BKL`. Este modo sacrifica rendimiento a cambio de garantías estrictas de corrección física y matemática paso a paso. Las pruebas internas incluyen:

#### A. Validez Física (Physical Sanity)
*   **Geometría SOS**: Se verifica en cada operación de la red que no se violen las restricciones del modelo Solid-On-Solid. Por ejemplo, se asegura que las alturas nunca sean negativas y que la migración superficial respete la gravedad (las partículas no pueden "levitar" o subir a un sitio más alto sin un evento de desorción previo).
    *   *Ubicación*: [`LatticeSOS.inc_height`](src/lattice.py#L47), [`LatticeSOS.dec_height`](src/lattice.py#L53), [`LatticeSOS.migration_targets`](src/lattice.py#L105).

#### B. Integridad de Contenedores (Binning Integrity)
*   **Conservación de Sitios**: El algoritmo BKL depende de clasificar cada sitio de la red en listas (bins) según su coordinación. En modo debug, el método [`_validate_integrity`](src/bkl.py#L160) recalcula el conteo total de sitios en estas listas y verifica que coincida exactamente con el tamaño de la red ($N \times N$). Esto detecta "fugas" de sitios donde una partícula podría perderse del sistema de simulación.

#### C. Sanidad de Tasas (Rate Sanity)
*   **Estabilidad Numérica**: Se comprueba exhaustivamente que todas las tasas de transición calculadas ($r_a, r_d, r_m, r_{inc}$) sean valores finitos y no negativos para todas las configuraciones posibles de vecinos (0-4). Esto previene la propagación silenciosa de `NaN` (Not a Number) o valores infinitos que colapsarían la simulación pasos más adelante.

#### D. Garantía de Determinismo
*   **Seed Check**: Para asegurar que cualquier error sea reproducible, el modo debug exige explícitamente que se proporcione una semilla (`rng_seed`) al generador de números aleatorios. Si no se provee, la inicialización falla preventivamente.

#### E. Chequeo Termodinámico (Detailed Balance Check)
*   **Micro-reversibilidad**: Cerca del equilibrio termodinámico (sobresaturación $S \approx 0$), las tasas globales de adsorción y desorción deben ser comparables. El sistema monitorea si existe una discrepancia de órdenes de magnitud injustificada entre $W_{ads}$ y $W_{des}$ en esta región, lo cual indicaría una violación de las leyes de la termodinámica (como la creación de energía libre de la nada).

## Flujo de Ejecución

El flujo típico de una simulación implica:
1.  Instancia de `LatticeSOS` e inicialización de la superficie (plana o rugosa).
2.  Definición de `KMCParams` con las condiciones físicas del experimento.
3.  Inicialización de `KMC_BKL` inyectando la red y los parámetros.
4.  Ejecución del bucle principal mediante `run()`, que itera sobre los pasos de Monte Carlo hasta alcanzar el tiempo final, registrando la historia de eventos y generando "snapshots" de la superficie.
