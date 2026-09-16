# notebooks - Guia de experimentos reproducibles

Esta carpeta contiene notebooks operativos para el flujo base, la linea v2 y comparaciones de sensibilidad.

## Inventario actual

- `Malaria002D_v1.ipynb`: pruebas del modelo alternativo 2D.
- `modern_kMC.ipynb`: exploracion del flujo kMC principal.
- `paper_kMC.ipynb`: experimentos de la linea paper/base historica.
- `paper_kMC_v2.ipynb`: experimentos con la linea v2 (`*_v2.py`).
- `experiments_1.ipynb` a `experiments_4.ipynb`: barridos de parametros y escenarios.
- `experiments_v2.ipynb`: baterias de experimentos sobre la formulacion v2.

## Archivos de apoyo en notebooks/

- `beta-hematina.txt`: referencia tabular usada en comparaciones.
- `adsorption_probabilities_vs_sigma_3.png`
- `adsorption_probabilities_vs_sigma_4.png`
- `adsorption_probabilities_vs_sigma_5.png`

## Orden sugerido de ejecucion

1. Activar entorno y dependencias del proyecto.
2. Ejecutar imports y configuracion de rutas.
3. Inicializar parametros/red para la linea elegida (base o v2).
4. Ejecutar simulaciones por bloques (tiempo o eventos).
5. Guardar graficas y GIF en `results/`.

## Buenas practicas

- Fijar semillas para reproducibilidad.
- No mezclar en una misma sesion experimentos de base y v2 sin reiniciar kernel.
- Registrar en markdown del notebook los parametros exactos de corrida.
- Exportar resultados finales con nombres trazables por experimento y tamano de red.
