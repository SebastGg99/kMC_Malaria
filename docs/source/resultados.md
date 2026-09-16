# results - Artefactos de salida

Este directorio concentra salidas graficas generadas por simulaciones y notebooks.

## Estado actual (artefactos observados)

### GIF de crecimiento

- `crystal_growth.gif`
- `crystal_growth_75.gif`
- `crystal_growth_NoDes.gif`
- `crystal_growth_NoDesNoMig.gif`
- `crystal_growth_NoMig.gif`
- `crystal_growth_S2.gif`
- `crystal_growth_S2_1.gif`
- `crystal_growth_[20, 20].gif`
- `crystal_growth_[30, 30].gif`
- `crystal_growth_[50, 50].gif`
- `crystal_growth_[100, 100].gif`

### Graficas de probabilidad de adsorcion

- `adsorption_probabilities_vs_sigma_1.png`
- `adsorption_probabilities_vs_sigma_2.png`
- `adsorption_probabilities_vs_sigma_3.png`
- `adsorption_probabilities_vs_sigma_5.png`

## Lectura tecnica de los resultados

- Los archivos `NoDes`, `NoMig` y `NoDesNoMig` corresponden a variantes con procesos desactivados.
- Los archivos con sufijo `[Lx, Ly]` corresponden a pruebas de escalado por tamano de red.
- Los PNG de probabilidad resumen barridos de sigma y clases de sitio de adsorcion.

## Convencion recomendada de nombres

Usar una estructura trazable por experimento:

`<experimento>__LxLy__sigma__seed__fecha.ext`

Ejemplo:

`growth_face101__30x30__sigma2.0__seed42__2026-03-10.gif`

## Politica de versionado

- Versionar resultados finales que respalden reportes, figuras de paper o validaciones clave.
- Evitar versionar salidas intermedias voluminosas cuando no aporten evidencia adicional.
