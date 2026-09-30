# archive/ — scripts archivados (decisión D7)

Copias **sin modificar** de dos scripts de `src/` que dependen exclusivamente de la
línea `_v2` (`bkl_v2.py`, `lattice_v2.py`, `params_v2.py`), descartada en
`.descartables/scripts/versions/`. No forman parte del paquete refactorizado y **no
se pueden ejecutar tal cual**.

| Archivo | Qué hacía | Por qué se archiva |
|---|---|---|
| `growthRate.py` | Reproducción tipo Fig. 7 del paper (tasa de crecimiento vs σ por cara, con `FACE_DATA`). 185 de 477 líneas comentadas; ruta de Windows escrita a mano | Solo usa `_v2`. Portarlo al motor estático exige antes resolver el factor 2 de δ (`Plan_Paper.md`) |
| `probabilityAnalysis.py` | `compute_probabilities` / `extract_probs`: probabilidades de adsorción por clase de sitio vs σ (tipo Fig. 8) | Solo usa `_v2`. Su único consumidor era `paper_kMC_v2.ipynb` (descartado) |

Si en el futuro se quieren recuperar, el camino es reescribirlos sobre
`src.static.KMC_BKL_Static` (isotrópico: `E_pb_over_kT_x == E_pb_over_kT_y`,
`delta_x == delta_y`) y `src.common.reference_data.FACE_DATA`, y usar
`Plotter.plot_growth_rate_analysis` / `plot_adsorption_probabilities`.
