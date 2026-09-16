# data - Convenciones de datos

La carpeta `data/` esta reservada para datasets de entrada curados del proyecto.

## Estado actual

- `data/` se encuentra vacia (solo estructura de directorio).
- Existen datos de referencia usados en notebooks dentro de `notebooks/` (por ejemplo `beta-hematina.txt`).

## Que tipos de datos deben ir en `data/`

- Tablas experimentales de referencia.
- Series temporales de validacion.
- Insumos para calibracion de parametros.

## Reglas de formato recomendadas

- Preferir CSV o TXT tabular con encabezados claros.
- Incluir unidades en nombres de columnas cuando aplique.
- Evitar binarios sin descripcion o sin script consumidor asociado.

## Trazabilidad minima por dataset

Para cada dataset nuevo, registrar en documentacion:

- Fuente.
- Fecha de descarga/generacion.
- Preprocesamiento aplicado.
- Notebook o script que lo consume.