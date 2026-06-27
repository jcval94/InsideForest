# Resumen de AB Testing de Cuellos de Botella

## Alcance

Este reporte compara cambios candidatos en la ruta principal de `InsideForestRegionClusterer.fit`: extracción, priorización y asignación de regiones.

## Hallazgos

- `stage_get_fro`: vectorized pandas str.extract parser. Mejora: 27.38x. Salida equivalente: True. Decision: implemented.
- `stage_get_summary`: group rules once and reuse one boolean mask. Mejora: 1.57x. Salida equivalente: True. Decision: implemented.
- `stage_prio_ranges`: NumPy indexed replacement without tiled DataFrames. Mejora: 1.49x. Salida equivalente: True. Decision: implemented.
- `stage_menu_catalog`: rank by coverage first, then score. Mejora: no comparable; el original no produjo salida util. Salida equivalente: None. Decision: implemented.
- `full_fit_default`: combined implemented changes. Mejora: 1.12x. Salida equivalente: True. Decision: implemented.

## Cambios Implementados

- `stage_get_fro`: vectorized pandas str.extract parser; 27.38x en esta corrida.
- `stage_get_summary`: group rules once and reuse one boolean mask; 1.57x en esta corrida.
- `stage_prio_ranges`: NumPy indexed replacement without tiled DataFrames; 1.49x en esta corrida.
- `stage_menu_catalog`: rank by coverage first, then score; corrige un caso de no progreso del selector.
- `full_fit_default`: combined implemented changes; 1.12x en esta corrida.

## Verificacion

- Las etapas con salida original verifican igualdad exacta de DataFrames.
- La prueba de pipeline completo compara las etiquetas finales del camino original y el nuevo.
- El caso `menu` documenta una condicion de no progreso en el original y verifica que el candidato regresa etiquetas.

Fuente CSV: `experiments/results/bottleneck_ab_test.csv`.
