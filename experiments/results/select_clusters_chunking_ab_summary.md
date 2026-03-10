# AB Testing: `select_clusters` chunked (B) vs full-matrix/anterior (A)

## Setup

- A (anterior): `batch_size=None` (matriz global de coincidencias).
- B (actual): `batch_size=2048` (procesamiento por bloques).
- Dataset sintético: columnas=8, reglas=120, tamaños de filas crecientes.
- Métricas: tiempo total (s) y memoria pico (MB, `tracemalloc`).

## Resultados

| n_rows | tiempo A (s) | tiempo B (s) | Δ tiempo B vs A | memoria A (MB) | memoria B (MB) | Δ memoria B vs A |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.113133 | 0.163114 | +44.18% | 4.677 | 4.676 | -0.01% |
| 10000 | 0.308820 | 0.234517 | -24.06% | 23.238 | 6.262 | -73.05% |
| 30000 | 0.964391 | 0.643455 | -33.28% | 69.632 | 9.168 | -86.83% |

## Conclusión

- B (chunked) es mejor globalmente: gana en memoria siempre y en tiempo para tamaños medianos/grandes.
- Promedio Δ tiempo: -4.39% (negativo favorece B).
- Promedio Δ memoria: -53.30% (negativo favorece B).
- Recomendación: usar B por defecto para cargas reales; mantener A (`batch_size=None`) para escenarios muy pequeños si fuera necesario.

Fuente de datos: `experiments/results/select_clusters_chunking_benchmark.csv`.