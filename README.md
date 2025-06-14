# Trabajo Final: Resolución del Problema de Asignación Cuadrática mediante Búsqueda Tabú

## Información General

*   **Asignatura:** Algoritmos Avanzados y Complejidad Computacional
*   **Título del Trabajo:** Trabajo Final Unidad II
*   **Autores:**
    *   Camila Llamirez
    *   Pablo Jordan
    *   Camilo Avendaño
*   **Profesor:** Mario Inostroza
*   **Fecha:** 13 de Junio de 2025

---

## Descripción del Proyecto

Este proyecto presenta una implementación de la metaheurística de **Búsqueda Tabú (Tabu Search)** para resolver el **Problema de Asignación Cuadrática (QAP)**, uno de los problemas de optimización combinatoria NP-hard más conocidos.

El objetivo es encontrar una asignación de instalaciones a localizaciones que minimice el costo total, el cual se calcula en función de los flujos entre instalaciones y las distancias entre localizaciones. La implementación utiliza una evaluación incremental de la función objetivo (cálculo delta) para explorar eficientemente el espacio de búsqueda. Se realizan múltiples ejecuciones por instancia para analizar la robustez y consistencia del algoritmo.

Los resultados detallados, incluyendo tablas de rendimiento, gráficos de convergencia y un análisis exhaustivo, se presentan en un informe técnico y en archivos de datos generados.

---

## Cómo Ejecutar

### Requisitos

*   Python 3.8 o superior.
*   Bibliotecas de Python:
    *   `numpy` (para operaciones numéricas)
    *   `matplotlib` (para la generación de gráficos)
    *   `tabulate` (para la creación de tablas en formato Markdown)

Puedes instalar las dependencias usando pip:
```bash
pip install numpy matplotlib tabulate
```

### Ejecución del Script Principal

El script principal `main.py` ejecuta la Búsqueda Tabú para todas las instancias definidas, genera los archivos de resultados y los gráficos de convergencia.

Para ejecutar el script, navega al directorio raíz del proyecto (`qap/`) y ejecuta:
```bash
python main.py
```

---

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```
qap/
├── informe/                    # Documentación y reporte técnico
│   ├── b.tex                 # Código fuente del informe en LaTeX
│   └── b.pdf                 # Informe final compilado en PDF
│
├── instances/                  # Instancias del problema QAP (formato QAPLIB)
│   └── *.dat
│
├── results/                    # Resultados generados por la ejecución del script
│   ├── graphs/                 # Gráficos de convergencia y boxplots
│   │   ├── *.png               # Gráficos de convergencia por instancia
│   │   └── global_gap_boxplot.png # Boxplot comparativo de GAPs
│   ├── per_iteration_costs.json # Costos detallados por iteración para cada ejecución
│   └── summary.txt             # Tabla resumen con métricas agregadas por instancia
│
├── solutions/                  # Soluciones óptimas conocidas (formato QAPLIB)
│   └── *.sln
│
├── main.py                     # Script principal para ejecutar el experimento
├── tabu_search.py              # Implementación del algoritmo de Búsqueda Tabú
├── utils.py                    # Funciones de utilidad (lectura de datos, cálculos, etc.)
├── resultadosporiteracion.md   # Tabla Markdown con los costos de las 11 ejecuciones por instancia
├── grupo2_mejores_soluciones_promedio_desv.md # Tabla Markdown con mejor solución, promedio y desv. estándar por instancia
└── README.md                   # Este archivo
```

---

## Resultados y Análisis

Los resultados de la experimentación se almacenan en varios formatos:

*   **`results/summary.txt`**: Contiene una tabla consolidada con el mejor costo, costo promedio, desviación estándar, GAPs y tiempo promedio de ejecución para cada instancia.
*   **`results/per_iteration_costs.json`**: Almacena los costos finales de cada una de las 11 ejecuciones para cada instancia, útil para análisis de consistencia más detallados.
*   **`results/graphs/`**: Incluye gráficos de convergencia para cada instancia (`<instancia>_convergence.png`) que muestran cómo evoluciona el costo de la mejor solución encontrada a lo largo de las evaluaciones de la función objetivo para las 11 ejecuciones. También contiene un boxplot (`global_gap_boxplot.png`) que compara la distribución de los GAPs obtenidos para todas las instancias.
*   **`resultadosporiteracion.md`**: Presenta una tabla en formato Markdown con los costos finales de las 11 ejecuciones para cada instancia, facilitando la visualización rápida.
*   **`grupo2_mejores_soluciones_promedio_desv.md`**: Ofrece una tabla Markdown con un resumen de la mejor solución, el costo promedio y la desviación estándar para cada instancia, extraído de `summary.txt`.

Para un análisis completo y discusión de estos resultados, incluyendo la metodología, el diseño experimental y las conclusiones detalladas, por favor consulta el informe técnico completo:

*   **Informe Completo:** `informe/b.pdf`

---
