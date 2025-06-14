import numpy as np
import time
import os
import matplotlib.pyplot as plt

def leer_instancia_qap(filepath):
    """
    Lee un archivo de instancia de QAPLIB.
    Retorna el tamaño del problema (n), la matriz de distancias (D) y la matriz de flujos (F).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    parts = lines[0].strip().split()
    if not parts: # Archivo vacío o primera línea en blanco
        return 0, None, None
    n = int(parts[0])
    
    all_numbers = []
    # Empezar a leer desde la línea 1
    for line in lines[1:]:
        stripped_line = line.strip()
        if stripped_line: # Solo procesar líneas no vacías
            all_numbers.extend(map(int, stripped_line.split()))
            
    dist_values = all_numbers[:n*n]
    flow_values = all_numbers[n*n:(n*n)+(n*n)] # Asegurarse de no leer de más
    
    if len(dist_values) < n*n or len(flow_values) < n*n:
        print(f"Error: El archivo {filepath} no contiene suficientes datos para las matrices de tamaño {n}x{n}")
        return n, None, None

    dist_matrix = np.array(dist_values).reshape((n, n))
    flow_matrix = np.array(flow_values).reshape((n, n))
    
    return n, dist_matrix, flow_matrix

def leer_solucion_optima(filepath, nombre_instancia):
    """
    Lee el costo óptimo desde un archivo de solución .sln de QAPLIB.
    Si el archivo no existe, comprueba si es una de las instancias con óptimo conocido pero sin fichero.
    """
    try:
        with open(filepath, 'r') as f:
            line = f.readline()
            parts = line.strip().split()
            return int(parts[1])
    except FileNotFoundError:
        # Valores óptimos conocidos obtenidos de la web de QAPLIB, aunque no haya fichero .sln
        if nombre_instancia == "esc32h":
            print(f"Info: No se encontró {filepath}, pero se usará el óptimo conocido para esc32h (438).")
            return 438
        if nombre_instancia == "esc64a":
            print(f"Info: No se encontró {filepath}, pero se usará el óptimo conocido para esc64a (116).")
            return 116
        
        print(f"Advertencia: No se encontró el archivo de solución en {filepath}. Se procederá sin óptimo conocido.")
        return 0
    except (IndexError, ValueError):
        print(f"Advertencia: Error al parsear el archivo {filepath}. Se procederá sin óptimo conocido.")
        return 0

def calcular_costo_delta(solucion, dist_matrix, flow_matrix, i, j):
    """
    Calcula el cambio en el costo (delta) al intercambiar las posiciones i y j.
    Esta es una forma mucho más eficiente que recalcular todo el costo.
    """
    n = len(solucion)
    delta = 0
    pi_i, pi_j = solucion[i], solucion[j]

    for k in range(n):
        if k != i and k != j:
            # Flujo con i
            delta += (flow_matrix[k, i] - flow_matrix[k, j]) * (dist_matrix[solucion[k], pi_j] - dist_matrix[solucion[k], pi_i])
            # Flujo con j
            delta += (flow_matrix[i, k] - flow_matrix[j, k]) * (dist_matrix[pi_j, solucion[k]] - dist_matrix[pi_i, solucion[k]])
            
    # Sumar el término doble
    delta += (flow_matrix[i, j] - flow_matrix[j, i]) * (dist_matrix[pi_j, pi_i] - dist_matrix[pi_i, pi_j])
    return delta


def busqueda_tabu(n, dist_matrix, flow_matrix, max_evaluaciones, tenencia_tabu, optimo_conocido):
    """
    Implementación de la Búsqueda Tabú para el QAP con cálculo de costo delta.
    Retorna la solución, el costo y el historial de convergencia.
    """
    # 1. Generar solución inicial aleatoria
    solucion_actual = np.random.permutation(n)
    costo_actual = np.sum(flow_matrix * dist_matrix[np.ix_(solucion_actual, solucion_actual)])
    
    mejor_solucion = np.copy(solucion_actual)
    mejor_costo = costo_actual
    
    lista_tabu = np.zeros((n, n), dtype=int)
    
    evaluaciones = 0
    iteracion = 0
    
    # Historial para graficar convergencia
    historial_costos = [mejor_costo]
    historial_evals = [0]

    while evaluaciones < max_evaluaciones:
        mejor_vecino_delta = float('inf')
        mejor_vecino_swap = None
        
        # 3. Explorar el vecindario
        for i in range(n):
            for j in range(i + 1, n):
                delta = calcular_costo_delta(solucion_actual, dist_matrix, flow_matrix, i, j)
                evaluaciones += 1

                costo_vecino = costo_actual + delta
                es_tabu = lista_tabu[i, j] > iteracion
                criterio_aspiracion = costo_vecino < mejor_costo
                
                if (not es_tabu or criterio_aspiracion) and delta < mejor_vecino_delta:
                    mejor_vecino_delta = delta
                    mejor_vecino_swap = (i, j)

                if evaluaciones >= max_evaluaciones: break
            if evaluaciones >= max_evaluaciones: break
        
        if mejor_vecino_swap:
            i, j = mejor_vecino_swap
            solucion_actual[i], solucion_actual[j] = solucion_actual[j], solucion_actual[i]
            costo_actual += mejor_vecino_delta
            
            lista_tabu[j, i] = iteracion + tenencia_tabu
            
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
                mejor_solucion = np.copy(solucion_actual)
                
                # Guardar historial
                historial_costos.append(mejor_costo)
                historial_evals.append(evaluaciones)

                gap_str = "N/A"
                if optimo_conocido > 0:
                    gap = 100 * (mejor_costo - optimo_conocido) / optimo_conocido
                    gap_str = f"GAP: {gap:.2f}%"
                print(f"Iter {iteracion}: Nuevo mejor costo = {mejor_costo} ({gap_str}), Evals: {evaluaciones}/{max_evaluaciones}", end='\r')
        
        iteracion += 1
    
    print() # Nueva línea después de la barra de progreso
    return mejor_solucion, int(mejor_costo), historial_costos, historial_evals

if __name__ == "__main__":
    # --- Parámetros y Semilla de Reproducibilidad ---
    # Fijamos la semilla del generador de números aleatorios para asegurar que los resultados
    # sean reproducibles. Cada vez que se ejecute el script, la secuencia de soluciones
    # iniciales aleatorias será la misma, llevando a los mismos resultados finales.
    np.random.seed(42)

    INSTANCIAS_A_PROBAR = [
        "bur26c", "chr25a", "esc32h", "esc64a", 
        "lipa60a", "tai80a", "tai80b", "tho150"
    ]

    MAX_EVALUACIONES = 100000
    TENENCIA_TABU = 20 # Ajustar este valor puede ser parte del experimento
    NUM_EJECUCIONES = 11

    resultados_globales = []
    gaps_por_instancia = {} # Para el boxplot

    # Crear directorio para gráficos si no existe
    graphs_dir = os.path.join("results", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    for nombre_instancia in INSTANCIAS_A_PROBAR:
        ruta_instancia = os.path.join("instances", f"{nombre_instancia}.dat")
        ruta_solucion = os.path.join("solutions", f"{nombre_instancia}.sln")

        # --- Cargar Instancia y Solución Óptima ---
        optimo_conocido = leer_solucion_optima(ruta_solucion, nombre_instancia)
        n, dist, flow = leer_instancia_qap(ruta_instancia)
        
        if dist is None or flow is None:
            print(f"Saltando instancia {nombre_instancia} por error de lectura.")
            continue
            
        print(f"\n{'='*20} PROBANDO INSTANCIA: {nombre_instancia.upper()} (n={n}, optimo={optimo_conocido}) {'='*20}")

        costos_ejecucion = []
        tiempos_ejecucion = []

        # --- Gráfico de Convergencia para la Instancia ---
        plt.figure(figsize=(12, 8))
        
        for i in range(NUM_EJECUCIONES):
            print(f"--- Ejecución {i+1}/{NUM_EJECUCIONES} ---")
            start_time = time.time()
            
            _, costo_final, hist_costos, hist_evals = busqueda_tabu(n, dist, flow, MAX_EVALUACIONES, TENENCIA_TABU, optimo_conocido)
            
            end_time = time.time()
            tiempo_total = end_time - start_time
            
            costos_ejecucion.append(costo_final)
            tiempos_ejecucion.append(tiempo_total)
            
            # Añadir la convergencia de esta ejecución al gráfico
            plt.plot(hist_evals, hist_costos, alpha=0.6)
            
            print(f"Ejecución {i+1} finalizada. Costo: {costo_final}, Tiempo: {tiempo_total:.2f}s")

        # --- Finalizar y guardar gráfico de convergencia ---
        if optimo_conocido > 0:
            plt.axhline(y=optimo_conocido, color='r', linestyle='--', label=f'Óptimo Conocido ({optimo_conocido})')
        plt.title(f'Convergencia de la Búsqueda para {nombre_instancia.upper()}', fontsize=16)
        plt.xlabel('Evaluaciones de la Función Objetivo', fontsize=12)
        plt.ylabel('Mejor Costo Encontrado', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.yscale('log') # Escala logarítmica para ver mejor las mejoras iniciales
        convergence_plot_path = os.path.join(graphs_dir, f"{nombre_instancia}_convergence.png")
        plt.savefig(convergence_plot_path)
        plt.close()
        print(f"Gráfico de convergencia guardado en: {convergence_plot_path}")

        # --- Calcular estadísticas para la instancia ---
        mejor_costo = min(costos_ejecucion)
        peor_costo = max(costos_ejecucion)
        avg_costo = np.mean(costos_ejecucion)
        std_costo = np.std(costos_ejecucion)
        avg_tiempo = np.mean(tiempos_ejecucion)
        
        optimo_str = str(optimo_conocido)
        mejor_gap_str = "N/A"
        avg_gap_str = "N/A"

        if optimo_conocido > 0:
            gaps_ejecucion = [100 * (c - optimo_conocido) / optimo_conocido for c in costos_ejecucion]
            gaps_por_instancia[nombre_instancia] = gaps_ejecucion
            mejor_gap = min(gaps_ejecucion)
            avg_gap = np.mean(gaps_ejecucion)
            mejor_gap_str = f"{mejor_gap:.2f}%"
            avg_gap_str = f"{avg_gap:.2f}%"
        else:
            optimo_str = "N/A"
        
        resultados_globales.append([
            nombre_instancia, n, optimo_str, mejor_costo, f"{avg_costo:.2f}",
            f"{std_costo:.2f}", mejor_gap_str, avg_gap_str, f"{avg_tiempo:.2f}s"
        ])

        # Guardar costos por ejecución para cada instancia
        if 'per_instance_costs' not in globals():
            global per_instance_costs
            per_instance_costs = {}
        per_instance_costs[nombre_instancia] = costos_ejecucion

    # Guardar todos los costos por ejecución en un archivo JSON
    import json
    per_iteration_output_filepath = os.path.join("results", "per_iteration_costs.json")
    try:
        with open(per_iteration_output_filepath, 'w') as f:
            json.dump(per_instance_costs, f, indent=4)
        print(f"\nCostos por iteración guardados exitosamente en: {per_iteration_output_filepath}")
    except IOError as e:
        print(f"\nError al guardar el archivo de costos por iteración: {e}")

    # --- Generar Boxplot Global de GAPs ---
    if gaps_por_instancia:
        labels = list(gaps_por_instancia.keys())
        data = list(gaps_por_instancia.values())
        
        plt.figure(figsize=(15, 8))
        plt.boxplot(data, labels=labels)
        plt.title('Distribución del %GAP por Instancia (11 ejecuciones)', fontsize=16)
        plt.ylabel('% GAP', fontsize=12)
        plt.xlabel('Instancia', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        boxplot_path = os.path.join(graphs_dir, "global_gap_boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"\nBoxplot global de GAPs guardado en: {boxplot_path}")

    # --- Imprimir y Guardar Tabla de Resultados Globales ---
    print("\n\n" + "="*80)
    print("TABLA DE RESULTADOS GLOBALES")
    print("="*80)
    
    headers = [
        "Instancia", "n", "Óptimo", "Mejor Costo", "Costo Prom", 
        "Costo StdDev", "Mejor GAP", "GAP Prom", "Tiempo Prom"
    ]
    
    header_format = "{:<12} {:<4} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10} {:<10}"
    row_format = "{:<12} {:<4} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10} {:<10}"
    
    table_content = []
    table_content.append(header_format.format(*headers))
    table_content.append("-" * 110)
    for row in resultados_globales:
        table_content.append(row_format.format(*row))
    
    for line in table_content:
        print(line)

    print("="*80)

    output_filepath = os.path.join("results", "summary.txt")
    try:
        with open(output_filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TABLA DE RESULTADOS GLOBALES\n")
            f.write("="*80 + "\n")
            for line in table_content:
                f.write(line + "\n")
            f.write("="*80 + "\n")
        print(f"\nResultados guardados exitosamente en: {output_filepath}")
    except IOError as e:
        print(f"\nError al guardar el archivo de resultados: {e}") 