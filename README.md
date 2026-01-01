# SOLUCIONM30
## FASE 1: Definición y Adquisición de Datos (El Terreno)
Antes de nada, acotamos los datos crudos que alimentarán el sistema.
· Objetivo: Obtener los datasets brutos de la zona M-30 "Arco Este".
· Fuente: Portal de Datos Abiertos del Ayuntamiento de Madrid.
· Periodo: 2018-2019 (Entrenamiento y Test "limpio" pre-pandemia).
· Tareas:
  - Mapeo de Sensores: Identificar los IDs de las espiras electromagnéticas entre el Nudo de Manoteras y el Nudo Sur.
  - Descarga Masiva: Automatizar la descarga de los CSVs mensuales.
  - Enriquecimiento: Descargar datos meteorológicos (AEMET/OpenWeather) para esas fechas.
## FASE 2: ETL y Preprocesamiento (La Refinería - KNIME)
Transformar datos sucios en información útil. Esta será la parte más densa en KNIME.
· Herramienta: KNIME Analytics Platform.
· Tareas:
  - Filtrado Espacial: Descartar todos los sensores que no sean de nuestra lista de la M-30.
  - Limpieza de Ruido: Eliminar registros con errores de sensor (ej: Velocidad=0 pero Intensidad>0). Imputación de valores perdidos (si falta un dato de 15 min, interpolar con el anterior y posterior).
  - Feature Engineering (Creación de variables):
      - Crear columna Día_Semana y Es_Festivo.
      - Crear columna Lluvia_Binaria (0/1).
      - Crucial: Crear variables de lag (retardo).
      - Ejemplo: Densidad_Manoteras_Hace_10min.
## FASE 3: El Motor Inteligente (Análisis y Cálculo - KNIME/Python)
Aquí es donde aplicamos la ciencia para calcular la "Velocidad Óptima".
· Herramienta: KNIME (con integración de Python Script si hace falta).
· Sub-fase 3.1: El Modelo Físico (Diagrama Fundamental):
  - Graficar Intensidad vs. Densidad con los datos reales.
  - Determinar matemáticamente el Punto Crítico de Colapso ($K_{crit}$) para ese tramo (ej: 45 veh/km).
· Sub-fase 3.2: El Modelo Predictivo (ML):
  - Entrenar algoritmo (Random Forest o XGBoost) para predecir la densidad futura ($t+15min$).
· Sub-fase 3.3: El Algoritmo de Optimización:
  - Aplicar la lógica de negocio: Si Predicción > Umbral Crítico $\rightarrow$ Bajar Límite a 70 km/h. Si Predicción < Umbral Crítico $\rightarrow$ Mantener 90 km/h.
  - Output de esta fase: Un dataset nuevo con dos columnas clave: Velocidad_Real_Registrada y Velocidad_Optima_Calculada.
## FASE 4: La Simulación Visual (El Frontend - Python)
Aquí entra tu "plantilla de frontend". Haremos una Simulación Macroscópica Visual.
· Herramienta: Python. Usar bibliotecas como Streamlit (muy rápido para dashboards), PyGame (si quieres ver "cochcecitos" moviéndose) o Matplotlib/Plotly animado.
· Concepto: Pantalla dividida.
  - Izquierda (Realidad): Muestra el flujo de tráfico tal cual ocurrió en el histórico (atasco, coches rojos parados).
  - Derecha (Tu Solución): Muestra el mismo flujo pero aplicando tu velocidad calculada. Los coches irán más lento (70 km/h) pero no se pondrán en rojo (parados).
  - Métricas en tiempo real: Un panel al lado que muestre "Tiempo medio de viaje" actualizándose segundo a segundo en la simulación.
