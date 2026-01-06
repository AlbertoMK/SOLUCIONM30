# 🚦 Optimización de Tráfico M-30 - Manual de Usuario

Este documento te guiará paso a paso para configurar y ejecutar el sistema de optimización de tráfico desde cero, incluso sin conocimientos previos del proyecto.

---

## 📋 1. Requisitos Previos

Necesitas tener instalado en tu ordenador:
*   **Python 3.10 o superior**: [Descargar aquí](https://www.python.org/downloads/).
*   **Git** (Opcional, si vas a clonar el repositorio).

---

## 🛠️ 2. Instalación

1.  **Descarga o Clona** este repositorio en tu ordenador.
3.  **Instala las dependencias** (librerías necesarias) ejecutando:

```powershell
pip install -r requirements.txt
```

---

## 📂 3. Preparación de los Datos

### Colocar los archivos
Debes copiar tus archivos CSV (que obtienes del portal de datos de Madrid) manualmente en estas carpetas:

1.  **Datos de Tráfico (Mes Enero 2019):**
    *   Coge tu archivo CSV de tráfico.
    *   Renómbralo a **`01-2019.csv`**.
    *   Crea la carpeta `01-2019 en data\raw\trafico`
    *   Mételo en: `data\raw\trafico\01-2019\`

2.  **Metadatos (Ubicación de Sensores):**
    *   Coge tu archivo de puntos de medida.
    *   Renómbralo a **`pmed_ubicacion_10_2018.csv`**.
    *   Mételo en: `data\raw\meta\`

> **Nota:** Si quieres visualizar otros meses, deberás crear su carpeta correspondiente (ej. `02-2019`) y colocar su archivo csv dentro.

---

## ⚙️ 4. Configuración Inicial

Antes de abrir la aplicación visual, debemos calcular los límites de velocidad de cada tramo basándonos en el histórico. Esto genera un archivo necesario para la simulación.

Ejecuta el script de calibración:

```powershell
python -m src.calibrate_limits
```

**Deberías ver:** Un mensaje de "✅ Resultados guardados en: .../sensor_limits.csv". 
Este paso solo es necesario hacerlo **una vez** (o cuando añadas nuevos datos).

---

## 🧪 5. Verificación (Script Principal)

Antes de lanzar el dashboard, es recomendable ejecutar el script principal para verificar que todo el sistema (carga de datos, física del tráfico y optimizador) funciona correctamente en consola.

Ejecuta:

```powershell
python -m main
```

**Deberías ver:** Un resumen del proceso, incluyendo la "Densidad Crítica" detectada, la velocidad media real vs simulada y el porcentaje de mejora. Si esto funciona sin errores, tu instalación es correcta.

---

## 🖥️ 6. Ejecutar la Aplicación

Ahora ya puedes iniciar el panel de control visual para ver la simulación.

Ejecuta:

```powershell
python -m streamlit run frontend/app.py
```

*   Se abrirá automáticamente una pestaña en tu navegador (normalmente en `http://localhost:8501`).
*   **Si no se abre**, copia esa dirección y pégala en Chrome/Edge/Firefox.

---

## 🕹️ Guía de Uso del Dashboard

Una vez en la web:
1.  **Selecciona una Fecha:** Usa el menú lateral para elegir el día que quieres analizar.
2.  **Elige un Sensor:** Haz clic en un punto del mapa o selecciona uno del desplegable en la barra lateral.
    *   *Rojo:* Sensor seleccionado.
    *   *Azul:* Otros sensores disponibles en la M-30.
3.  **Dale al Play:** Pulsa el botón `▶️ START` en el centro de la pantalla.
4.  **Observa:**
    *   **Izquierda (Reality):** Muestra qué pasó realmente ese día.
    *   **Derecha (Digital Twin):** Muestra qué habría pasado si el sistema de límites dinámicos hubiera estado activo.
    *   **Velocidad/Densidad:** Compara cómo mejoran los indicadores.

---

## ❓ Solución de Problemas Frecuentes

*   **Error `FileNotFoundError` o "Sample file not found":**
    *   Casi seguro que los archivos en `data/raw` no están bien colocados o nombrados. Revisa el **Paso 3**.
*   **El mapa sale vacío:**
    *   Falta el archivo de metadatos en `data/raw/meta` o no tiene el formato correcto (separador `;`).
*   **Error `ModuleNotFoundError`:**
    *   No has instalado las dependencias. Repite el comando `pip install -r requirements.txt`.
