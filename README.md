# Optimización de Flujo Vehicular en la M-30 mediante Velocidad Variable 🚗📉

> *Aplicación de Ciencia de Datos y Gemelos Digitales para la mitigación del "Efecto Acordeón" en el tráfico de Madrid.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Data Science](https://img.shields.io/badge/Focus-Data%20Science-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Descripción del Proyecto

Este proyecto aborda la problemática de los atascos en la autopista de circunvalación **M-30 de Madrid**, específicamente en el **Arco Este (Ventas - O'Donnell)**.

El objetivo principal es desarrollar un sistema de software que, mediante el análisis de datos históricos y algoritmos de **Machine Learning**, calcule una **Velocidad Límite Dinámica (Variable Speed Limit)** óptima. Esta velocidad varía a lo largo del día para armonizar el flujo, maximizar el caudal de vehículos (throughput) y evitar las ondas de choque (efecto acordeón) antes de que se produzcan.

El sistema incluye un **Gemelo Digital (Dashboard)** desarrollado en Python que permite simular y comparar visualmente el escenario real (histórico) frente al escenario optimizado por el algoritmo.

---

## 📂 Estructura del Repositorio actual

El proyecto sigue una arquitectura modular para asegurar la separación de responsabilidades entre la ingeniería de datos, la lógica científica y la interfaz de usuario.

```text
TFG_Trafico_Madrid/
│
├── data/                          # Almacenamiento de datos
│   ├── raw/                       # Datasets originales (datos.madrid.es)
│   ├── processed/                 # Datos limpios y estructurados
│   └── external/                  # Datos meteorológicos y metadatos de sensores
│
├── src/                           # Núcleo del procesamiento y lógica
│   ├── __init__.py
│   ├── config.py                  # Configuración global (IDs sensores M-30, rutas)
│   ├── data_loader.py             # Scripts de ingestión y descarga
│   ├── preprocessor.py            # Limpieza ETL e ingeniería de características
│   ├── physics.py                 # Diagrama Fundamental del Tráfico (Q = K * V)
│   ├── optimizer.py               # Algoritmo de decisión de velocidad óptima
│   └── models.py                  # Modelos ML (Random Forest/XGBoost)
│
├── simulation/                    # Módulo de simulación comparativa
│   ├── __init__.py
│   ├── engine.py                  # Motor de cálculo de métricas (A vs B)
│   └── scenarios.py               # Definición de escenarios de prueba
│
├── frontend/                      # Interfaz Visual (Gemelo Digital)
│   ├── app.py                     # Punto de entrada (Streamlit/Dash)
│   └── components/                # Gráficos y mapas interactivos
│
├── notebooks/                     # Jupyter Notebooks para experimentación (Sandbox)
├── requirements.txt               # Dependencias del proyecto
└── main.py                        # Script maestro de ejecución
