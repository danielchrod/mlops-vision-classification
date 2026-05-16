# Proyecto Food Segmentation MLOps

Este proyecto implementa un pipeline completo de **MLOps** para la segmentación semántica de ingredientes alimentarios utilizando el dataset **FoodSeg103**. El sistema incluye desde el entrenamiento con trazabilidad hasta el despliegue de una API contenedorizada.

## 1. Arquitectura del Proyecto
El sistema se basa en una arquitectura de microservicios ligera, diseñada para la reproducibilidad y escalabilidad.

* **Modelo Core:** DeepLabV3+ con encoder ResNet50 (104 clases).
* **API:** Desarrollada con **FastAPI** para inferencia de baja latencia.
* **Contenedorización:** **Docker** (imagen basada en `python:3.9-slim`).
* **Trazabilidad:** Integración total con **Weights & Biases**.

## 2. Estructura de Carpetas
Siguiendo las mejores prácticas de ingeniería de software:
```text
.
├── data/               # Subconjunto de imágenes para validación local
├── models/             # Pesos del modelo entrenado (mejor_modelo_B.pth)
├── src/                # Código fuente modular
│   ├── inference_api.py # Endpoint FastAPI para predicciones
│   ├── train.py        # Script de entrenamiento modular
│   └── data_loader.py  # Gestión de datasets y transformaciones
├── tests/              # Pruebas unitarias y de integración
├── Dockerfile          # Receta para la imagen de producción
├── requirements.txt    # Dependencias fijas del proyecto
└── README.md           # Documentación principal