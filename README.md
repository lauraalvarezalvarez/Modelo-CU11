# 🐄 CU11 - Detección de Anomalías de Comportamiento en Ganado

Sistema de IA para la identificación prematura de enfermedades mediante el análisis de patrones de conducta y datos de sensores IoT en tiempo real.

## 🎯 Objetivo
Identificar cambios sutiles en la actividad (rumia, movimiento, descanso) para reducir las pérdidas de producto en un 10% antes de que los síntomas sean visibles.

## 📊 Datos utilizados
- Sensores IoT (Acelerómetros/IMU para monitorear actividad).
- Sensores de temperatura interna (bolos ruminales) y ambiente.
- Vídeo para análisis de postura y patrones de marcha.
- Dataset: MMCOWS (Multimodal Dairy Cattle Dataset).

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
