from ultralytics import YOLO
import yaml
from pathlib import Path
import os
import torch
import sys
import cv2
import pandas as pd
from collections import defaultdict
import time

# Stats
from src.get_stats.analysis import *
# Preprocessing
from src.data_processing.preprocess import *
# Training
from src.training.trainer import train_yolo
from src.training.utils_train import generate_splits
#Prediction
from src.predict.predictor import load_model, run_inference, save_predictions
from src.predict.postprocess import format_output


# Rutas
def load_config():
    root_path = Path(__file__).resolve().parent.parent
    config_path = root_path / "config" / "config.yaml"
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['root_path'] = root_path
    return config

# ===== LÓGICA PRINCIPAL DE PREPROCESAMIENTO =====
def process_dataset():
    """ Raw -> Processed  """
    config = load_config()
    run_preprocessing(config)

# ===== LÓGICA PRINCIPAL DE ENTRENAMIENTO =====
def train():
    config = load_config()
    generate_splits(config) # Genera splits (train/val/test) en data/splits
    train_yolo(config)     # Ejecuta entrenamiento de YOLOv8

def predict():
    """
    Ejecuta inferencia (Orquestador).
    """
    config = load_config()
    root_path = config['root_path']

    # 1. Definir rutas
    artifacts_dir = root_path / "models" / "artifacts" / "yolo_run" / "weights"
    model_path = artifacts_dir / "best.pt"

    videos_dir = root_path / config['paths']['raw_path'] / config['paths']['folders']['videos']
    predictions_dir = Path(config['paths']['predictions_path'])

    # Buscar videos
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
    if not video_files:
        print(f" Error: No se encontraron videos en {videos_dir}")
        return

    # Selección del video (Usamos índice 8 como indicaste en tu ejemplo, o fallback)
    try:
        source_video = video_files[8]
    except IndexError:
        source_video = video_files[0] # Fallback al primero si no hay tantos
        print(f" Aviso: No hay video índice 8, usando el primero: {source_video.name}")

    output_video_path = predictions_dir / f"pred_{source_video.name}"
    stats_csv_path = predictions_dir / f"stats_{source_video.stem}.csv"

    print(f"--- PREDICT: Iniciando Pipeline ---")

    # 2. Cargar Modelo (predictor.py)
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Ejecutar Inferencia (predictor.py)
    # Devuelve los datos crudos (y_pred)
    y_pred_raw = run_inference(model, source_video, output_video_path)

    # 4. Post-procesado y Formato (postprocess.py)
    # Convierte datos crudos a DataFrame
    df_results = format_output(y_pred_raw)

    # 5. Guardar Resultados (predictor.py)
    save_predictions(df_results, stats_csv_path)

    print("\n Proceso finalizado exitosamente.")
    print(df_results.head())

def get_stats():
    """
    Genera un reporte técnico del dataset:
    - Descripción de inputs/outputs (Features).
    - Conteo de instancias por clase y split.
    Guarda resultados en models/metrics/
    """
    config = load_config()
    generate_stats_report(config)
if __name__ == "__main__":
    train()