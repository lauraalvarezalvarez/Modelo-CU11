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

# ===== LÓGICA PARA LA PREDICCIÓN =====
# def predict():
#     """
#     Ejecuta inferencia sobre un video, realiza tracking y genera estadísticas.
#     """
#     config = load_config()
#     root_path = config['root_path']

#     # 1. Definir rutas de Entrada y Salida
#     artifacts_dir = root_path / "models" / "artifacts" / "yolo_run" / "weights"
#     model_path = artifacts_dir / "best.pt"

#     # Busca un video en data/raw/videos (toma el primero que encuentre)
#     videos_dir = root_path / config['paths']['raw_path'] / config['paths']['folders']['videos']
#     video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))

#     if not video_files:
#         print(f" Error: No se encontraron videos en {videos_dir}")
#         return

#     source_video = video_files[8] # Usamos el segundo video encontrado

#     # Carpetas de salida
#     predictions_dir = root_path / config['paths']['predictions_path']
#     output_video_path = predictions_dir / f"pred_{source_video.name}"
#     stats_csv_path = predictions_dir / f"stats_{source_video.stem}.csv"

#     print(f"--- PREDICT: Iniciando Inferencia ---")
#     print(f"Modelo: {model_path}")
#     print(f"Video fuente: {source_video}")
#     print(f"Salida video: {output_video_path}")
#     print(f"Salida datos: {stats_csv_path}")

#     # 2. Cargar Modelo Entrenado
#     if not model_path.exists():
#         print(" Error: No existe el modelo best.pt. Ejecuta train() primero.")
#         return

#     model = YOLO(str(model_path))

#     # 3. Configurar Video
#     cap = cv2.VideoCapture(str(source_video))
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#     # Writer para guardar el video procesado
#     video_writer = cv2.VideoWriter(
#         str(output_video_path),
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         fps, (w, h)
#     )

#     # 4. Estructuras para Datos de Comportamiento
#     # Diccionario: {track_id: {class_id: frames_count}}
#     behavior_data = defaultdict(lambda: defaultdict(int))
#     class_names = model.names # {0: 'standing', 1: 'lying', ...}

#     # 5. Bucle de Inferencia (Frame a Frame)
#     frame_count = 0
#     start_time = time.time()

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         frame_count += 1

#         # --- TRACKING CON YOLOv8 ---
#         # persist=True es VITAL para el tracking (mantiene memoria entre frames)
#         results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

#         if results[0].boxes.id is not None:
#             # Obtener datos: cajas, ids de track, y clases
#             boxes = results[0].boxes.xyxy.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             cls_ids = results[0].boxes.cls.int().cpu().tolist()

#             # Anotador visual
#             annotated_frame = results[0].plot()

#             # Lógica de Acumulación de Tiempo
#             for track_id, cls_id in zip(track_ids, cls_ids):
#                 behavior_data[track_id][cls_id] += 1

#                 # Calcular tiempo actual en segundos (frames / fps)
#                 seconds = behavior_data[track_id][cls_id] / fps

#                 # --- LÓGICA DE ALERTAS (ANOMALÍA SIMPLE) ---
#                 # Ejemplo: Si una vaca lleva > 5 segundos "standing" (solo demo)
#                 # En producción esto sería horas.
#                 action_name = class_names[cls_id]
#                 label = f"ID:{track_id} {action_name} {seconds:.1f}s"

#                 # Dibujar info extra en el video
#                 # (YOLO ya dibuja cajas, aquí podríamos añadir alertas personalizadas)

#         else:
#             annotated_frame = frame # Si no detecta nada, guarda el frame original

#         video_writer.write(annotated_frame)

#         if frame_count % 30 == 0:
#             print(f"Procesando frame {frame_count}...", end='\r')

#     # 6. Finalización y Guardado de Datos
#     cap.release()
#     video_writer.release()

#     # Exportar CSV final
#     # Convertimos frames a segundos/minutos para el reporte
#     rows = []
#     for tid, actions in behavior_data.items():
#         row = {'cow_id': tid}
#         total_frames = 0
#         for cid, count in actions.items():
#             action_name = class_names[cid]
#             row[f"{action_name}_sec"] = round(count / fps, 2)
#             total_frames += count
#         row['total_tracked_sec'] = round(total_frames / fps, 2)
#         rows.append(row)

#     df_stats = pd.DataFrame(rows)
#     df_stats.to_csv(stats_csv_path, index=False)

#     print(f"\n Procesamiento finalizado.")
#     print(f"Video guardado: {output_video_path}")
#     print(f"Reporte CSV: {stats_csv_path}")
#     print(df_stats.head())


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

if __name__ == "__main__":
    train()