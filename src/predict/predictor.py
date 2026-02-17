from ultralytics import YOLO
import cv2
from pathlib import Path
from collections import defaultdict
import time

def load_model(model_path):
    """
    Carga el modelo YOLOv8 desde la ruta especificada.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f" Error: No existe el modelo en {model_path}")

    print(f" Cargando modelo: {path.name}")
    return YOLO(str(path))

def run_inference(model, source_video, output_video_path):
    """
    Ejecuta el tracking sobre el video y devuelve los datos crudos de comportamiento.
    Args:
        model: Modelo YOLO cargado.
        source_video: Ruta al video de entrada.
        output_video_path: Ruta donde guardar el video anotado.

    Returns:
        y_pred (tuple): Contiene (behavior_data, fps, class_names)
    """
    # Configurar Video
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise IOError(f" No se pudo abrir el video: {source_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Writer para guardar el video procesado
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # Estructuras para Datos (Raw Data)
    # Diccionario: {track_id: {class_id: frames_count}}
    behavior_data = defaultdict(lambda: defaultdict(int))
    class_names = model.names

    frame_count = 0
    print(f" Iniciando inferencia en: {Path(source_video).name}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # --- TRACKING ---
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            # Obtener datos
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            # Dibujar cajas
            annotated_frame = results[0].plot()

            # Acumular frames por acción
            for track_id, cls_id in zip(track_ids, cls_ids):
                behavior_data[track_id][cls_id] += 1

        else:
            annotated_frame = frame

        video_writer.write(annotated_frame)

        if frame_count % 50 == 0:
            print(f"   Procesando frame {frame_count}...", end='\r')

    cap.release()
    video_writer.release()
    print(f"\n Video guardado en: {output_video_path}")

    # Retornamos los datos crudos necesarios para el post-procesado
    return behavior_data, fps, class_names

def save_predictions(df_pred, out_path):
    """
    Guarda el DataFrame final en un CSV.
    """
    df_pred.to_csv(out_path, index=False)
    print(f" Predicciones guardadas en: {out_path}")