from ultralytics import YOLO
from pathlib import Path
import torch
import sys

def train_yolo(config):
    """
    Ejecuta el entrenamiento de YOLOv8 leyendo los datos desde data/splits/
    """
    root_path = config['root_path']
    # En data/splits están organizadas en train/val/test para YOLO.
    splits_path = Path(config['paths']['splits_path'])
    dataset_yaml = splits_path / "dataset.yaml"

    # Directorio de salida de modelos
    output_dir = root_path / "models" / "artifacts"

    print(f"--- TRAIN: Configurando YOLOv8 ---")
    print(f" Leyendo dataset desde: {dataset_yaml}")

    # Verificación de seguridad
    if not dataset_yaml.exists():
        raise FileNotFoundError(f" No se encuentra {dataset_yaml}. Asegúrate de que se han generado los splits.")

    # Detección de Hardware
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = 0
        print(f" GPU Activa: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(" AVISO: Usando CPU (El entrenamiento será lento)")

    # 1. Cargar Modelo Base
    # Usamos 'yolov8s.pt' (Small) como base. Puedes cambiar a 'n' (Nano) o 'm' (Medium).
    model = YOLO('yolov8s.pt') 

    # 2. Leer Hiperparámetros del Config
    epochs = config['training'].get('epochs', 50)
    batch_size = config['training'].get('batch_size', 16)

    # 3. Ejecutar Entrenamiento
    try:
        model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(output_dir), # Carpeta raíz de salida
            name="yolo_run",         # Subcarpeta del experimento
            exist_ok=True,           # Sobrescribir si existe
            pretrained=True,
            plots=True,
            device=device,
            workers=4                # Ajustar según tu CPU
        )
        print(f" Entrenamiento finalizado.")
        print(f" Modelo guardado en: {output_dir / 'yolo_run' / 'weights' / 'best.pt'}")

    except Exception as e:
        print(f" Error crítico durante el entrenamiento: {e}")
        raise e