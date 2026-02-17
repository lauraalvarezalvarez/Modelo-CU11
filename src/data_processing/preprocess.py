# Paso de video a frames y guardado en directorio especificado
import pandas as pd
import os
import shutil
import yaml
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tqdm import tqdm
from .load_data import load_raw_annotations

# --- FUNCIONES AUXILIARES ---
def convert_bbox_to_yolo(x1, y1, x2, y2):
    """
    Convierte BBox formato AVA (normalizado x1,y1,x2,y2) 
    a formato YOLO (x_center, y_center, width, height).
    """
    # Asumimos que x1, y1, x2, y2 ya vienen normalizados (0-1) en el CSV de AVA.
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + (width / 2)
    y_center = y1 + (height / 2)
    return x_center, y_center, width, height


# --- FUNCTION RAW TO PROCESSED ---
def clean_and_format_data(config):
    """
    Lee raw, convierte a formato YOLO y guarda todo junto en processed/
    """
    root_path = config['root_path']
    raw_frames_dir = root_path / config['paths']['raw_path'] / config['paths']['folders']['labelframes']
    # Ajusta esta ruta si tu CSV está en otra subcarpeta, según tu config actual
    ann_file = root_path / config['paths']['raw_path'] / config['paths']['folders']['annotations'] / "ava_train_v2.1.csv"

    processed_dir = Path(config['paths']['processed_path'])

    # Crear carpetas unificadas
    img_dir = processed_dir / "all_images"
    lbl_dir = processed_dir / "all_labels"

    if processed_dir.exists():
        # Opcional: Limpiar processed si quieres empezar de cero siempre
        # shutil.rmtree(processed_dir) 
        pass

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- ETAPA 1: Estandarizando datos en {processed_dir} ---")

    # Leer CSV
    df = load_raw_annotations(config)
    if df is None:
        return None

    # Filtrar clases
    class_mapping = config['dataset']['class_mapping']
    df = df[df['action_id'].isin(class_mapping.keys())].copy()
    df['class_id'] = df['action_id'].map(class_mapping)

    # Nombre de archivo estandarizado (relleno de 5 ceros)
    df['filename'] = df['video_id'] + "_" + df['timestamp'].astype(str).str.zfill(5) + ".jpg"

    unique_images = df['filename'].unique()
    print(f" Total imágenes únicas encontradas: {len(unique_images)}")

    count = 0
    # Procesar todo a la carpeta unificada
    for img_name in tqdm(unique_images, desc="Formateando a YOLO"):
        src = raw_frames_dir / img_name

        # Solo procesamos si la imagen física existe
        if not src.exists():
            continue

        # 1. Copiar imagen (Si ya existe, shutil.copy la sobrescribe, lo cual está bien)
        shutil.copy(src, img_dir / img_name)

        # 2. Generar TXT
        label_name = img_name.replace('.jpg', '.txt')
        subset = df[df['filename'] == img_name]

        with open(lbl_dir / label_name, 'w') as f:
            for _, row in subset.iterrows():
                xc, yc, w, h = convert_bbox_to_yolo(row['x1'], row['y1'], row['x2'], row['y2'])
                f.write(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        count += 1

    print(f" Etapa 1 completada. {count} imágenes listas en 'processed/'.")
    return unique_images # Devolvemos lista para usar en el split

def run_preprocessing(config):
    clean_and_format_data(config)
