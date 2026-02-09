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

# def create_dataset_yaml(config, target_path):
#     """Genera el dataset.yaml en la carpeta de splits (donde entrena YOLO)."""
#     names_dict = {int(k): v for k, v in config['dataset']['class_names'].items()}

#     # Splits
#     yaml_content = {
#         'path': str(target_path), # Ruta absoluta a data/splits
#         'train': 'train/images',
#         'val': 'val/images',
#         'test': 'test/images',
#         'names': names_dict
#     }

#     yaml_out = target_path / 'dataset.yaml'
#     with open(yaml_out, 'w') as f:
#         yaml.dump(yaml_content, f, sort_keys=False)
#     print(f" Configuración YOLO generada en: {yaml_out}")

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

# --- FUNCTION SPLIT DATASET ---
# def create_splits(config, all_images_list):
#     """
#     Toma los datos de processed/ y los distribuye en splits/ (Train/Val/Test)
#     """
#     if all_images_list is None or len(all_images_list) == 0:
#         print(" No hay imágenes para dividir.")
#         return

#     splits_dir = Path(config['paths']['splits_path'])
#     processed_dir = Path(config['paths']['processed_path'])

#     # Limpiar splits anteriores para asegurar limpieza
#     if splits_dir.exists():
#         shutil.rmtree(splits_dir)

#     print(f"--- ETAPA 2: Generando Splits (80% Train, 15% Val, 5% Test) ---")

#     # 1. Extraer el Group ID (Video ID) de cada nombre de archivo
#     # Formato esperado: "videoID_timestamp.jpg"
#     video_ids = [img.split('_')[0] for img in all_images_list]
#     unique_videos = np.unique(video_ids)
#     n_videos = len(unique_videos)

#     print(f" Análisis de fuentes: {n_videos} videos únicos encontrados.")

#     train_imgs, val_imgs, test_imgs = [], [], []

#     # --- ESTRATEGIA A: SI HAY MUCHOS VIDEOS (>3) ---
#     # Hacemos split por Video ID (Lo ideal)
#     if n_videos >= 3:
#         print(" Estrategia: Group Split (Separación por Video ID).")

#         # Split 1: Separar Train (80%) de Resto (20%)
#         gss_1 = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
#         train_idx, rest_idx = next(gss_1.split(all_images_list, groups=video_ids))

#         train_imgs = [all_images_list[i] for i in train_idx]
#         rest_imgs = [all_images_list[i] for i in rest_idx]
#         rest_groups = [video_ids[i] for i in rest_idx]

#         # Split 2: Separar Resto en Val (75% de resto) y Test (25% de resto) -> Aprox 15% y 5% total
#         gss_2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
#         val_idx, test_idx = next(gss_2.split(rest_imgs, groups=rest_groups))

#         val_imgs = [rest_imgs[i] for i in val_idx]
#         test_imgs = [rest_imgs[i] for i in test_idx]

#     # --- ESTRATEGIA B: SI HAY POCOS VIDEOS (1-2) ---
#     # Hacemos split Secuencial (Cortamos el video en el tiempo)
#     # Train: Minuto 0 al 8, Val: Minuto 8 al 9, Test: Minuto 9 al 10
#     else:
#         print(" Estrategia: Sequential Split (Pocos videos, cortando por tiempo).")
#         # Ordenamos los frames para asegurar continuidad temporal
#         # Esto asume que el nombre del archivo permite ordenación (video_001 < video_002)
#         all_images_sorted = sorted(all_images_list)

#         n_total = len(all_images_sorted)
#         idx_train = int(n_total * 0.80)
#         idx_val = int(n_total * 0.95) # 80% a 95% es Val (15%)

#         train_imgs = all_images_sorted[:idx_train]
#         val_imgs = all_images_sorted[idx_train:idx_val]
#         test_imgs = all_images_sorted[idx_val:]

#     split_map = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

#     # Distribución física
#     for split_name, images in split_map.items():
#         (splits_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
#         (splits_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)

#         print(f" Distribuyendo {split_name}: {len(images)} frames (de videos distintos o bloques temporales).")

#         for img_name in tqdm(images, desc=f"Copiando {split_name}"):
#             src_img = processed_dir / "all_images" / img_name
#             src_lbl = processed_dir / "all_labels" / img_name.replace('.jpg', '.txt')

#             if src_img.exists():
#                 shutil.copy(src_img, splits_dir / split_name / 'images' / img_name)
#             if src_lbl.exists():
#                 shutil.copy(src_lbl, splits_dir / split_name / 'labels' / img_name.replace('.jpg', '.txt'))

#     create_dataset_yaml(config, splits_dir)
#     print(f" Etapa 2 completada. Datos listos en {splits_dir}.")