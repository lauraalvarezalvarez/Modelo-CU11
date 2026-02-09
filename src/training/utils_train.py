import os
import shutil
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm

def create_dataset_yaml(config, target_path):
    names_dict = {int(k): v for k, v in config['dataset']['class_names'].items()}
    yaml_content = {
        'path': str(target_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': names_dict
    }
    with open(target_path / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def generate_splits(config):
    """
    Toma los datos de processed/all_images y genera la estructura data/splits/
    Aplicando lógica de NO mezclar frames del mismo video (Data Leakage prevention).
    """
    root_path = config['root_path']
    processed_dir = Path(config['paths']['processed_path'])
    splits_dir = Path(config['paths']['splits_path'])

    img_source_dir = processed_dir / "all_images"
    lbl_source_dir = processed_dir / "all_labels"

    if not img_source_dir.exists():
        raise FileNotFoundError(f" No se encuentran datos procesados en {processed_dir}. Ejecuta data_processing primero.")

    # Listar todas las imágenes disponibles
    all_images_list = [f.name for f in img_source_dir.glob("*.jpg")]

    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    print(f"--- UTILS TRAIN: Generando Splits (80% Train, 15% Val, 5% Test) ---")

    # --- LOGICA DE SPLIT POR VIDEO ---
    video_ids = [img.split('_')[0] for img in all_images_list]
    unique_videos = np.unique(video_ids)
    n_videos = len(unique_videos)

    train_imgs, val_imgs, test_imgs = [], [], []

    # ESTRATEGIA A: Group Split (>3 videos)
    if n_videos >= 3:
        print(f" Estrategia: Group Split por Video ID ({n_videos} videos).")
        # 1. Train vs Resto (80/20)
        gss_1 = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
        train_idx, rest_idx = next(gss_1.split(all_images_list, groups=video_ids))

        train_imgs = [all_images_list[i] for i in train_idx]
        rest_imgs_temp = [all_images_list[i] for i in rest_idx]
        rest_groups = [video_ids[i] for i in rest_idx]

        # 2. Resto -> Val/Test (75/25 de resto => 15/5 total)
        gss_2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
        val_idx, test_idx = next(gss_2.split(rest_imgs_temp, groups=rest_groups))

        val_imgs = [rest_imgs_temp[i] for i in val_idx]
        test_imgs = [rest_imgs_temp[i] for i in test_idx]

    # ESTRATEGIA B: Sequential Split (Pocos videos)
    else:
        print(" Estrategia: Sequential Split (Pocos videos, corte temporal).")
        all_images_sorted = sorted(all_images_list)
        n = len(all_images_sorted)
        idx_train = int(n * 0.80)
        idx_val = int(n * 0.95)

        train_imgs = all_images_sorted[:idx_train]
        val_imgs = all_images_sorted[idx_train:idx_val]
        test_imgs = all_images_sorted[idx_val:]

    # Distribución Física
    split_map = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

    for split_name, images in split_map.items():
        (splits_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (splits_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)

        print(f" {split_name}: {len(images)} frames.")
        for img_name in tqdm(images, desc=f"Copying {split_name}"):
            shutil.copy(img_source_dir / img_name, splits_dir / split_name / 'images' / img_name)
            shutil.copy(lbl_source_dir / img_name.replace('.jpg', '.txt'), splits_dir / split_name / 'labels' / img_name.replace('.jpg', '.txt'))

    create_dataset_yaml(config, splits_dir)
    print(f" Splits generados en {splits_dir}")