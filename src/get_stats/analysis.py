import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import yaml

def get_class_status(percentage):
    """
    Determina el estado de la clase según su representatividad
    para ayudar a redactar la tabla de la memoria.
    """
    if percentage >= 30.0:
        return " Clase Mayoritaria"
    elif 10.0 <= percentage < 30.0:
        return " Bien representada"
    elif 1.0 <= percentage < 10.0:
        return " Minoritaria"
    else:
        return " Desbalance Extremo"

def scan_labels_directory(directory, desc="Scanning"):
    """
    Auxiliar: Lee todos los txt de un directorio y devuelve un Counter.
    """
    counter = Counter()
    files = list(directory.glob("*.txt"))

    for f_path in tqdm(files, desc=desc):
        with open(f_path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    counter[int(parts[0])] += 1
    return counter, len(files)

def analyze_global_stats(config):
    """
    Analiza data/processed/all_labels (ANTES de los splits).
    Genera la tabla base para el EDA.
    """
    root_path = config['root_path']
    processed_labels = Path(config['paths']['processed_path']) / "all_labels"
    class_names = config['dataset']['class_names']

    print(f"\n--- GET STATS: Analizando Dataset Global ({processed_labels}) ---")

    if not processed_labels.exists():
        print(f" No existe {processed_labels}. Ejecuta data_processing primero.")
        return pd.DataFrame()

    counter, n_files = scan_labels_directory(processed_labels, desc="Global Analysis")
    total_instances = sum(counter.values())

    stats = []
    for cls_id, count in counter.items():
        pct = (count / total_instances) * 100 if total_instances > 0 else 0
        stats.append({
            "Class_ID": cls_id,
            "Class_Name": class_names.get(cls_id, str(cls_id)),
            "Total_Samples": count,
            "Percentage": round(pct, 2),
            "Status": get_class_status(pct) # <--- AÑADIDO PARA TU MEMORIA
        })
    df = pd.DataFrame(stats).sort_values(by="Total_Samples", ascending=False)
    return df, n_files, total_instances

def analyze_splits_stats(config):
    """
    Analiza data/splits (train/val/test).
    """
    splits_path = Path(config['paths']['splits_path'])
    class_names = config['dataset']['class_names']
    stats = []

    print(f"\n--- GET STATS: Analizando Splits ({splits_path}) ---")

    for split in ['train', 'val', 'test']:
        lbl_dir = splits_path / split / 'labels'
        if not lbl_dir.exists(): continue

        counter, n_files = scan_labels_directory(lbl_dir, desc=f"Scanning {split}")
        total_split = sum(counter.values())

        for cls_id, count in counter.items():
            pct = (count / total_split) * 100 if total_split > 0 else 0
            stats.append({
                "Split": split,
                "Class_Name": class_names.get(cls_id, str(cls_id)),
                "Count": count,
                "Split_Freq_%": round(pct, 2)
            })

    return pd.DataFrame(stats)

def get_data_dictionary(config):
    """Genera el diccionario de datos (Features)"""
    # (Misma lógica que tenías antes, resumida para este ejemplo)
    class_names = config['dataset']['class_names']
    features = [
        {"Feature_Name": "image_data", "Data_Type": "Tensor (float32)", "Description": "Input RGB 640x640", "Role": "Input"},
        {"Feature_Name": "class_id", "Data_Type": "Categorical", "Description": f"Target ID: {class_names}", "Role": "Target"},
        {"Feature_Name": "bbox_coords", "Data_Type": "Float (x,y,w,h)", "Description": "Normalized Box Coordinates", "Role": "Target"}
    ]
    return pd.DataFrame(features)

def generate_stats_report(config):
    root_path = config['root_path']
    metrics_dir = root_path / "models" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Features
    df_feat = get_data_dictionary(config)
    df_feat.to_csv(metrics_dir / "dataset_features.csv", index=False)

    # 2. Global Stats (Processed)
    df_global, n_imgs, n_anns = analyze_global_stats(config)
    global_path = metrics_dir / "global_stats.csv"
    if not df_global.empty:
        df_global.to_csv(global_path, index=False)
        print(f"\n Resumen Global (Imágenes: {n_imgs}, Anotaciones: {n_anns})")
        # Imprimimos tabla bonita en consola para verificar
        print(df_global[['Class_Name', 'Total_Samples', 'Percentage', 'Status']].to_string(index=False))

    # 3. Splits Stats
    df_splits = analyze_splits_stats(config)
    splits_path = metrics_dir / "splits_stats.csv"
    if not df_splits.empty:
        df_splits.to_csv(splits_path, index=False)

        # Pivot table para ver si los splits están balanceados entre sí
        pivot = df_splits.pivot_table(index="Class_Name", columns="Split", values="Split_Freq_%", fill_value=0)
        print("\n Balanceo entre Splits (%) - Deben ser similares:")
        print(pivot)

    print(f"\n Archivos generados en {metrics_dir}:")
    print(f" - dataset_features.csv")
    print(f" - global_stats.csv (Usa este para la sección 5.2.1)")
    print(f" - splits_stats.csv")