import pandas as pd
from pathlib import Path
import os

def load_raw_annotations(config):
    """
    Carga el CSV de anotaciones crudas y devuelve un DataFrame.
    Maneja la ubicación del archivo y los nombres de columnas.
    """
    root_path = config['root_path']
    raw_path = root_path / config['paths']['raw_path']
    ann_folder = config['paths']['folders']['annotations']
    raw_file = config['raw_files']['annotations']

    # Nombre del archivo hardcodeado o podría venir del config
    csv_path = raw_path / ann_folder / raw_file

    if not csv_path.exists():
        raise FileNotFoundError(f" Error: No se encuentra el archivo de anotaciones en {csv_path}")

    # Definición de columnas (Esquema de datos)
    col_names = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']

    print(f" Cargando datos crudos desde: {csv_path}")

    # Lectura
    try:
        df = pd.read_csv(csv_path, names=col_names, header=None, dtype={'video_id': str})
        return df
    except Exception as e:
        print(f" Error leyendo el CSV: {e}")
        return None