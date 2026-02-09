import sys
import os
from pathlib import Path

# --- Configuración del entorno ---
# Añadimos la carpeta raíz del proyecto al sys.path para poder importar 'src'
# scripts/train.py está en ROOT/scripts/, así que la raíz es el padre (..)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Importamos la función train del núcleo en src
from src.main import train

if __name__ == "__main__":
    print(f"Iniciando script de entrenamiento desde: {project_root}")
    # Ejecuta la lógica definida en src/main.py 
    train()