import sys
import os
from pathlib import Path

# Configurar entorno para importar src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.main import predict

if __name__ == "__main__":
    print(f"Iniciando Inferencia (Video + Tracking) desde: {project_root}")
    predict()