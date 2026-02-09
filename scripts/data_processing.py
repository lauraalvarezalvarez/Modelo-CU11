import sys
import os
from pathlib import Path

# Configurar entorno para importar src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.main import process_dataset

if __name__ == "__main__":
    process_dataset()