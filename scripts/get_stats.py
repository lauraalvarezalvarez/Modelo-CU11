import sys
from pathlib import Path

# Configurar entorno para importar src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.main import get_stats

if __name__ == "__main__":
    print(f" Iniciando generación de estadísticas desde: {project_root}")
    get_stats()