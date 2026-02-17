from pathlib import Path


def get_project_root():
    """
    Devuelve la ruta raíz del proyecto.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "config").exists() and (parent / "src").exists():
            return parent

    raise RuntimeError("No se pudo encontrar la raíz del proyecto. Asegúrate de que las carpetas 'config' y 'src' estén presentes.")