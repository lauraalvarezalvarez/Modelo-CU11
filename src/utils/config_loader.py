import yaml
from pathlib import Path
from .paths import get_project_root


def load_config():

    root = get_project_root()

    config_path = root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # convertir rutas relativas → absolutas
    for k, v in cfg.items():
        if isinstance(v, str) and ("/" in v or "\\" in v):
            cfg[k] = str(root / v)

    cfg["PROJECT_ROOT"] = str(root)

    return cfg
