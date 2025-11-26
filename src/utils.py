
from pathlib import Path
import yaml
import pandas as pd

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    for d in ["outputs", "data", "logs"]:
        Path(d).mkdir(exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[save_csv] saved -> {p}")
    return p
