from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/interim/clinvar_labeled.parquet")


def load_clinvar_labeled(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_clinvar.py` first."
        )
    return pd.read_parquet(path)
