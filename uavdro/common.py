"""Config loading and small I/O helpers shared by all scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/shanghai.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    cfg = yaml.safe_load(p.read_text())
    # resolve every path relative to the repo root
    cfg["paths"] = {k: (REPO / v) for k, v in cfg["paths"].items()}
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/shanghai.yaml")
    return ap


def read_parquet_or_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    alt = path.with_suffix(".csv")
    if alt.exists():
        return pd.read_csv(alt)
    raise FileNotFoundError(f"Missing {path} (and {alt.name})")


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: Path):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2, default=float) + "\n")


def read_site_list(path: Path) -> list[str]:
    """Footprint file: CSV with a `site_id` column, or plain text one id per line."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)["site_id"].astype(str).tolist()
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def read_design(path: Path) -> dict[str, int]:
    """Charger design CSV with columns site_id,c."""
    d = pd.read_csv(path)
    return dict(zip(d["site_id"].astype(str), d["c"].astype(int)))


def write_design(c: dict[str, int], path: Path):
    ensure_dir(Path(path).parent)
    pd.DataFrame({"site_id": list(c), "c": [int(v) for v in c.values()]}).to_csv(path, index=False)
