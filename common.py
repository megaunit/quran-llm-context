"""Portable I/O helper added for release; the original common.py was absent from the handoff.
Experiment logic remains in the archived scripts. Paths are relative to this directory.
"""
from pathlib import Path
import csv
import json
import yaml
ROOT = Path(__file__).resolve().parent

def path(value):
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else ROOT / p)

def load_config(filename="config.yaml"):
    with open(path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)

def _csv(filename):
    with open(path(filename), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_dataset(cfg):
    result = []
    for row in _csv(cfg["dataset"]):
        row["surah"], row["ayah"] = int(row["surah"]), int(row["ayah"])
        row["key"] = f"{row['surah']}:{row['ayah']}"
        result.append(row)
    return result

def load_few_shot_keys(cfg):
    rows = _csv(cfg["few_shot"])
    return [f"{int(r['surah'])}:{int(r['ayah'])}" for r in rows]

def check_no_overlap(dataset, few_keys):
    overlap = {r["key"] for r in dataset}.intersection(few_keys)
    if overlap:
        raise ValueError(f"Few-shot examples overlap evaluation verses: {sorted(overlap)}")

def load_cache(cfg):
    with open(path(cfg["cache"]), encoding="utf-8") as f:
        return json.load(f)

def load_categories(cfg):
    with open(path(cfg["categories"]), encoding="utf-8") as f:
        return yaml.safe_load(f)
