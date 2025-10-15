"""Data discovery, filename-based age parsing, and stratified splits."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PARTS, NUM_BINS, SEED, VALID_EXTS, AGE_MAX


def _has_valid_ext(p: Path) -> bool:
    """Return True if file looks like an image we accept (UTKFace often has .jpg, .png)."""
    name = p.name.lower()
    return p.suffix.lower() in VALID_EXTS or name.endswith('.jpg.chip.jpg')


def parse_age_from_filename(pathlike: str | Path) -> int | None:
    """
    Parse age from UTKFace-style filenames such as: '27_1_2_20170116174525125.jpg'.
    Strategy:
    1) Prefer digits before first underscore (e.g., '27_...').
    2) Fallback: take first token, strip non-digits, check 0..AGE_MAX.
    """
    base = Path(pathlike).name
    m = re.match(r'^(\d{1,3})_', base)
    if m:
        age = int(m.group(1))
        if 0 <= age <= AGE_MAX:
            return age
    first = re.sub(r'\D+', '', base.split('_')[0])
    if first.isdigit():
        age = int(first)
        if 0 <= age <= AGE_MAX:
            return age
    return None


def collect_images(parts: list[Path]) -> list[str]:

    """Recursively gather absolute filepaths for all images under the given folders."""
    all_files: list[str] = []
    for folder in parts:
        if not folder.exists() or not folder.is_dir():
            print(f"[WARN] Folder not found: {folder}")
            continue
        files = [str(fp) for fp in folder.rglob('*') if fp.is_file() and _has_valid_ext(fp)]
        print(f"[INFO] {folder} → {len(files)} files")
        all_files.extend(files)
    return all_files


def build_dataframe(files: list[str]) -> pd.DataFrame:

    """Create a shuffled DataFrame with columns [filepath, age]; drop rows with missing age."""
    rows = []
    skipped = 0
    for f in files:
        age = parse_age_from_filename(f)
        if age is None:
            skipped += 1
            continue
        rows.append((f, age))
    if skipped:
        print(f"[INFO] Skipped files (no/invalid age): {skipped}")
    if len(rows) < 50:
        raise RuntimeError("Too few valid labeled images (<50). Check filename patterns.")
    df = pd.DataFrame(rows, columns=["filepath", "age"]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print("Age describe:\n", df['age'].describe())
    return df

def stratified_split(df: pd.DataFrame, bins: int = NUM_BINS, seed: int = SEED):
    """Stratify continuous ages into bins; split into train/val/test preserving distribution."""
    a_min, a_max = int(df['age'].min()), int(df['age'].max())
    if a_min == a_max:
        df['age_bin'] = 0
    else:
        edges = np.linspace(a_min, a_max, bins + 1)
        edges = np.unique(np.round(edges, 6))
        if len(edges) < 3:
            edges = np.array([a_min, (a_min + a_max) / 2.0, a_max])
        df['age_bin'] = pd.cut(df['age'], bins=edges, include_lowest=True, labels=False)
    df['age_bin'] = df['age_bin'].fillna(0).astype(int)


    try:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df['age_bin'])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df['age_bin'])
    except ValueError as e:
        print(f"[WARN] Stratified split failed ({e}); falling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)


    return (
        train_df[["filepath", "age"]].reset_index(drop=True),
        val_df[["filepath", "age"]].reset_index(drop=True),
        test_df[["filepath", "age"]].reset_index(drop=True),
    )
