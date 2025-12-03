import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("DRP Project Data")

IC50_PATH = DATA_DIR / "sorted_IC50_82833_580_170.csv"
DRUG_SMILES_PATH = DATA_DIR / "drug_smiles.csv"

CELL_SOURCES = {
    "exp": DATA_DIR / "CCLE_2369_EXP.csv",
    "cnv": DATA_DIR / "CCLE_2369_binary_cnv.csv",
    "mut": DATA_DIR / "CCLE_2369_hotspot_mut.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drug response prediction baseline")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fp-bits", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "cell", "drug", "cell_drug"],
    )
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def load_ic50(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "DepMap_ID": "cell_id",
        "Drug name": "drug_name",
        "Drug Id": "drug_id"
    })
    keep_cols = ["cell_id", "drug_name", "drug_id", "IC50"]
    df = df[keep_cols].dropna()
    df = df.drop_duplicates(subset=["cell_id", "drug_name"])
    return df


def load_drugs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "drug_name": "drug_name",
        "IsomericSMILES": "smiles"
    })
    df = df[["drug_name", "smiles"]].dropna()
    df = df.drop_duplicates(subset=["drug_name"])
    return df


def load_cell_block(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = df.columns[0]

    df = df.rename(columns={id_col: "cell_id"})
    feature_cols = [c for c in df.columns if c != "cell_id"]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["cell_id"]).drop_duplicates(subset=["cell_id"])

    df = df.set_index("cell_id")
    df = df.add_prefix(f"{prefix}_")
    return df.reset_index()


def load_cells(sources: Dict[str, Path]) -> pd.DataFrame:
    blocks: List[pd.DataFrame] = []

    for prefix, path in sources.items():
        if path.exists():
            blocks.append(load_cell_block(path, prefix))

    base = blocks[0]
    for blk in blocks[1:]:
        base = base.merge(blk, on="cell_id", how="inner")

    return base.dropna()


def main() -> None:
    pass  # TODO: implement training/testing workflow


if __name__ == "__main__":
    main()