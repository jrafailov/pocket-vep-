"""Feature-block registry.

Each feature category is a plug-in that takes a ClinVar-shaped DataFrame and
returns a per-row feature DataFrame. Adding a category = write a class + add
one line to FEATURE_REGISTRY.

Prerequisites per block (see scripts/build_structure_cache.py):
    sequence   -> data/interim/uniprot_mapping.parquet
                  data/processed/plddt_cache.parquet
                  (set SequenceFeatures(include_plddt=False) to skip these)
    structure  -> data/interim/uniprot_mapping.parquet
                  data/processed/structure_features.parquet
    evolution  -> data/processed/conservation_cache.parquet
                  (built by scripts/build_conservation_cache.py from UCSC
                  phyloP / phastCons bigWigs; joins on Chromosome + PositionVCF)

Materialized matrix:
    scripts/build_feature_matrix.py runs every block once, inner-joins them,
    and writes data/processed/feature_matrix.parquet plus a sidecar
    data/processed/feature_matrix.schema.json mapping each registry key to its
    output columns. Training reads this single file and selects columns by
    feature key, so every feature-set variant trains on the same row set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import FeatureBlock
from .evolution import EvolutionFeatures
from .sequence import SequenceFeatures
from .structure import StructureFeatures

FEATURE_REGISTRY: dict[str, FeatureBlock] = {
    "sequence": SequenceFeatures(include_plddt=True),
    "structure": StructureFeatures(),
    "evolution": EvolutionFeatures(),
}


def build_feature_matrix(df: pd.DataFrame, feature_keys: Iterable[str]) -> pd.DataFrame:
    """Run every requested feature block and inner-join on index.

    Any row that at least one block could not handle is dropped.
    """
    keys = list(feature_keys)
    if not keys:
        raise ValueError("feature_keys must be non-empty")
    unknown = set(keys) - set(FEATURE_REGISTRY)
    if unknown:
        raise KeyError(f"Unknown feature block(s): {sorted(unknown)}")
    parts = [FEATURE_REGISTRY[k].transform(df) for k in keys]
    return pd.concat(parts, axis=1, join="inner")


def schema_path_for(matrix_path: str | Path) -> Path:
    """Sidecar JSON lives next to the parquet: foo.parquet -> foo.schema.json."""
    p = Path(matrix_path)
    return p.with_name(p.stem + ".schema.json")


def load_feature_matrix(
    path: str | Path = "data/processed/feature_matrix.parquet",
) -> tuple[pd.DataFrame, dict]:
    """Load the materialized feature matrix and its schema.

    Returns (df, schema). `df` contains every feature column from every block,
    plus the label and any passthrough columns. `schema` maps registry keys
    (e.g. "sequence", "structure") to their column lists, and also carries
    "label_col" and "passthrough" entries.
    """
    matrix_path = Path(path)
    schema_p = schema_path_for(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"{matrix_path} not found. Build it first:\n"
            f"    python scripts/build_feature_matrix.py"
        )
    if not schema_p.exists():
        raise FileNotFoundError(
            f"{schema_p} not found. Re-run scripts/build_feature_matrix.py "
            f"to regenerate the schema sidecar."
        )
    df = pd.read_parquet(matrix_path)
    schema = json.loads(schema_p.read_text())
    return df, schema


def select_features(
    df: pd.DataFrame, feature_keys: Iterable[str], schema: dict
) -> pd.DataFrame:
    """Slice the materialized matrix down to the requested feature blocks.

    Unlike build_feature_matrix, this does not recompute anything -- the
    matrix already contains every block's columns and was inner-joined at
    build time, so all rows have full coverage.
    """
    keys = list(feature_keys)
    if not keys:
        raise ValueError("feature_keys must be non-empty")
    unknown = [k for k in keys if k not in schema]
    if unknown:
        raise KeyError(
            f"Unknown feature block(s) {unknown}; schema knows {sorted(schema)}"
        )
    cols: list[str] = []
    for k in keys:
        cols.extend(schema[k])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Schema lists columns not present in the matrix: {missing[:5]}..."
        )
    return df[cols]


__all__ = [
    "FEATURE_REGISTRY",
    "FeatureBlock",
    "EvolutionFeatures",
    "SequenceFeatures",
    "StructureFeatures",
    "build_feature_matrix",
    "load_feature_matrix",
    "select_features",
    "schema_path_for",
]
