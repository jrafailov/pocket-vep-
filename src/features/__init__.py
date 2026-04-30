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
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import FeatureBlock
from .sequence import SequenceFeatures
from .structure import StructureFeatures

FEATURE_REGISTRY: dict[str, FeatureBlock] = {
    "sequence": SequenceFeatures(include_plddt=True),
    "structure": StructureFeatures(),
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


__all__ = [
    "FEATURE_REGISTRY",
    "FeatureBlock",
    "SequenceFeatures",
    "StructureFeatures",
    "build_feature_matrix",
]
