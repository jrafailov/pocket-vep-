"""Shared loaders for the AlphaFold-derived caches.

Two feature blocks (sequence + structure) both need to join ClinVar variants
against per-residue tables keyed on (uniprot_id, position). This module
centralizes those paths and the gene->uniprot mapping so the join logic is
defined once.

Build these files with:
    python scripts/build_structure_cache.py --stage map
    python scripts/build_structure_cache.py --stage download
    python scripts/build_structure_cache.py --stage plddt      # -> plddt_cache.parquet
    python scripts/build_structure_cache.py --stage features   # -> structure_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PLDDT_CACHE = Path("data/processed/plddt_cache.parquet")
STRUCTURE_CACHE = Path("data/processed/structure_features.parquet")
CONSERVATION_CACHE = Path("data/processed/conservation_cache.parquet")
UNIPROT_MAPPING = Path("data/interim/uniprot_mapping.parquet")


def _require(path: Path, build_cmd: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n    {build_cmd}"
        )
    return path


def load_gene_to_uniprot() -> pd.DataFrame:
    """Return [gene_symbol, uniprot_id] (one row per gene, reviewed preferred)."""
    path = _require(
        UNIPROT_MAPPING,
        "python scripts/build_structure_cache.py --stage map",
    )
    return pd.read_parquet(path)[["gene_symbol", "uniprot_id"]]


def load_plddt_cache() -> pd.DataFrame:
    """Return [uniprot_id, position, wt_aa, plddt]."""
    path = _require(
        PLDDT_CACHE,
        "python scripts/build_structure_cache.py --stage plddt",
    )
    return pd.read_parquet(path)


def load_conservation_cache() -> pd.DataFrame:
    """Return [Chromosome, PositionVCF, phylop100way, phastcons100way].

    Per-variant evolutionary conservation from UCSC phyloP / phastCons bigWigs.
    """
    path = _require(
        CONSERVATION_CACHE,
        "python scripts/build_conservation_cache.py",
    )
    return pd.read_parquet(path).drop_duplicates(["Chromosome", "PositionVCF"])


def load_structure_cache() -> pd.DataFrame:
    """Return [uniprot_id, position, wt_aa, ss, sasa, in_pocket, dist_to_nearest_pocket, druggability]."""
    path = _require(
        STRUCTURE_CACHE,
        "python scripts/build_structure_cache.py --stage features",
    )
    # Older parquets built before the uid-dedupe fix can contain duplicate
    # (uniprot_id, position) rows; downstream reindex needs a unique key.
    return pd.read_parquet(path).drop_duplicates(["uniprot_id", "position"])


def attach_uniprot(df: pd.DataFrame, gene_col: str = "GeneSymbol") -> pd.Series:
    """Return a Series of uniprot_id aligned to df.index (NaN where unmapped)."""
    mapping = load_gene_to_uniprot().set_index("gene_symbol")["uniprot_id"]
    return df[gene_col].map(mapping)
