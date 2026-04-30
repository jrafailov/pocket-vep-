"""Materialize a single feature parquet for fair feature-set comparison.

Runs every feature block once, inner-joins their outputs, attaches the label
and a few passthrough columns, and writes:

    data/processed/feature_matrix.parquet         -- features + label + passthrough
    data/processed/feature_matrix.schema.json     -- {block_name: [columns], ...}

run_experiments.py reads from these so that "sequence", "structure", and
"combined" experiments all train on the *same row set* (the inner-joined
intersection). Without this, a sequence-only run uses every ClinVar row and a
structure run uses only AlphaFold-covered rows -- gains from adding structure
features get confounded with the smaller row budget.

Run after scripts/build_structure_cache.py.

    python scripts/build_feature_matrix.py
    python scripts/build_feature_matrix.py --no-plddt
    python scripts/build_feature_matrix.py --blocks sequence
    python scripts/build_feature_matrix.py \\
        --structure-cache data/processed/structure_features.debug.parquet \\
        --out data/processed/feature_matrix.debug.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_clinvar_labeled  # noqa: E402
from src.features import schema_path_for  # noqa: E402
from src.features.base import FeatureBlock  # noqa: E402
from src.features.sequence import SequenceFeatures  # noqa: E402
from src.features.structure import StructureFeatures  # noqa: E402

LABEL_COL = "ML_Label"
PASSTHROUGH_COLS = ["GeneSymbol", "protein_change_clean"]
DEFAULT_BLOCKS = ["sequence", "structure"]
DEFAULT_OUT = ROOT / "data/processed/feature_matrix.parquet"


def _instantiate_blocks(names: list[str], include_plddt: bool) -> dict[str, FeatureBlock]:
    blocks: dict[str, FeatureBlock] = {}
    for name in names:
        if name == "sequence":
            blocks[name] = SequenceFeatures(include_plddt=include_plddt)
        elif name == "structure":
            blocks[name] = StructureFeatures()
        else:
            raise ValueError(f"Unknown feature block: {name!r}")
    return blocks


def _override_structure_cache(path: Path) -> None:
    """Point the structure cache loader at a custom parquet (debug subsets)."""
    if not path.exists():
        raise FileNotFoundError(f"--structure-cache path does not exist: {path}")
    from src.data import structure_cache as sc

    sc.STRUCTURE_CACHE = path
    print(f"[build] structure cache override -> {path}")


def build(
    data_path: Path,
    out_path: Path,
    block_names: list[str],
    include_plddt: bool,
) -> None:
    print(f"[build] loading {data_path}")
    df = load_clinvar_labeled(data_path)
    print(f"[build]   {len(df):,} labeled rows")

    blocks = _instantiate_blocks(block_names, include_plddt=include_plddt)

    parts: dict[str, pd.DataFrame] = {}
    for name, block in blocks.items():
        print(f"[build] running block: {name}")
        out = block.transform(df)
        print(f"[build]   {name}: {len(out):,} rows x {out.shape[1]} cols")
        parts[name] = out

    # Inner-join every block on row index -- only rows covered by every block survive.
    joined = pd.concat(list(parts.values()), axis=1, join="inner")
    print(f"[build] inner-joined: {len(joined):,} rows x {joined.shape[1]} cols")

    # Attach the label and passthrough columns from the original ClinVar parquet.
    joined[LABEL_COL] = df.loc[joined.index, LABEL_COL]
    for col in PASSTHROUGH_COLS:
        if col not in df.columns:
            print(f"[build]   WARN: passthrough column {col!r} not in {data_path.name}; skipping")
            continue
        joined[col] = df.loc[joined.index, col]

    # Schema sidecar -- maps block name to the columns it owns.
    schema: dict[str, object] = {name: list(parts[name].columns) for name in blocks}
    schema["label_col"] = LABEL_COL
    schema["passthrough"] = [c for c in PASSTHROUGH_COLS if c in joined.columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_path, index=True)
    schema_p = schema_path_for(out_path)
    schema_p.write_text(json.dumps(schema, indent=2))

    print(f"[build] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[build] wrote {schema_p}")
    print(f"[build] label balance:")
    print(joined[LABEL_COL].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=ROOT / "data/interim/clinvar_labeled.parquet",
                    type=Path)
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--blocks", nargs="+", default=DEFAULT_BLOCKS,
                    choices=DEFAULT_BLOCKS,
                    help="Subset of feature blocks to materialize. Default: all.")
    plddt_grp = ap.add_mutually_exclusive_group()
    plddt_grp.add_argument("--include-plddt", dest="include_plddt",
                           action="store_true", default=True,
                           help="Include AlphaFold pLDDT in sequence features (default).")
    plddt_grp.add_argument("--no-plddt", dest="include_plddt", action="store_false",
                           help="Skip pLDDT (lets you build features without "
                                "data/processed/plddt_cache.parquet).")
    ap.add_argument("--structure-cache", type=Path, default=None,
                    help="Override path for structure_features.parquet "
                         "(useful for building a debug matrix from a "
                         "--limit'd structure cache).")
    args = ap.parse_args()

    if args.structure_cache is not None:
        _override_structure_cache(args.structure_cache)

    build(
        data_path=args.data,
        out_path=args.out,
        block_names=args.blocks,
        include_plddt=args.include_plddt,
    )


if __name__ == "__main__":
    main()
