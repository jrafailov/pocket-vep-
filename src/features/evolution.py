from __future__ import annotations

import pandas as pd

from ..data.structure_cache import load_conservation_cache


class EvolutionFeatures:
    """Evolutionary-conservation features sourced from UCSC bigWigs.

    Columns produced (only those present in the cache):
      - phylop100way     : phyloP rate of evolution at the variant position
                           (positive = conserved, negative = accelerated).
      - phastcons100way  : phastCons HMM-smoothed conservation probability (0-1).

    Cache is per-locus (Chromosome, PositionVCF), shared across multi-allelic
    variants at the same position. Variants whose locus has no signal in
    either track are dropped, same contract as StructureFeatures.

    Build the cache with:
        python scripts/build_conservation_cache.py
    """

    name = "evolution"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cache = load_conservation_cache().set_index(["Chromosome", "PositionVCF"])

        work = df[["Chromosome", "PositionVCF"]].copy()
        work["Chromosome"] = work["Chromosome"].astype(str)
        work = work.dropna(subset=["PositionVCF"])
        work["PositionVCF"] = work["PositionVCF"].astype(int)

        keys = list(zip(work["Chromosome"], work["PositionVCF"]))
        joined = cache.reindex(keys)
        joined.index = work.index

        score_cols = [c for c in ("phylop100way", "phastcons100way")
                      if c in joined.columns]
        out = joined[score_cols].astype(float)
        return out.dropna()
