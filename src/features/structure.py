from __future__ import annotations

import pandas as pd

from ..data.structure_cache import attach_uniprot, load_plddt_cache, load_structure_cache

CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")
SS_CATEGORIES = ["H", "E", "L"]


class StructureFeatures:
    """Pocket / structure-level features per research proposal.

    Columns produced:
      - dist_to_nearest_pocket : min CA distance to any fpocket-predicted pocket residue
      - in_pocket              : 0/1, residue lines a pocket
      - druggability           : fpocket druggability score of the nearest pocket
      - sasa                   : DSSP relative SASA
      - ss_H / ss_E / ss_L     : one-hot of DSSP 3-state secondary structure
      - plddt                  : AlphaFold per-residue confidence

    pLDDT is grouped with structure features because it is a structural-quality
    signal (AlphaFold's confidence at the position), not a property of the
    amino-acid sequence itself. Keeping it here keeps the sequence-vs-structure
    ablation clean.

    Isoform policy: rows whose parsed WT_AA does not match the AlphaFold
    canonical residue at Position are dropped (ClinVar positions are
    transcript-specific; AlphaFold is canonical isoform only). This is the
    agreed behaviour per the plan's isoform-mismatch decision.
    """

    name = "structure"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed = df["protein_change_clean"].str.extract(r"([A-Z])(\d+)([A-Z\*=])")
        work = pd.DataFrame(
            {
                "WT_AA": parsed[0],
                "Position": pd.to_numeric(parsed[1], errors="coerce"),
            },
            index=df.index,
        )
        work = work.loc[work["WT_AA"].isin(CANONICAL_AA)].copy()

        work["uniprot_id"] = attach_uniprot(df).loc[work.index]
        work = work.dropna(subset=["uniprot_id", "Position"])
        work["Position"] = work["Position"].astype(int)

        cache = load_structure_cache().set_index(["uniprot_id", "position"])
        keys = list(zip(work["uniprot_id"], work["Position"]))

        joined = cache.reindex(keys)
        joined.index = work.index

        # pLDDT lives in its own parquet because --stage plddt and --stage
        # features run separately, but the key is identical.
        plddt = load_plddt_cache().set_index(["uniprot_id", "position"])["plddt"]
        joined["plddt"] = plddt.reindex(keys).values

        # Isoform mismatch filter: AlphaFold canonical residue must agree
        # with the WT_AA that ClinVar encoded for this variant.
        keep = (joined["wt_aa"] == work["WT_AA"]) & joined["wt_aa"].notna()
        joined = joined.loc[keep]

        # Drop any residual NaNs on the structural columns (e.g. positions
        # present in cache but with missing features).
        joined = joined.dropna(
            subset=["ss", "sasa", "in_pocket", "dist_to_nearest_pocket", "plddt"]
        )

        ss_cat = pd.Categorical(joined["ss"], categories=SS_CATEGORIES)
        ss_oh = pd.get_dummies(ss_cat, prefix="ss", dtype=int)
        ss_oh.index = joined.index

        out = pd.concat(
            [
                joined[
                    [
                        "dist_to_nearest_pocket",
                        "in_pocket",
                        "druggability",
                        "sasa",
                        "plddt",
                    ]
                ].astype(float),
                ss_oh,
            ],
            axis=1,
        )
        return out
