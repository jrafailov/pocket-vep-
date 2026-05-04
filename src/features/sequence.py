from __future__ import annotations

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")

AA_MASS = {
    "A": 71.04, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.11, "W": 186.21, "Y": 163.18, "V": 99.13,
}

AA_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

AA_CHARGE = {
    "R": 1, "K": 1, "H": 0.1,
    "D": -1, "E": -1,
    "A": 0, "N": 0, "C": 0, "Q": 0, "G": 0, "I": 0, "L": 0,
    "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _blosum_lookup(wt: str, mt: str) -> float:
    try:
        return float(BLOSUM62[wt, mt])
    except (KeyError, IndexError):
        return np.nan


class SequenceFeatures:
    """Sequence-only features for the ablation's seq arm.

    Columns produced:
      - Physicochemical deltas (Delta_Mass, Delta_Hydro, Delta_Charge)
      - BLOSUM62 substitution score
      - One-hot WT_AA + MT_AA identity

    Conservation scores live in EvolutionFeatures and pLDDT lives in
    StructureFeatures, so this block is purely amino-acid level.
    """

    name = "sequence"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed = df["protein_change_clean"].str.extract(r"([A-Z])(\d+)([A-Z\*=])")
        work = pd.DataFrame(
            {
                "WT_AA": parsed[0],
                "Position": pd.to_numeric(parsed[1], errors="coerce"),
                "MT_AA": parsed[2],
            },
            index=df.index,
        )

        # Keep only canonical 20 AAs on both sides (drops frameshifts, stops, '=', 'X').
        mask = work["WT_AA"].isin(CANONICAL_AA) & work["MT_AA"].isin(CANONICAL_AA)
        work = work.loc[mask].copy()

        work["Delta_Mass"] = work["MT_AA"].map(AA_MASS) - work["WT_AA"].map(AA_MASS)
        work["Delta_Hydro"] = work["MT_AA"].map(AA_HYDRO) - work["WT_AA"].map(AA_HYDRO)
        work["Delta_Charge"] = work["MT_AA"].map(AA_CHARGE) - work["WT_AA"].map(AA_CHARGE)
        work["BLOSUM62"] = [
            _blosum_lookup(wt, mt) for wt, mt in zip(work["WT_AA"], work["MT_AA"])
        ]

        numeric_cols = ["Delta_Mass", "Delta_Hydro", "Delta_Charge", "BLOSUM62"]
        numeric = work[numeric_cols]

        wt_cat = pd.Categorical(work["WT_AA"], categories=CANONICAL_AA)
        mt_cat = pd.Categorical(work["MT_AA"], categories=CANONICAL_AA)
        wt_oh = pd.get_dummies(wt_cat, prefix="WT_AA", dtype=int)
        mt_oh = pd.get_dummies(mt_cat, prefix="MT_AA", dtype=int)
        wt_oh.index = work.index
        mt_oh.index = work.index

        out = pd.concat([numeric, wt_oh, mt_oh], axis=1)
        return out.dropna()
