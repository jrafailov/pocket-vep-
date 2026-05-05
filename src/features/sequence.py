from __future__ import annotations

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.Data.IUPACData import protein_weights
from Bio.SeqUtils.ProtParamData import kd

CANONICAL_AA = list("ACDEFGHIKLMNPQRSTVWY")

# Average mass of one water molecule (Da). A residue inside a peptide chain
# has lost one H2O relative to the free amino acid via the peptide bond.
_WATER_DA = 18.01528

# Average residue masses in Daltons. Source: Bio.Data.IUPACData.protein_weights
# (free amino acid average mass) minus _WATER_DA. Pinned by tests/test_sequence_constants.py.
AA_MASS = {aa: round(protein_weights[aa] - _WATER_DA, 4) for aa in CANONICAL_AA}

# Kyte-Doolittle hydropathy index. Source: Bio.SeqUtils.ProtParamData.kd
# (Kyte J, Doolittle RF. J Mol Biol 157(1):105-32, 1982).
AA_HYDRO = {aa: kd[aa] for aa in CANONICAL_AA}

# Net charge at pH 7. R/K fully protonated, D/E fully deprotonated; H is ~10%
# protonated at pH 7 (pKa ~6.0). No clean BioPython equivalent -- values match
# standard biochemistry references (Lehninger; IPC2 / Bjellqvist pKa tables).
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
