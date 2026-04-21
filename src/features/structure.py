from __future__ import annotations

import pandas as pd


class StructureFeatures:
    """Placeholder for pocket / 3D structure features."""

    name = "structure"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Structure features are not implemented yet. "
            "Add pocket-based feature extraction here."
        )
