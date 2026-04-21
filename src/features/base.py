from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureBlock(Protocol):
    """Contract for a feature category.

    Implementations take the labeled DataFrame and return a feature
    DataFrame indexed by the *subset* of rows they can produce features for.
    Rows the block cannot handle should be dropped from the returned index.
    """

    name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
