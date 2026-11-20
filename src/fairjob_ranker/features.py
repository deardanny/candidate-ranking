"""Fit on train once; persist the exact transform for validation and serving."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .schema import CATEGORICAL, FEATURES, NUMERICAL, VERSION


def validate_features(frame: pd.DataFrame) -> None:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Feature input must be nonempty and finite.")
    values = frame[CATEGORICAL].to_numpy()
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Categorical codes must be integers.")


@dataclass
class FeatureTransformer:
    vocabularies: dict[str, list[int]]
    means: dict[str, float]
    scales: dict[str, float]

    def __post_init__(self):
        # Build lookup indexes once at fit/load, not once per HTTP request.
        self._indices = {c: pd.Index(self.vocabularies[c], dtype="int64") for c in CATEGORICAL}
        for index in self._indices.values():
            index.get_indexer([0])  # Warm the lookup engine before concurrent reads.
        self._means = np.array([self.means[c] for c in NUMERICAL], dtype=np.float64)
        self._scales = np.array([self.scales[c] for c in NUMERICAL], dtype=np.float64)

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "FeatureTransformer":
        validate_features(train)
        vocabularies = {c: sorted(int(v) for v in train[c].unique()) for c in CATEGORICAL}
        means = {c: float(train[c].mean()) for c in NUMERICAL}
        scales = {c: float(train[c].std(ddof=0)) or 1.0 for c in NUMERICAL}
        return cls(vocabularies, means, scales)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_features(frame)
        columns = {}
        for column in CATEGORICAL:
            # Even raw code 0 becomes a known positive index. Index 0 is OOV only.
            columns[column] = (self._indices[column].get_indexer(frame[column]) + 1).astype("int32")
        numerical = ((frame[NUMERICAL].to_numpy(dtype=np.float64) - self._means) / self._scales).astype("float32")
        columns.update({column: numerical[:, i] for i, column in enumerate(NUMERICAL)})
        result = pd.DataFrame(columns, index=frame.index)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Feature transformation overflowed.")
        return result[FEATURES]

    def save(self, path: Path) -> None:
        write_json(path, {
            "schema_version": VERSION, "feature_order": FEATURES, "fitted_on": "train",
            "oov_index": 0, "vocabularies": self.vocabularies, "means": self.means, "scales": self.scales,
        })

    @classmethod
    def load(cls, path: Path) -> "FeatureTransformer":
        saved = read_json(path)
        if saved["schema_version"] != VERSION or saved["feature_order"] != FEATURES:
            raise ValueError("Feature artifact schema mismatch.")
        return cls(saved["vocabularies"], saved["means"], saved["scales"])

