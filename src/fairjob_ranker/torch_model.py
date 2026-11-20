"""Shared CPU model, tensor conversion and verified inference bundle."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import FeatureTransformer
from .io import read_json, sha256
from .schema import CATEGORICAL, FEATURES, NUMERICAL, VERSION


class EmbeddingMLP(nn.Module):
    def __init__(self, category_sizes: list[int], embedding_dim: int = 8,
                 hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim, padding_idx=0) for size in category_sizes
        ])
        width = len(category_sizes) * embedding_dim + len(NUMERICAL)
        layers = []
        for hidden in hidden_dims if hidden_dims is not None else [128, 64]:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([*embedded, numerical], dim=1)).squeeze(1)


def to_tensors(transformed: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """The same column order and dtypes are used in training and HTTP inference."""
    if transformed.columns.tolist() != FEATURES:
        raise ValueError("Transformed feature order mismatch.")
    return (
        torch.from_numpy(transformed[CATEGORICAL].to_numpy(dtype=np.int64, copy=True)),
        torch.from_numpy(transformed[NUMERICAL].to_numpy(dtype=np.float32, copy=True)),
    )


@torch.inference_mode()
def predict_tensors(model: EmbeddingMLP, categorical: torch.Tensor, numerical: torch.Tensor,
                    batch_size: int = 8192) -> np.ndarray:
    model.eval()
    parts = [torch.sigmoid(model(categorical[start:start + batch_size], numerical[start:start + batch_size])).numpy()
             for start in range(0, len(categorical), batch_size)]
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0, dtype=np.float64)


@dataclass
class ModelBundle:
    model: EmbeddingMLP
    transformer: FeatureTransformer
    metadata: dict
    fallback: dict
    directory: Path

    def predict(self, raw_features: pd.DataFrame) -> np.ndarray:
        return predict_tensors(self.model, *to_tensors(self.transformer.transform(raw_features)))


def load_bundle(directory: Path) -> ModelBundle:
    metadata = read_json(directory / "metadata.json")
    if metadata["schema_version"] != VERSION or metadata["feature_order"] != FEATURES:
        raise ValueError("Model bundle schema mismatch.")
    for name in ["weights.pt", "features.json", "ad_catalog.parquet", "fallback.json"]:
        if sha256(directory / name) != metadata["artifact_sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if metadata["feature_version"] != metadata["artifact_sha256"]["features.json"] or metadata["catalog_version"] != metadata["artifact_sha256"]["ad_catalog.parquet"]:
        raise ValueError("Bundle feature or catalog version mismatch.")
    transformer = FeatureTransformer.load(directory / "features.json")
    sizes = [len(transformer.vocabularies[c]) + 1 for c in CATEGORICAL]
    if sizes != metadata["architecture"]["category_sizes"]:
        raise ValueError("Vocabulary and embedding sizes do not match.")
    model = EmbeddingMLP(**metadata["architecture"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    fallback = read_json(directory / "fallback.json")
    values = [fallback["train_ctr"], *fallback["scores"].values()]
    if fallback["fitted_on"] != "train" or not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
        raise ValueError("Invalid fallback scores.")
    return ModelBundle(model, transformer, metadata, fallback, directory)
