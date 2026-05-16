"""Configuration for TurbovecTier."""
from dataclasses import dataclass

@dataclass
class TurbovecConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    bit_width: int = 4
    oversample_factor: int = 5
    max_dense_candidates: int = 500
    storage_path: str = "./turbovec_data"
