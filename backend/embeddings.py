"""
Small embedding model for Railway-safe deployments.
Uses a lightweight SentenceTransformer that downloads quickly.
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List
import numpy as np


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # SMALL model (very fast + tiny download)
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


def embed_texts(texts: List[str]) -> list[np.ndarray]:
    model = _get_model()
    return list(model.encode(texts, convert_to_numpy=True))


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
