"""
Tiny local embedding model for Railway.
No Groq embeddings (Groq does NOT support embedding models).
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from typing import List

# Very small model (~25 MB)
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    model = _get_model()
    return list(model.encode(texts, convert_to_numpy=True))


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
