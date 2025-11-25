"""
Groq-based embedding wrapper.
Fully replaces SentenceTransformer while keeping the same return format.
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Any
from groq import Groq


# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable.")

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)

# Choose a small + fast embedding model
EMBED_MODEL = "nomic-embed-text-v1"   # Very fast, high-quality 768-dimensional vectors


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Returns list of numpy arrays (one embedding per text),
    exactly like SentenceTransformer encode().
    """

    # Groq supports batch embedding natively
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    out = []
    for item in resp.data:
        # Convert python list -> numpy array for compatibility
        vec = np.array(item.embedding, dtype=np.float32)
        out.append(vec)

    return out


def embed_text(text: str) -> np.ndarray:
    """Single text embedding (matches original behavior)."""
    return embed_texts([text])[0]
