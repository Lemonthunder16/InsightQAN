"""
Thin wrapper around the Groq API (replacing local Ollama).
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import requests
from groq import Groq


# Add your API key directly
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DEFAULT_MODEL = "llama-3.1-8b-instant"


class LLMError(RuntimeError):
    pass


def chat(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Call Groq's chat API with streaming, but return final text as a string.
    This keeps the output identical to your original Ollama wrapper.
    """

    if not GROQ_API_KEY:
        raise LLMError("Missing GROQ_API_KEY")

    # Create groq client
    client = Groq(api_key=GROQ_API_KEY)

    try:
        # STREAM=True just like your sample code
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        # Collect streamed tokens
        full_response = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            full_response += token

        return full_response

    except Exception as e:
        raise LLMError(f"Groq API streaming error: {str(e)}")
