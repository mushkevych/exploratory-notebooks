from __future__ import annotations

import os

from crewai import LLM


def build_local_llm(model: str = "ollama/llama3.1:8b") -> LLM:
    """Create a CrewAI LLM client backed by host-local Ollama."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    llm_instance = LLM(model=model, base_url=base_url, temperature=0.1)
    print(f"Created LLM instance: {llm_instance}")
    return llm_instance
