
import streamlit as st

from src.embeddings.embedder import Embedder
from src.models.model import Model


@st.cache_resource(show_spinner="Loading embedding model...")
def get_shared_embedder() -> Embedder:
    """Create one cached embedder instance."""
    return Embedder()


@st.cache_resource(show_spinner="Loading language model...")
def get_shared_llm(model_key: str) -> Model:
    """Create one cached LLM instance per model key."""
    return Model(model_key)
