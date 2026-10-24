"""Wikipedia search with recent article suggestions."""

import streamlit as st
from streamlit_searchbox import st_searchbox


def render_search_panel(
    search_function,
    recent_searches: list[str],
) -> str | None:
    """Render autocomplete with recent articles in the dropdown."""

    st.subheader("🔎 Find an article")

    recent_options = recent_searches[:8]

    def search_with_recents(query: str) -> list[str]:
        """Show recent articles for an empty query."""
        if not query or not query.strip():
            return recent_options

        return search_function(query.strip())

    selected_article = st_searchbox(
        search_function=search_with_recents,
        placeholder="Type a Wikipedia article...",
        label="Wikipedia article",
        key="wiki_article_search",
        debounce=300,
        default_options=recent_options,
    )

    pending_article = st.session_state.pop(
        "pending_article",
        None,
    )

    if pending_article:
        selected_article = pending_article

    return selected_article