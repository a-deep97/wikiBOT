
"""Sidebar article search for wikiBOT."""

import streamlit as st

from src.app_web.components.search_panel import (
    render_search_panel,
)


def render_sidebar(
    search_function,
    recent_searches: list[str],
) -> str | None:
    """Render the sidebar and return the selected article title."""

    with st.sidebar:
        st.markdown("## 🧠 wikiBOT")
        st.caption("Your Wikipedia research assistant")
        st.divider()

        selected_article = render_search_panel(
            search_function=search_function,
            recent_searches=recent_searches,
        )

    return selected_article