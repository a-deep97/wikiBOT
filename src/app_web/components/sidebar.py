"""Sidebar navigation for wikiBOT."""

import streamlit as st

from src.app_web.components.search_panel import (
    render_search_panel,
)


def render_sidebar(
    search_function,
    recent_searches: list[str],
) -> tuple[str, str | None]:
    """
    Render the application sidebar.

    Returns:
        A tuple containing the active page and selected article title.
    """

    with st.sidebar:
        st.markdown("# 🧠 wikiBOT")
        st.caption("Your personal Wikipedia research assistant")
        st.divider()

        active_section = st.radio(
            "WORKSPACE",
            options=[
                "📖 Article Reader",
                "💬 Chat",
                "📚 QnA",
            ],
            key="active_section",
            label_visibility="visible",
        )

        st.divider()

        selected_article = render_search_panel(
            search_function=search_function,
            recent_searches=recent_searches,
        )

        st.divider()
        st.caption("Powered by retrieval-augmented generation")

    return active_section, selected_article