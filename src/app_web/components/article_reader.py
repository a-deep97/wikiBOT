"""Article reader UI for wikiBOT."""

import streamlit as st

from src.app_web.helpers import render_article_sections
from src.app_web.styles import render_page_header


from urllib.parse import quote

from src.app_web.helpers import (
    count_words,
    render_article_sections,
    render_status_bar,
)


def _wikipedia_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(
        title.replace(" ", "_"),
        safe="()_,:'",
    )


def render_article_reader(
    backend,
    article_title: str | None,
) -> None:
    """Render the currently loaded Wikipedia article."""

    render_page_header(
        title=article_title or "Article Reader",
        subtitle=(
            "Explore the article, review its sections, "
            "and use the same knowledge base to ask questions."
        ),
    )

    if not backend or not article_title:
        st.info(
            "Search for a Wikipedia article in the sidebar "
            "to start exploring its content."
        )

        st.markdown(
            """
            <div class="wikibot-card">
                <h3>📖 Your research workspace</h3>
                <p>
                    Load an article to read its sections and use
                    wikiBOT to ask questions about the content.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    render_status_bar(backend, article_title)

    try:
        sections = backend.get_article_sections()
    except Exception as error:
        st.error(f"Unable to load article sections: {error}")
        return

    # ---- Overview: quick stats and a link to the source article ----

    stats_columns = st.columns([1, 1, 1, 1.6])

    stats_columns[0].metric("Sections", len(sections))
    stats_columns[1].metric("Words", f"{count_words(sections):,}")
    stats_columns[2].metric("Search chunks", backend.current_chunk_count)

    with stats_columns[3]:
        st.link_button(
            "Open on Wikipedia ↗",
            _wikipedia_url(article_title),
            use_container_width=True,
        )

    # ---- Section browser ----

    with st.container(border=True):
        st.markdown("#### 📄 Article content")

        filter_column, expand_column = st.columns([4, 1])

        with filter_column:
            query = st.text_input(
                "Filter sections",
                placeholder="Filter sections by title or text...",
                label_visibility="collapsed",
                key=f"section_filter_{article_title}",
            )

        with expand_column:
            expand_all = st.toggle(
                "Expand all",
                key=f"expand_all_{article_title}",
            )

        st.divider()

        try:
            shown = render_article_sections(
                backend,
                query=query,
                expand_all=expand_all,
            )
        except Exception as error:
            st.error(f"Unable to render article sections: {error}")
            return

        if query.strip() and shown:
            st.caption(f"Showing {shown} of {len(sections)} sections")