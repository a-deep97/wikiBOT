
"""Saved question-and-answer UI."""

import streamlit as st

from src.app_web.helpers import get_qna, delete_qna
from src.app_web.styles import render_page_header


def render_qna_panel(
    backend,
    article_title: str | None,
) -> None:
    """Display and manage saved Q&A for the selected article."""

    render_page_header(
        title="Saved Q&A",
        subtitle="Review previously saved questions and answers.",
        eyebrow="KNOWLEDGE HISTORY",
    )

    if not backend or not article_title:
        st.info(
            "Load a Wikipedia article before viewing its saved Q&A."
        )
        return

    st.caption(f"Current article: **{article_title}**")

    try:
        qna_records = get_qna(
            backend=backend,
            title=article_title,
        )
    except Exception as error:
        st.error(f"Unable to load saved Q&A: {error}")
        return

    if not qna_records:
        st.markdown(
            """
            <div class="wikibot-card">
                <h3>💬 No saved questions yet</h3>
                <p>
                    Your saved questions and answers will appear
                    here when they are available for this article.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(qna_records)} saved question(s)")

    # Holds the ID of the record awaiting deletion confirmation.
    if "confirm_delete_qna_id" not in st.session_state:
        st.session_state.confirm_delete_qna_id = None

    for index, record in enumerate(qna_records, start=1):
        record_id = record.get("id")
        question = record.get("question", "Untitled question")
        answer = record.get("answer", "")

        with st.expander(
            f"{index}. {question}",
            expanded=(index == 1),
        ):
            st.markdown(answer or "_No answer saved._")

            if record_id is None:
                st.warning(
                    "This record has no ID and cannot be deleted "
                    "through the current interface."
                )
                continue

            # Request confirmation before deleting this record.
            if st.button(
                "🗑️ Delete",
                key=f"delete_qna_{record_id}",
            ):
                st.session_state.confirm_delete_qna_id = record_id

            if (
                st.session_state.confirm_delete_qna_id
                == record_id
            ):
                st.warning(
                    "Are you sure you want to permanently delete "
                    "this saved question and answer?"
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    if st.button(
                        "Yes, delete",
                        key=f"confirm_delete_qna_{record_id}",
                        type="primary",
                    ):
                        try:
                            deleted = delete_qna(
                                backend,
                                record_id,
                            )

                            st.session_state.confirm_delete_qna_id = None

                            if deleted:
                                st.success("Saved Q&A deleted.")
                            else:
                                st.warning(
                                    "This record no longer exists."
                                )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                f"Unable to delete saved Q&A: {error}"
                            )

                with cancel_col:
                    if st.button(
                        "Cancel",
                        key=f"cancel_delete_qna_{record_id}",
                    ):
                        st.session_state.confirm_delete_qna_id = None
                        st.rerun()
