"""Chat interface for asking questions about the loaded article."""

import streamlit as st

from src.app_web.styles import render_page_header


def render_chat_interface(
    backend,
    article_title: str | None,
) -> None:
    """Render the chat UI and process submitted questions."""

    render_page_header(
        title="Chat with wikiBOT",
        subtitle=(
            "Ask questions about the article loaded into your "
            "current knowledge base."
        ),
        eyebrow="RESEARCH ASSISTANT",
    )

    if not backend or not article_title:
        st.info(
            "Load a Wikipedia article from the sidebar before "
            "starting a conversation."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="wikibot-card">
                <h3>💡 Start exploring</h3>
                <p>
                    Ask for a summary, an explanation of a concept,
                    or the relationship between ideas in this article.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display the conversation saved in this Streamlit session.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        f"Ask about {article_title}...",
        key="article_chat_input",
    )

    if not question:
        return

    # Show the user's message immediately in this run.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:    
        with st.chat_message("assistant"):
            with st.spinner("Searching the article and preparing an answer..."):
                answer = backend.ask(question)

            if answer is None:
                answer = "I couldn't generate an answer for that question."
            elif not isinstance(answer, str):
                answer = str(answer)

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Save this question-answer pair to the Q&A records.
        if st.button(
            "➕ Add to Q&A",
            key=f"save_qna_{len(st.session_state.messages)}",
        ):
            try:
                from src.app_web.helpers import add_qna

                add_qna(
                    backend=backend,
                    title=article_title,
                    question=question,
                    answer=answer,
                )

                st.success("Question and answer saved to Q&A!")

            except Exception as error:
                st.error(f"Could not save Q&A: {error}")

    except Exception as error:
        st.error(f"Unable to answer the question: {error}")