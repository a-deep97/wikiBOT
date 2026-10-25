import html

import streamlit as st


def show_loading() -> None:
    """Show the wikiBOT loading message."""

    st.markdown(
        """
        <div class="wikibot-loader">
            <div class="loader-orbit">
                <div class="loader-core">🤖</div>
            </div>

            <div class="loader-title">
                Waking up wikiBOT
            </div>

            <div class="loader-text">
                Loading the brain, connecting the knowledge base...
            </div>
        </div>

        <style>
        .wikibot-loader {
            height: 70vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        .loader-orbit {
            width: 90px;
            height: 90px;
            border: 3px solid rgba(128, 128, 128, 0.2);
            border-top: 3px solid currentColor;
            border-radius: 50%;
            animation: wikibot-spin 1.2s linear infinite;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .loader-core {
            font-size: 38px;
            animation: wikibot-pulse 1.5s ease-in-out infinite;
        }

        .loader-title {
            margin-top: 24px;
            font-size: 24px;
            font-weight: 600;
        }

        .loader-text {
            margin-top: 8px;
            font-size: 14px;
            opacity: 0.65;
        }

        @keyframes wikibot-spin {
            from {
                transform: rotate(0deg);
            }

            to {
                transform: rotate(360deg);
            }
        }

        @keyframes wikibot-pulse {
            0%, 100% {
                transform: scale(1);
            }

            50% {
                transform: scale(1.12);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chunk_body(content: str) -> str:
    """Remove the "Article:" / "Section:" header lines added to chunks."""
    lines = content.split("\n")
    while lines and lines[0].startswith(("Article: ", "Section: ")):
        lines.pop(0)
    return "\n".join(lines).strip()


def section_question(article_title: str, section_title: str) -> str:
    """Turn a section title into a natural question about it."""
    parts = [part.strip() for part in section_title.split(">")]
    leaf = parts[-1]

    if leaf.lower() == "introduction":
        return f"Give me a short summary of {article_title}."
    if len(parts) > 1:
        return f"What does the article say about {leaf} ({parts[0]})?"
    return f"What does the article say about {leaf}?"


def render_status_bar(backend, article_title: str | None) -> None:
    """Show the loaded article, model, and device as small chips."""
    chips = []
    if article_title:
        chips.append(f"📖 {html.escape(article_title)}")

    llm = getattr(backend, "llm", None)
    if llm is not None:
        device = getattr(getattr(llm, "device", None), "type", "cpu")
        model_name = str(getattr(llm, "model_name", "model"))
        chips.append(f"🤖 {html.escape(model_name.split('/')[-1])}")
        chips.append("⚡ GPU" if device == "cuda" else "🐢 CPU")

    if not chips:
        return

    chip_html = "".join(f'<span class="wikibot-chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="wikibot-chips">{chip_html}</div>', unsafe_allow_html=True)


def render_article_sections(backend) -> None:
    """Render the currently loaded article sections."""

    sections = backend.get_article_sections()

    if not sections:
        return

    for section in sections:
        title = section["title"].strip()
        text = section["text"].strip()

        if not title:
            continue

        with st.expander(title, expanded=False):
            if text:
                st.write(text)
            else:
                st.caption("No content available.")


def add_qna(backend, title: str, question: str, answer: str) -> bool:
    """
    Add a Q&A record through the backend.

    The actual backend implementation will be added later.
    """

    # TODO: Call backend API to persist the Q&A record.
    # backend.add_qna(
    #     title=title,
    #     question=question,
    #     answer=answer,
    # )
    print("adding qna record:", title, question, answer)
    return True


def get_qna(backend, title: str | None = None) -> list[dict[str, str]]:
    """
    Fetch saved Q&A records through the backend.

    The actual backend implementation will be added later.
    """

    # TODO: Call backend API to fetch saved Q&A records.
    # return backend.get_qna(title=title)
    print("fetching qna records for:", title)
    return []