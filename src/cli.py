import click

from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_MODEL
from src.vector.bm25_store import BM25Store

from .data.wikipedia import fetch_wikipedia_article
from .data.processor import chunk_text
from .embeddings.embedder import Embedder
from .vector.faiss_store import FAISSStore
from .models.model import Model
from .pipeline import RAGPipeline
from .database.sqlite_store import SQLiteStore
from .knowledge.manager import KnowledgeManager

database = SQLiteStore()

knowledge_manager = KnowledgeManager(
    database
)

def select_model() -> str:
    print("\nAvailable models:")
    print("-----------------")

    model_keys = list(AVAILABLE_MODELS.keys())

    for index, key in enumerate(model_keys, start=1):
        model = AVAILABLE_MODELS[key]

        default_marker = (
            " (default)"
            if key == DEFAULT_MODEL
            else ""
        )

        print(
            f"{index}. {key} - "
            f"{model['name']}"
            f"{default_marker}"
        )

    print()

    while True:
        selection = input(
            f"Select model [{DEFAULT_MODEL}]: "
        ).strip()

        # Enter → default model
        if not selection:
            return DEFAULT_MODEL

        if selection in AVAILABLE_MODELS:
            return selection

        print(
            f"Invalid selection '{selection}'. "
            f"Choose one of: "
            f"{', '.join(model_keys)}"
        )
        
@click.command()
def askwiki():
    click.secho("wikiBot (CLI)", fg="cyan", bold=True)

    embedder = Embedder()
    model_key = select_model()
    llm = Model(model_key)

    rag_pipeline = None
    title = None

    while True:

        if rag_pipeline is None:
            title = click.prompt("Enter Wikipedia Page Title")

            try:
                chunks = knowledge_manager.get_article(
                    title
                )

                if not chunks:
                    raise ValueError(
                        "Wikipedia article did not produce any chunks."
                    )

                chunk_texts = [
                    chunk["content"]
                    for chunk in chunks
                ]

                embeddings = embedder.embed_documents(
                    chunk_texts
                )

                dimension = embeddings.shape[1]

                vector_store = FAISSStore(
                    dimension=dimension
                )

                vector_store.add(
                    embeddings,
                    chunks
                )

                bm25_store = BM25Store()
                bm25_store.build(chunks)

                rag_pipeline = RAGPipeline(
                    embedder=embedder,
                    vector_store=vector_store,
                    bm25_store=bm25_store,
                    llm=llm,
                    top_k=3,
                    candidate_k=10,
                    similarity_threshold=0.35
                )

                click.secho(
                    f"Loaded article for '{title}'",
                    fg="green"
                )

                click.secho(
                    f"Created {len(chunks)} chunks",
                    fg="green"
                )

            except Exception as e:
                click.secho(
                    f"Error: {str(e)}",
                    fg="red"
                )
                continue

        click.secho(
            "\nAsk a question "
            "(type 'new' for new topic, or 'exit' to quit):",
            fg="yellow"
        )

        question = click.prompt("Your question")

        if question.lower() in ["exit", "quit"]:
            click.secho(
                "Goodbye!",
                fg="cyan"
            )
            break

        if question.lower() == "new":
            rag_pipeline = None
            title = None
            continue

        try:
            answer = rag_pipeline.ask(question)

            click.secho(
                f"Answer: {answer}\n",
                fg="green"
            )

        except Exception as e:
            click.secho(
                f"Error: {str(e)}",
                fg="red"
            )


if __name__ == "__main__":
    askwiki()
