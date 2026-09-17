import click

from .data.wikipedia import fetch_wikipedia_text
from .data.processor import chunk_text
from .embeddings.embedder import Embedder
from .vector.faiss_store import FAISSStore
from .models.model import Model
from .pipeline import RAGPipeline


@click.command()
def askwiki():
    click.secho("wikiBot (CLI)", fg="cyan", bold=True)

    embedder = Embedder()
    llm = Model()

    rag_pipeline = None
    title = None

    while True:

        if rag_pipeline is None:
            title = click.prompt("Enter Wikipedia Page Title")

            try:
                text = fetch_wikipedia_text(title)

                chunks = chunk_text(text)

                if not chunks:
                    raise ValueError(
                        "Wikipedia article did not produce any chunks."
                    )

                embeddings = embedder.embed_documents(chunks)

                dimension = embeddings.shape[1]

                vector_store = FAISSStore(
                    dimension=dimension
                )

                vector_store.add(
                    embeddings,
                    chunks
                )

                rag_pipeline = RAGPipeline(
                    embedder=embedder,
                    vector_store=vector_store,
                    llm=llm,
                    top_k=3
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
