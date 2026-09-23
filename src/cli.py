import click

from .backend.manager import WikiBotBackend
from .models.data import AVAILABLE_MODELS, DEFAULT_MODEL


def choose_model() -> str:
    """Display available models and return the selected model key."""

    click.echo("\nAvailable models:")

    model_keys = list(AVAILABLE_MODELS.keys())

    for index, key in enumerate(model_keys, start=1):
        model = AVAILABLE_MODELS[key]

        default_marker = (
            " (default)"
            if key == DEFAULT_MODEL
            else ""
        )

        click.echo(
            f"{index}. {key} - "
            f"{model['name']}"
            f"{default_marker}"
        )

    choice = click.prompt(
        "\nSelect model",
        default="1"
    )

    try:
        index = int(choice) - 1
    except ValueError:
        click.echo(
            "Invalid selection. Using default model."
        )
        return DEFAULT_MODEL

    if index < 0 or index >= len(model_keys):
        click.echo(
            "Invalid selection. Using default model."
        )
        return DEFAULT_MODEL

    return model_keys[index]


def ask_questions(
    backend: WikiBotBackend,
    article_title: str
):
    """Run the question-answer loop for an article."""

    click.echo(
        f"\nLoaded article: {article_title}"
    )

    click.echo(
        "\nAsk questions about the article."
    )

    click.echo(
        "Type 'new' to load another article."
    )

    click.echo(
        "Type 'exit' to quit."
    )

    while True:

        question = click.prompt(
            "\nQuestion",
            prompt_suffix=": "
        )

        command = question.strip().lower()

        if command == "exit":
            return "exit"

        if command == "new":
            return "new"

        try:
            answer = backend.ask(question)

            click.echo(
                f"\nAnswer: {answer}"
            )

        except Exception as error:

            click.echo(
                f"\nError generating answer: {error}"
            )


@click.command()
def main():
    """Start wikiBOT."""

    click.echo(
        "\n================================"
    )

    click.echo(
        "          wikiBOT"
    )

    click.echo(
        "================================"
    )

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------

    model_key = choose_model()

    click.echo(
        f"\nSelected model: {model_key}"
    )

    # --------------------------------------------------
    # Create backend
    # --------------------------------------------------

    backend = WikiBotBackend(
        model_key=model_key,
        top_k=3,
        candidate_k=10,
        similarity_threshold=0.35
    )

    try:

        # Load shared resources once.
        backend.load_resources()

        # --------------------------------------------------
        # Article loop
        # --------------------------------------------------

        while True:

            article_title = click.prompt(
                "\nWikipedia article"
            )

            article_title = article_title.strip()

            if not article_title:
                click.echo(
                    "Please enter an article title."
                )
                continue

            try:

                backend.load_knowledge(
                    article_title
                )

            except Exception as error:

                click.echo(
                    f"\nError loading article: {error}"
                )

                continue

            result = ask_questions(
                backend,
                article_title
            )

            if result == "exit":
                break

            if result == "new":
                continue

    finally:

        backend.close()

        click.echo(
            "\nGoodbye!"
        )


if __name__ == "__main__":
    main()