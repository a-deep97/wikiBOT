import click

from data.wikipedia import fetch_wikipedia_text
from rag.pipeline import generate_answer

@click.command()
def askwiki():
    click.secho("wikiBot (CLI)", fg="cyan", bold=True)


context = None
title = None

while True:
    if not context:
        title = click.prompt("Enter Wikipedia Page Title")

        try:
            context = fetch_wikipedia_text(title)
            click.secho(
                f"Loaded article for '{title}'",
                fg="green"
            )
        except Exception as e:
            click.secho(str(e), fg="red")
            continue

    click.secho(
        "\nAsk a question "
        "(type 'new' for new topic, or 'exit' to quit):",
        fg="yellow"
    )

    question = click.prompt("Your question")

    if question.lower() in ["exit", "quit"]:
        click.secho(" Goodbye!", fg="cyan")
        break

    elif question.lower() == "new":
        context = None
        title = None
        continue

    try:
        answer = generate_answer(question, context)
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
