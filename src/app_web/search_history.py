import json
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "search_history.json"
MAX_RECENT_SEARCHES = 10


def load_search_history() -> list[str]:
    """Load recent searches from disk."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)

        if not isinstance(history, list):
            return []

        return [
            item for item in history
            if isinstance(item, str)
        ][:MAX_RECENT_SEARCHES]

    except (json.JSONDecodeError, OSError):
        return []


def save_search_history(history: list[str]) -> None:
    """Persist recent searches to disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history[:MAX_RECENT_SEARCHES], file, indent=2)


def add_search(title: str) -> list[str]:
    """Add a search to history, keeping the newest first."""
    title = title.strip()

    if not title:
        return load_search_history()

    history = load_search_history()
    history = [item for item in history if item.casefold() != title.casefold()]
    history.insert(0, title)
    history = history[:MAX_RECENT_SEARCHES]

    save_search_history(history)
    return history