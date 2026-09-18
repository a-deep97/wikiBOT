AVAILABLE_MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "type": "causal",
    },
    "flan": {
        "name": "google/flan-t5-large",
        "type": "seq2seq",
    },
    "mistral": {
        "name": "mistral-community/Mistral-7B-Instruct-v0.3",
        "type": "causal",
    },
}

DEFAULT_MODEL = next(iter(AVAILABLE_MODELS))