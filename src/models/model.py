import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from .data import AVAILABLE_MODELS, DEFAULT_MODEL


class Model:
    def __init__(self, model_key: str = DEFAULT_MODEL):
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        config = AVAILABLE_MODELS[model_key]

        self.model_key = model_key
        self.model_name = config["name"]
        self.model_type = config["type"]

        self.device = self._detect_device()

        print(f"[MODEL] Loading {self.model_name}")
        print(f"[MODEL] Device: {self.device}")

        if self.device.type == "cuda":
            print(
                f"[MODEL] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )

        elif self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            )

        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}"
            )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"[MODEL] Model device: "
            f"{next(self.model.parameters()).device}"
        )

    @staticmethod
    def _detect_device() -> torch.device:
        """
        Select CUDA when available.
        Fall back to CPU otherwise.
        """

        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "\n[WARNING] CUDA is not available."
        )
        print(
            "[WARNING] Falling back to CPU inference."
        )
        print(
            "[WARNING] LLM generation may be significantly slower.\n"
        )

        return torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100
    ) -> str:

        try:
            if self.model_type == "causal":
                return self._generate_causal(
                    prompt,
                    max_new_tokens
                )

            return self._generate_seq2seq(
                prompt,
                max_new_tokens
            )

        except Exception:
            import traceback

            print("\n========== MODEL ERROR ==========")
            traceback.print_exc()
            print("=================================\n")

            raise

    def _generate_causal(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )

        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
        else:
            input_ids = inputs

        input_ids = input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_length = input_ids.shape[-1]

        generated_tokens = outputs[0][input_length:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

    def _generate_seq2seq(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()