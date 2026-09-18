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

        print(f"[MODEL] Loading {self.model_name}")

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

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 10
    ) -> str:

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            input_length = inputs["input_ids"].shape[1]

            generated_tokens = outputs[0][input_length:]

            answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            return answer

        except Exception:
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
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_length = inputs.shape[1]

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

    def test_generation(self):
        prompt = "What is Google?"

        print("\n[MODEL TEST]")
        print("Generating...")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        print("[MODEL TEST] Tokenization complete")
        print(f"[MODEL TEST] Input tokens: {inputs['input_ids'].shape}")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        print("[MODEL TEST] Generation complete")

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print("[MODEL TEST] Result:")
        print(result)