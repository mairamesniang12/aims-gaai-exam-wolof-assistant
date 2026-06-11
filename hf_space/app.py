"""Gradio Space — Wolof Educational Assistant (LoRA fine-tuned Qwen3-0.6B)"""
from __future__ import annotations
import os
import re
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "niangmariame513/wolof-assistant-qwen3")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful Wolof language educational assistant trained at AIMS Sénégal. "
    "Answer clearly and concisely. If you do not know the answer, say so honestly.",
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)

EXAMPLES = [
    ["Ana nga?"],
    ["Yal na Yàlla baal ma."],
    ["Wolof dafa am nit yu bare?"],
    ["What is the Wolof word for school?"],
    ["Jëf-jëf bu baax."],
]

DISCLAIMER = """
⚠️ **Limitations:**
- This is a classroom demo model, not a production assistant.
- Wolof is a low-resource language — outputs may be imperfect.
- The model was fine-tuned on ~6300 examples only.
- Do not use for critical or sensitive decisions.
"""


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_answer(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    return text.strip()


def load_model():
    print(f"Loading tokenizer from {MODEL_REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Loading LoRA adapter from {MODEL_REPO_ID}...")
    model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    print("Model ready!")
    return model, tokenizer


MODEL, TOKENIZER = load_model()
DEVICE = pick_device()


def render_prompt(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate(user_message: str, max_new_tokens: int, temperature: float) -> str:
    if not user_message.strip():
        return "⚠️ Please enter a prompt."

    prompt = render_prompt(user_message)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    do_sample = temperature > 0.0
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": TOKENIZER.eos_token_id,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **kwargs)

    generated = output_ids[0, inputs["input_ids"].shape[-1]:]
    return clean_answer(TOKENIZER.decode(generated, skip_special_tokens=True))


with gr.Blocks(title="Wolof Assistant — AIMS GAAI") as demo:
    gr.Markdown("""
# 🇸🇳 Wolof Educational Assistant
**Fine-tuned Qwen3-0.6B with LoRA — AIMS Sénégal GAAI Exam Project**

This assistant was trained on 3 Wolof data sources:
- CohereLabs AYA dataset (general)
- Soynade Wolof orthography dataset (domain)
- Synthetic Wolof instructions (generated)

*Instructor: Dr. Papa-Séga WADE*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="Your message (Wolof or English)",
                placeholder="Ex: Ana nga? / What does 'jërejëf' mean?",
                lines=4
            )
            with gr.Row():
                max_tokens = gr.Slider(16, 256, value=128, step=8, label="Max tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            submit_btn = gr.Button("Send", variant="primary")

        with gr.Column(scale=2):
            output_box = gr.Textbox(label="Assistant response", lines=8)

    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_box,
        label="Example prompts"
    )

    gr.Markdown(DISCLAIMER)

    submit_btn.click(
        fn=generate,
        inputs=[prompt_box, max_tokens, temperature],
        outputs=output_box
    )
    prompt_box.submit(
        fn=generate,
        inputs=[prompt_box, max_tokens, temperature],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
