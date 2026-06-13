# Individual Technical Note
**Student:** Mariame Niang  
**GitHub:** mairamesniang12  
**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  

---

## 1. What exact part did I implement?

Within our group of five, I was responsible for the **model training and
the Hugging Face deployment** (Steps 2, 5, 6, and 8 of the pipeline).

**Training** (`train_lora_assistant_only.py`): I ran the full LoRA training on
Google Colab T4 GPU with assistant-only -100 label masking. System and user
tokens were masked (label = -100), so the model only learned to generate
assistant responses. Total trainable parameters: 10,092,544 (1.67% of the full
model). Training ran for 3 epochs over 5670 examples prepared by Fatima.

**Hugging Face Hub deployment** (`push_to_hub.py`): I pushed the trained LoRA
adapter to `niangmariame513/wolof-assistant-qwen3` and configured the
repository as public. I also set up the HF authentication workflow for the team
using `hf auth login`.

**Hugging Face Space** (`hf_space/app.py`): I modified the starter `app.py`
to use `gr.Blocks` instead of `gr.Interface`, added example Wolof prompts,
and deployed the Space at `niangmariame513/wolof-assistant-demo`. I also
replaced the placeholder `MODEL_REPO_ID` with the real adapter repository.

**GitHub**: I managed the group GitHub repository at
`github.com/mairamesniang12/aims-gaai-exam-wolof-assistant`, handling the
remote setup, merge conflicts, and final push.

The other group members contributed as follows:
- **Fatima**: data preparation and documentation (3 sources, chat formatting, splits)
- **Fatou Bintou**: evaluation metrics and model card
- **Christian**: `context_state_machine.py` improvement and qualitative testing
- **Jean**: project report and demo preparation

---

## 2. Which technical choice did I make, and what alternative did I reject?

**Choice: Google Colab T4 GPU over local CPU.**  
I first tested training locally on CPU (Windows, `aims-sen-cv-2026` conda
environment) and observed that 100 examples took 41 minutes. The full dataset
(5670 examples, 3 epochs) would have taken several days. I switched to Google
Colab T4 GPU, which completed the full training in approximately 2.5 hours.

**Choice: Gradio `gr.Blocks` interface over simple `gr.Interface`.**  
The starter `app.py` used `gr.Interface`, which is less flexible. I replaced it
with `gr.Blocks` to add a structured layout, example prompts in Wolof, and a
visible limitations/guardrail section as required by the exam rubric.

**Choice: Keeping Wolof over switching to English or French.**  
Wolof is spoken by over 10 million people in Senegal and represents a genuine
low-resource African language challenge. Three real public datasets were
available (AYA, Soynade, Synthetic), making the methodology reproducible.
English or French would have been easier but less meaningful for AIMS Sénégal.

---

## 3. What problem or failure did I observe, and how did I diagnose it?

**Problem 1 — torchao version conflict.**  
When launching training on Colab, the script crashed with:
```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported.
```
I diagnosed this by reading the full traceback, which pointed to
`peft/tuners/lora/torchao.py`. The fix was to install the correct version
before training: `pip install torchao==0.16.0`.

**Problem 2 — Hugging Face token 401 Unauthorized.**  
The stored HF token was invalid (previously revoked for security reasons).
`python push_to_hub.py --check-auth` returned `HfHubHTTPError: Invalid user
token`. I diagnosed this by running `hf auth whoami` and confirmed the token
was expired. I generated a new Write token (`aims-exam-2026`) at
huggingface.co/settings/tokens and re-authenticated with `hf auth login --force`.

**Problem 3 — Model responds in English to Wolof input.**  
During deployment testing on the Space, the prompt "Ana nga?"
(meaning "How are you?" in Wolof) generated the response "What?" instead of
the expected "Mangi fi rekk". This is a documented limitation reported by
Fatou Bintou in the evaluation, and I included it in the model card limitations
section.

---

## 4. How did I verify that my contribution works?

**Training verification:** The training loss decreased from 5.536 at step 1
to approximately 0.037 at the final steps (epoch 2.4+). The validation loss
decreased from 1.556 at epoch 0.14 to 0.145 at epoch 2.26, confirming the
model was learning and generalizing without severe overfitting.

**Deployment verification:** The Hugging Face Space at
`niangmariame513/wolof-assistant-demo` shows status **Running** (green) and
responds to user prompts. A test with "Ana nga?" produced a response, confirming
end-to-end inference works from Hub model to deployed app.

**Hub verification:** Running `python push_to_hub.py --check-auth` returned:
```
Authenticated Hugging Face user: niangmariame513
```
The model repository at `niangmariame513/wolof-assistant-qwen3` is publicly
accessible with all adapter files (`adapter_config.json`,
`adapter_model.safetensors`, `tokenizer` files).

---

## 5. If I had one extra day, what would I improve first and why?

I would improve the **Hugging Face Space** by adding a side-by-side comparison
between the base Qwen3-0.6B model (without the LoRA adapter) and the fine-tuned
model on the same Wolof prompt. This would visually demonstrate the added value
of fine-tuning during the final demo.

A secondary improvement would be to add a loading indicator and error message
in the Gradio interface for cases where the model takes too long to respond or
generates an empty output, making the Space more robust for the presentation.
