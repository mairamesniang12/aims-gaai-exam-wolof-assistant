# Model Card: Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B

## Model Summary

- **Base model:** `Qwen/Qwen3-0.6B`
- **Adaptation method:** LoRA / PEFT
- **Target task:** Instruction following in Wolof — answering educational
  questions, language learning assistance, and general Wolof conversation
- **Target users:** Senegalese learners and speakers who want a digital
  assistant in their native language; researchers interested in low-resource
  African language NLP
- **Target language/domain:** Wolof (Senegambian language, 10+ million speakers)
- **Hugging Face model repo:**
  https://huggingface.co/niangmariame513/wolof-assistant-qwen3
- **Hugging Face Space:**
  https://huggingface.co/spaces/niangmariame513/wolof-assistant-demo

---

## Intended Use

This model is designed to assist Wolof speakers with basic educational
questions and language learning tasks. It can:

- Answer simple questions in Wolof about vocabulary, grammar, and culture
- Assist with orthographic correction of Wolof text
- Provide short explanations on topics covered in the training data
  (greetings, proverbs, daily life, geography of African capitals)
- Serve as a classroom demonstration of low-resource LLM fine-tuning

This adapter is intended for **educational and research use only**.

---

## Out-of-Scope Use

This model should **not** be used for:

- Medical, legal, or financial advice in any language
- Tasks requiring factual accuracy about current events (no web access)
- Wolof-to-French or Wolof-to-English translation
  (translation tasks scored F1=0.00 in evaluation)
- Production applications without additional safety evaluation
- Voice-based interfaces (Wolof is not natively supported by Whisper ASR)
- Any task requiring guaranteed factual correctness

---

## Data Methodology

Three separated data sources were used. Each source was kept in a distinct
file throughout the pipeline.

| Source | Type | Size | License/Access | Cleaning Method | Role |
|--------|------|------|----------------|-----------------|------|
| CohereLabs/aya_dataset (Wolof subset) | Public | 500 examples | Apache 2.0 / Public HF Hub | Filtered to Wolof language only; limited to 500 examples | train/val/eval |
| soynade-research/Wolof-Non-Standard-Orthography | Public | 500 examples | Public HF Hub | Kept as-is; non-standard orthography documented as limitation | train/val/eval |
| syntetic_wolof_instruct_data.jsonl (provided starter) | Synthetic | 5300 examples | Provided by course instructor | No filtering applied; already in correct instruction format | train/val/eval |

---

## Data Splits

| Split | Number of examples | Ratio | Notes |
|-------|--------------------|-------|-------|
| Train | 5670 | 90% | Split per source family with deterministic seed |
| Validation | 315 | 5% | Used for checkpoint selection during training |
| Evaluation | 315 | 5% | Held-out; used for all reported metrics |

Splits were generated **per source family** before merging to prevent
data leakage. The combined eval file `data/splits/eval_all.jsonl` contains
25 AYA + 25 Soynade + 265 Synthetic examples.

---

## Chat Template and Training Labels

Every example was converted from raw instruction format to the following
chat template using `src/download_datasets.py`:

```text
system:    You are a helpful Wolof language educational assistant trained
           at AIMS Sénégal. Answer clearly and concisely. If you do not
           know the answer, say so honestly.
user:      <instruction> <input>
assistant: <output>
```

**Training labels:**

- `system` tokens: `-100` (ignored in loss — model does not learn to reproduce system prompt)
- `user` tokens: `-100` (ignored in loss — model does not learn to copy the question)
- `padding` tokens: `-100` (ignored in loss — padding does not pollute gradients)
- `assistant` output tokens: learned by the model (normal token IDs in labels)

This assistant-only masking was verified by tokenization statistics:

```
avg_total tokens per example:       129.0
avg_supervised tokens per example:   33.9
examples kept: 5670 / 5670 (skipped: 0)
```

Only 26% of tokens per example contributed to the training loss.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-0.6B |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 10,092,544 (1.67% of 606,142,464 total) |
| Learning rate | 1e-4 (cosine schedule with warmup) |
| Epochs | 3 |
| Training examples | 5670 |
| Hardware | Google Colab T4 GPU (15 GB VRAM) |
| Training duration | ~2.5 hours |
| Max sequence length | 1024 tokens |
| Checkpoint policy | best_adapter (min val loss) + latest_adapter (end of training) |
| Optimizer | AdamW |

---

## Evaluation

Evaluation was performed on the held-out eval set (315 examples) using
`evaluation.py --generate --model-choice qwen --adapter auto`.

**Automatic metrics:**

| Metric | Value |
|--------|-------|
| Exact Match | 0.3492 (34.9%) |
| Token F1 | 0.7499 (75.0%) |
| BLEU | 0.6665 (66.7%) |
| ROUGE-L | 0.7456 (74.6%) |

**Representative examples:**

| Prompt | Reference | Model Output | Comment |
|--------|-----------|--------------|---------|
| "Alger moy péeyu ban réw?" | "Alger moy péeyu réwum Algérie" | "Alger moy péyum réwum Algiers" | Partial success: correct content, surface spelling errors. F1=0.60, BLEU=0.39 |
| "Ndàx man ngaa jubbanti mbind mii? Bu ñu leen waxee..." | "Waaw! Mi ngii: Bu ñu léen wàxee..." | "Waaw! Mi ngii: Jubbaanj ti mbind mi..." | Partial success: correct opening, diverges after. F1=0.38, BLEU=0.11 |
| "Moscou mooy gëblag man réew?" | "Moscou mooy gëblag réewum Russie." | "Moscou mooya réewum Moscou." | Failure: repeated city name instead of country. F1=0.44, BLEU=0.31 |
| "wagnil ma : benn, ñaar, ñett..." | "un, deux, trois, quatre..." | "Benn: Naa ngi ci, mën na nekk..." | Failure: did not translate to French, responded in Wolof. F1=0.00, BLEU=0.04 |
| "Yamoussoukro mooy gëblag man réew?" | "Yamoussoukro mooy gëblag réewum Côte d'Ivoire." | "Yamoussuksu kër gi 'Réew?' ak 'gëblàg.'..." | Failure: hallucination, wrong content entirely. F1=0.00, BLEU=0.09 |

---

## Deployment

- **Model Hub URL:**
  https://huggingface.co/niangmariame513/wolof-assistant-qwen3
- **Space URL:**
  https://huggingface.co/spaces/niangmariame513/wolof-assistant-demo
- **Inference framework:** Gradio (`gr.Blocks`)
- **Required hardware:** CPU sufficient (no GPU required for inference);
  GPU recommended for faster response times
- **Average latency:** ~5–15 seconds per response on CPU (Colab free tier)

The Space loads the LoRA adapter directly from the Hugging Face Hub
repository. It does not use any local checkpoint.

---

## Limitations

- **Small dataset:** ~6300 training examples is insufficient for robust
  Wolof instruction following. A production system would require 50,000+
  high-quality examples.

- **English fallback:** The model sometimes responds in English to Wolof
  prompts (e.g., "Ana nga?" → "What?"), reflecting the base model's
  pre-training distribution.

- **Translation failures:** Wolof-to-French or Wolof-to-English translation
  tasks systematically fail (F1=0.00, BLEU=0.04 on number translation).
  This task type is underrepresented in all three training sources.

- **Hallucination:** The model generates plausible-sounding but factually
  incorrect answers for topics outside its training distribution
  (e.g., Yamoussoukro geography: F1=0.00).

- **Orthography inconsistency:** The Soynade dataset uses non-standard
  Wolof orthography, causing spelling inconsistencies in outputs
  (e.g., "péyum" instead of "péeyu").

- **Whisper ASR incompatibility:** Wolof is not natively supported by
  Whisper, limiting voice-based interface development on top of this model.

- **No factual grounding:** The model has no access to external knowledge
  sources and cannot verify or update factual claims.

---

## Safety and Responsible Use

**Guardrails implemented:**

A keyword-based safety filter was added to `src/context_state_machine.py`.
It detects prompts requesting medical, legal, financial, or harmful content
and returns a guardrail message instead of a model response:

> "This assistant is designed for Wolof educational purposes only.
> For medical, legal, or critical decisions, please consult a qualified
> professional."

**Refusal behavior:**

The deployed Gradio Space displays a visible limitations section informing
users that:
- This is a classroom demo model, not a production assistant
- Wolof is a low-resource language — outputs may be imperfect
- The model was fine-tuned on ~6300 examples only
- Do not use for critical or sensitive decisions

**Prompt injection risks:**

The `context_state_machine.py` retrieves training examples to augment
prompts. A confidence score mechanism was added to reject low-quality
retrievals (below similarity threshold) and prevent them from being
injected into the model context. However, this mitigation is not complete
and the model should not be used in adversarial environments.

**Recommended use:**

Always present model outputs as suggestions, not facts. Verify important
information with native Wolof speakers or authoritative sources.

---

## Authors

- **Group:** AIMS Sénégal GAAI Exam — Wolof Assistant Project
- **Members:** Mairame Niang, Fatima sané, Fatou Bintou, Christian, Jean
- **Course:** Applied Generative and Agentic AI, AIMS Sénégal
- **Instructor:** Dr. Papa-Séga WADE
- **Year:** 2026
