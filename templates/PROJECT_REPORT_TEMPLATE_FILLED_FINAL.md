# Real LLM Deployment Project Report

**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  
**Group:** Mairame Niang, Fatima sané, Fatou Bintou, Christian, Jean  
**GitHub:** https://github.com/mairamesniang12/aims-gaai-exam-wolof-assistant  
**HF Model:** https://huggingface.co/niangmariame513/wolof-assistant-qwen3  
**HF Space:** https://huggingface.co/spaces/niangmariame513/wolof-assistant-demo  

---

## 1. Problem Definition

- **Use case:** Wolof educational assistant that answers basic questions and
  assists with language learning tasks in Wolof, the most widely spoken
  language in Senegal (over 10 million speakers).

- **Users:** Senegalese learners and speakers who want a digital assistant in
  their native language; researchers and students interested in low-resource
  African language NLP.

- **Why fine-tuning is needed:** The base model (Qwen3-0.6B) has very limited
  Wolof capability because Wolof is severely underrepresented in its
  pre-training data. Without fine-tuning, the model ignores Wolof input or
  responds in English. Fine-tuning on Wolof instruction-response pairs teaches
  the model to follow instructions in Wolof and produce relevant answers.

- **Why a small model is appropriate:** Qwen3-0.6B fits in memory on a single
  T4 GPU (15 GB VRAM) with LoRA, making it trainable within the compute
  constraints of this project (Google Colab free tier). Only 1.67% of
  parameters are trainable (10,092,544 out of 606,142,464), enabling fast
  iteration with limited resources.

---

## 2. Data Preparation

Describe each source separately.

| Source | Number of examples | Task type | Cleaning/filtering | Category |
|--------|--------------------|-----------|--------------------|----------|
| CohereLabs/aya_dataset (Wolof subset) | 500 | Instruction following (general) | Filtered to Wolof language only; limited to 500 examples using --max-aya-examples 500 | General language base |
| soynade-research/Wolof-Non-Standard-Orthography | 500 | Text understanding (domain-specific) | Kept as-is; non-standard orthography documented as a limitation | Domain-specific Wolof |
| syntetic_wolof_instruct_data.jsonl (provided starter) | 5300 | Instruction following (synthetic) | No filtering applied; already in correct instruction format | Synthetic instructions |

Each source was kept in a **separate file** throughout the pipeline:
- `data/wolof_aya.jsonl` → `data/chat_aya.json`
- `data/wolof_soynade.jsonl` → `data/chat_soynade.json`
- `data/syntetic_wolof_instruct_data.jsonl` → `data/chat_synth.json`

**Chat format conversion:**

Every example was converted from the raw schema:

```json
{
  "instruction": "user task or question",
  "input": "category or context",
  "output": "expected assistant answer",
  "source": "source_name",
  "source_detail": "dataset_name_or_generation_method"
}
```

To the system/user/assistant chat format using `src/download_datasets.py`:

```json
[
  {"role": "system",    "content": "You are a helpful Wolof language educational assistant."},
  {"role": "user",      "content": "<instruction> <input>"},
  {"role": "assistant", "content": "<output>"}
]
```

The tokenization step confirmed correct conversion:
- `avg_total tokens per example: 129.0`
- `avg_supervised tokens per example: 33.9`
- `kept=5670, skipped=0`

---

## 3. Splitting Strategy

**Train/validation/eval sizes:**

| Source | Train (90%) | Validation (5%) | Eval (5%) | Total |
|--------|-------------|-----------------|-----------|-------|
| AYA | 450 | 25 | 25 | 500 |
| Soynade | 450 | 25 | 25 | 500 |
| Synthetic | 4770 | 265 | 265 | 5300 |
| **Total** | **5670** | **315** | **315** | **6300** |

**Why this split prevents data leakage:**

The split was performed **per source family** with a deterministic seed,
before merging the sources into a combined dataset. This means:

1. No example from a given source can appear in both the training set and
   the evaluation set, because splitting happens before any merging.
2. The evaluation set contains examples proportional to each source (25 from
   AYA, 25 from Soynade, 265 from Synthetic), preventing one source from
   dominating the evaluation signal.
3. The deterministic seed ensures reproducibility: running
   `python src/download_datasets.py all` again produces identical
   train/validation/eval files.

The combined eval file `data/splits/eval_all.jsonl` (315 examples) was used
for all automatic metric evaluation.

---

## 4. Training Methodology

**Base model choice:**

We selected `Qwen/Qwen3-0.6B` because:
- It fits on a T4 GPU (15 GB VRAM) with LoRA without quantization
- It supports the chat template format (system/user/assistant) natively
- It has documented multilingual capability despite its small size (0.6B params)
- It is freely available on Hugging Face Hub with no access restrictions

**LoRA configuration:**

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 10,092,544 (1.67% of total) |
| Epochs | 3 |
| Hardware | Google Colab T4 GPU |
| Training duration | ~2.5 hours |

**Assistant-only loss:**

The training script `train_lora_assistant_only.py` implements assistant-only
loss masking. The model is trained to predict only assistant tokens.
All other tokens receive label `-100`.

**Why labels are set to `-100` for system/user/padding tokens:**

In PyTorch's `CrossEntropyLoss`, a target value of `-100` is the default
`ignore_index`. When a token has label `-100`, its contribution to the loss
is zero and no gradient is computed for it. This means:

- **System tokens** (label = -100): the model does not learn to reproduce
  the system prompt
- **User tokens** (label = -100): the model does not learn to copy the
  user question
- **Padding tokens** (label = -100): padding does not pollute the gradient
- **Assistant tokens** (normal token IDs): only here the model is penalized
  for incorrect predictions

Without this masking, the model would waste capacity learning to repeat
prompts instead of learning to generate correct answers.

The tokenization statistics confirmed correct masking was applied:

```
avg_total tokens per example:       129.0
avg_supervised tokens per example:   33.9
examples kept: 5670 / 5670 (skipped: 0)
```

Only 33.9 out of 129.0 tokens per example (26%) contributed to the loss.

**Checkpoint policy:**

The training script saves two checkpoints:
- `outputs/qwen3_0_6b_wolof_lora/best_adapter/` — saved when validation
  loss reaches a new minimum during training
- `outputs/qwen3_0_6b_wolof_lora/latest_adapter/` — saved at end of training

The `best_adapter` was used for evaluation and Hugging Face Hub upload.

**Monitoring:**

Training loss was logged every 100 steps. Validation loss was computed
every ~0.14 epochs (every 100 training steps). TensorBoard logs were saved
to `outputs/tensorboard/qwen3_0_6b_wolof_lora/`.

---

## 5. Evaluation

**Automatic metrics on 315 held-out eval examples:**

```
=== Aggregate metrics ===
    exact_match: 0.3492
       token_f1: 0.7499
           bleu: 0.6665
        rouge_l: 0.7456
```

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Exact Match | 0.3492 (34.9%) | 34.9% of outputs match reference exactly |
| Token F1 | 0.7499 (75.0%) | Strong token-level overlap with reference answers |
| BLEU | 0.6665 (66.7%) | Good n-gram precision — strong for only 6300 training examples |
| ROUGE-L | 0.7456 (74.6%) | Good longest common subsequence recall |

**Training and validation loss progression:**

| Epoch | Train Loss | Eval Loss |
|-------|-----------|-----------|
| 0.14 | 5.536 → 1.886 | 1.556 |
| 0.28 | ~0.85 | 0.626 |
| 0.56 | ~0.35 | 0.261 |
| 1.00 | ~0.21 | 0.196 |
| 1.55 | ~0.18 | 0.162 |
| 2.00 | ~0.09 | 0.145 |
| 2.26 (final) | ~0.037 | 0.148 |

**Qualitative examples (real outputs from evaluation.py):**

✅ **Partial success — geography question:**
```
Q:          Alger moy péeyu ban réw?
Reference:  Alger moy péeyu réwum Algérie
Prediction: Alger moy péyum réwum Algiers
Scores:     EM=0.00, F1=0.60, BLEU=0.39, ROUGE-L=0.60
```
The model understood the question and gave a partially correct answer
(wrong spelling: "péyum" instead of "péeyu", and "Algiers" instead of
"Algérie"). Content is correct, surface form has errors.

✅ **Partial success — orthographic correction:**
```
Q:          Ndàx man ngaa jubbanti mbind mii?
            Bu ñu leen waxee: "Buleen di yàq ci suuf si"...
Reference:  Waaw! Mi ngii: Bu ñu léen wàxee...
Prediction: Waaw! Mi ngii: Jubbaanj ti mbind mi...
Scores:     EM=0.00, F1=0.38, BLEU=0.11, ROUGE-L=0.38
```

❌ **Failure — Wolof to French number translation:**
```
Q:          wagnil ma : benn, ñaar, ñett, ñent, juróom...
Reference:  un, deux, trois, quatre, cinq, six...
Prediction: Benn: Naa ngi ci, mën na nekk lu dëgg...
Scores:     EM=0.00, F1=0.00, BLEU=0.04, ROUGE-L=0.00
```
The model failed completely on translation tasks. It responded in Wolof
instead of French, ignoring the translation instruction.

❌ **Failure — geography (hallucination):**
```
Q:          Yamoussoukro mooy gëblag man réew?
Reference:  Yamoussoukro mooy gëblag réewum Côte d'Ivoire.
Prediction: Yamoussuksu kër gi "Réew?" ak "gëblàg."...
Scores:     EM=0.00, F1=0.00, BLEU=0.09, ROUGE-L=0.00
```
The model hallucinated a response that does not answer the question.

**Overall interpretation:**

BLEU 66.7% and ROUGE-L 74.6% are strong results for a model trained on
~6300 examples. The model performs well on language tasks within its training
distribution (orthographic correction, simple questions) but fails on
translation tasks and geography questions that are underrepresented in the
training data.

---

## 6. Deployment

- **Hugging Face model repo link:**
  https://huggingface.co/niangmariame513/wolof-assistant-qwen3
  Files: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files.
  Visibility: Public.

- **Hugging Face Space link:**
  https://huggingface.co/spaces/niangmariame513/wolof-assistant-demo
  Status: **Running** (green). Loads adapter from Hub (not local checkpoint).

- **Usage examples on deployed Space:**
  ```
  Input:  "Ana nga?"
  Output: "What?"
  Note:   English fallback — documented failure case

  Input:  "Alger moy péeyu ban réw?"
  Output: "Alger moy péyum réwum Algiers"
  Note:   Partial success — correct content, surface errors
  ```

- **Model card link:**
  https://huggingface.co/niangmariame513/wolof-assistant-qwen3
  Includes: model description, 3 data sources with counts, training
  configuration (LoRA rank 16, alpha 32, 1.67% trainable), real evaluation
  results (BLEU=0.6665, ROUGE-L=0.7456), loading code example, limitations.

---

## 7. Limitations and Risks

- **Hallucination:** The model generates plausible-sounding but incorrect
  answers for topics outside its training distribution (e.g., Yamoussoukro
  geography question: F1=0.00, prediction completely wrong).

- **Poor categories:** Translation tasks (Wolof→French numbers, F1=0.00)
  and some geography questions are systematically failed because these
  categories are underrepresented in the training data.

- **Data bias:** The synthetic dataset (4770 examples) dominates training
  compared to AYA (450) and Soynade (450). If synthetic examples contain
  errors or mixed-language outputs, the model inherits these biases.
  The Soynade dataset uses non-standard Wolof orthography, causing
  spelling inconsistencies in generated text.

- **Unsafe/out-of-scope prompts:** The model has not been evaluated for
  harmful outputs in a Wolof-specific cultural context. A keyword-based
  safety filter was added to `context_state_machine.py` but it is not
  a robust content moderation system.

- **Prompt injection if using retrieval or state machine context:** The
  `context_state_machine.py` retrieves context examples from training data
  to augment prompts. A malicious input could manipulate retrieved context
  and inject unintended content into the model prompt. Mitigated by the
  confidence score filter but not fully eliminated.

- **Whisper ASR incompatibility:** Wolof is not natively supported by
  Whisper, limiting the ability to build voice-based interfaces.

---

## 8. What You Improved

- **Data quality:** Each source was documented separately with its origin,
  size, and limitations. We verified 5670/5670 examples passed tokenization
  with zero skipped, and documented the Soynade non-standard orthography
  as a formal limitation.

- **`src/context_state_machine.py`:** Two improvements were added:
  (1) a confidence score mechanism that returns a fallback message when
  context retrieval quality is low, preventing poor context from being
  injected into the model prompt; (2) a keyword-based safety filter that
  detects out-of-scope requests (medical, legal, harmful) and returns a
  guardrail message.

- **Deployment app:** The starter `app.py` used `gr.Interface` with a
  placeholder `MODEL_REPO_ID`. We replaced it with a `gr.Blocks` interface
  that loads from the real Hub repository, displays the project context
  (AIMS Sénégal, instructor, 3 data sources), provides clickable example
  Wolof prompts, and shows a visible limitations/guardrail section.

- **Evaluation methodology:** We included both automatic metrics and
  qualitative examples with real scores from `evaluation.py`, explicitly
  documenting failure cases (translation: F1=0.00, hallucination: F1=0.00)
  with root-cause analysis, going beyond reporting only aggregate scores.

---

## 9. Individual Technical Notes

Each student writes half a page. Maximum one page per student.

### Student 1: Mairame Niang

- **Contribution:** Model training on Google Colab T4 GPU using
  `train_lora_assistant_only.py`; Hugging Face Hub deployment using
  `push_to_hub.py`; Gradio Space setup and deployment at
  `niangmariame513/wolof-assistant-demo` (modified `hf_space/app.py`
  from `gr.Interface` to `gr.Blocks`); GitHub repository management.

- **Technical choice and rejected alternative:** Chose Google Colab T4 GPU
  over local CPU training. Local test showed 100 examples took 41 minutes
  on CPU (Windows, `aims-sen-cv-2026` conda environment); full training
  (5670 examples, 3 epochs) would have taken several days. Colab T4
  completed training in ~2.5 hours. Also chose `gr.Blocks` over `gr.Interface`
  for a more structured Space with example prompts and guardrail section.

- **Problem or failure diagnosed:** `torchao` version conflict on Colab:
  `ImportError: Found version 0.10.0, but only versions above 0.16.0 are
  supported`. Diagnosed by reading the full traceback pointing to
  `peft/tuners/lora/torchao.py`. Fixed with `pip install torchao==0.16.0`.
  Also resolved a `401 Unauthorized` HF token error by generating a new
  Write token and re-authenticating with `hf auth login --force`.

- **Verification evidence:** Training loss decreased from 5.536 to ~0.037
  over 3 epochs. Validation loss stabilized at ~0.145. Space status shows
  **Running** (green) and responds to prompts. `python push_to_hub.py
  --check-auth` returned: `Authenticated Hugging Face user: niangmariame513`.

- **Next improvement:** Add a side-by-side comparison in the Space between
  the base Qwen3-0.6B (no adapter) and the fine-tuned model on the same
  Wolof prompt, to visually demonstrate the added value of fine-tuning
  during the final demo.

---

### Student 2: Fatima sané

- **Contribution:** Data preparation using `src/download_datasets.py`;
  downloading and separating the three Wolof data sources (AYA, Soynade,
  Synthetic); verifying chat format conversion for all three sources;
  writing `data/README.md` documenting each source, its origin, and size.

- **Technical choice and rejected alternative:** Kept three separate source
  files instead of merging into a single `wolof_all.jsonl`. Merging would
  have been simpler but would violate the exam methodology requirement for
  separated sources and make per-source traceability impossible.

- **Problem or failure diagnosed:** Running `python src/download_datasets.py
  prepare` produced: `WARNING: missing source file for aya` and `WARNING:
  missing source file for soynade`. Only the synthetic source was processed.
  Diagnosed by reading script output. Fixed by running the `all` subcommand:
  `python src/download_datasets.py all --max-aya-examples 500
  --max-soynade-examples 500 --validation-ratio 0.05 --eval-ratio 0.05`.

- **Verification evidence:** Script confirmed: `Saved chat file for aya:
  500 rows`, `Saved chat file for soynade: 500 rows`, `Saved chat file for
  synth: 5300 rows`, `Saved combined eval JSONL: 315 rows`. Training script
  later confirmed: `Tokenized examples: kept=5670, skipped=0`.

- **Next improvement:** Add a data quality filtering step to remove examples
  where the output is shorter than 5 tokens, duplicated across sources, or
  in a language other than Wolof. This would improve training signal quality,
  especially important for a low-resource language like Wolof.

---

### Student 3: Fatou Bintou

- **Contribution:** Model evaluation using `evaluation.py` on the held-out
  eval set (315 examples); collecting and interpreting real automatic metrics
  (EM=0.3492, F1=0.7499, BLEU=0.6665, ROUGE-L=0.7456); writing and
  uploading the completed model card to `niangmariame513/wolof-assistant-qwen3`.

- **Technical choice and rejected alternative:** Chose to report both BLEU
  and ROUGE-L instead of BLEU alone. BLEU measures n-gram precision but not
  recall. ROUGE-L captures the longest common subsequence, more informative
  for short Wolof outputs where word order may vary. Together they give a
  more complete picture of generation quality.

- **Problem or failure diagnosed:** Real evaluation revealed systematic
  failures on translation tasks (Wolof→French numbers: F1=0.00, BLEU=0.04)
  and geography questions outside training distribution (F1=0.00, BLEU=0.09).
  Diagnosed as data imbalance: synthetic dataset (4770 examples) dominates,
  and these task types are underrepresented in all three sources.

- **Verification evidence:** Real evaluation output from `evaluation.py`:
  `exact_match: 0.3492, token_f1: 0.7499, bleu: 0.6665, rouge_l: 0.7456`
  on 315 examples. Model card publicly visible at HF Hub with all required
  sections including these real metric values.

- **Next improvement:** Add per-source evaluation breakdown: compute metrics
  separately for AYA (25 examples), Soynade (25 examples), and Synthetic
  (265 examples) to reveal which source drives performance and which needs
  more data or cleaning.

---

### Student 4: Christian

- **Contribution:** Improvement of `src/context_state_machine.py` with
  a confidence score mechanism and a keyword-based safety filter; qualitative
  testing of the deployed Space with 10 different prompts covering greetings,
  educational questions, general Wolof phrases, and out-of-domain requests.

- **Technical choice and rejected alternative:** Chose a keyword-based
  safety filter over a classifier-based filter. A text classifier would be
  more accurate but requires an additional model, adding latency and
  complexity to the Space. A simple keyword list is sufficient for a
  classroom demo. For the confidence score, chose a length-based proxy
  over cosine similarity to avoid embedding computation overhead on CPU.

- **Problem or failure diagnosed:** Out-of-domain prompts produced
  confident but wrong answers (geography: F1=0.00 confirmed by evaluation).
  The model has no mechanism to detect when a question is outside its
  training distribution. This motivated adding the confidence score and
  safety filter to `context_state_machine.py`.

- **Verification evidence:** Tested 5 prompts designed to trigger the
  safety filter and confirmed they returned the guardrail message. Cross-
  referenced with real evaluation examples: Yamoussoukro (F1=0.00, BLEU=0.09)
  confirms out-of-domain failure; Alger (F1=0.60) confirms partial success
  on geography within training distribution.

- **Next improvement:** Replace keyword filter with embedding-based
  out-of-distribution detector using `paraphrase-multilingual-MiniLM-L12-v2`.
  If cosine similarity between user query and training data centroids < 0.4,
  display: "This question may be outside my knowledge." Directly addresses
  the F1=0.00 failures on translation and geography tasks.

---

### Student 5: Jean baptiste

- **Contribution:** Writing the project report using
  `templates/PROJECT_REPORT_TEMPLATE.md`; preparing the final demo script
  with 5 representative prompts (including both success and failure cases);
  verifying the complete submission checklist before the final presentation.

- **Technical choice and rejected alternative:** Structured the report
  following the 7-step pipeline (data → training → evaluation → deployment)
  instead of a free-form essay. This maps directly to the grading rubric.
  Chose to include the full training loss curve table (not just final metrics)
  to demonstrate that training converged and validation loss did not overfit.

- **Problem or failure diagnosed:** First draft of the report used estimated
  metric values instead of real evaluation output. Diagnosed the gap when
  the real `evaluation.py` output became available, showing BLEU=0.6665
  and ROUGE-L=0.7456 (different from initial estimates). Updated all metric
  tables and qualitative examples with real values from the actual run.

- **Verification evidence:** All 8 items from the exam README submission
  checklist verified. Real metrics confirmed in report: `exact_match=0.3492,
  token_f1=0.7499, bleu=0.6665, rouge_l=0.7456`. Demo prompts tested on
  live Space — responses confirmed within acceptable time.

- **Next improvement:** Add qualitative comparison between base Qwen3-0.6B
  (no adapter) and fine-tuned model on the same Wolof prompts in the report,
  to directly show the improvement from fine-tuning and support the training
  quality criterion (15% of grade).
