# Individual Technical Note
**Student:** Fatima  
**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  

---

## 1. What exact part did I implement?

I was responsible for **data preparation and documentation**.

Specifically, I worked on understanding and verifying the three data sources
used in the project:

- `data/wolof_aya.jsonl` — CohereLabs AYA dataset, Wolof subset (general language)
- `data/wolof_soynade.jsonl` — Soynade Wolof Non-Standard Orthography (domain-specific)
- `data/syntetic_wolof_instruct_data.jsonl` — Synthetic Wolof instructions (provided starter)

I reviewed the data schema for each source, verifying that every row contained
the required fields: `instruction`, `input`, `output`, `source`, and
`source_detail`. I also wrote the `data/README.md` documenting each source,
its origin, its size, and how it was used in training.

I verified the chat-formatted output files (`data/chat_aya.json`,
`data/chat_soynade.json`, `data/chat_synth.json`) to confirm that every example
was correctly converted to the `system / user / assistant` format required by
the training script.

---

## 2. Which technical choice did I make, and what alternative did I reject?

**Choice: Three separated source files over one merged dataset.**  
An alternative would have been to merge all data into a single
`wolof_all.jsonl` file before training. I rejected this because the exam
methodology explicitly requires separated sources for traceability and
evaluation. Keeping sources separate also allows per-source evaluation to
detect which source contributes most to model quality.

**Choice: Keeping the 90/5/5 split ratio.**  
I considered using an 80/10/10 split to have more validation data, but the
dataset is already small (~6300 examples total). Using 90% for training
maximizes the learning signal, which is more important given the limited
size of the Wolof data available.

---

## 3. What problem or failure did I observe, and how did I diagnose it?

**Problem — Missing source files at first run.**  
When running `python src/download_datasets.py prepare` for the first time,
the script produced two warnings:
```
WARNING: missing source file for aya
WARNING: missing source file for soynade
```
This meant only the synthetic source was being used, which would have failed
the data methodology requirement (3 separated sources). I diagnosed this by
reading the script output and the README, which indicated that the `all`
subcommand was needed to download the AYA and Soynade datasets from
Hugging Face before preparing. Running:
```
python src/download_datasets.py all --max-aya-examples 500 --max-soynade-examples 500
```
resolved the issue and produced all three source files correctly.

---

## 4. How did I verify that my contribution works?

I verified the data preparation by checking the output of the preparation
script, which confirmed:

```
Saved chat file for aya: 500 rows
Saved chat file for soynade: 500 rows
Saved chat file for synth: 5300 rows
Saved combined eval JSONL: 315 rows
```

I also opened `data/chat_aya.json` and manually inspected 5 examples to
confirm the system/user/assistant format was correctly applied. The training
script later confirmed: `Tokenized examples: kept=5670, skipped=0`,
meaning all examples passed the tokenization check with no data loss.

---

## 5. If I had one extra day, what would I improve first and why?

I would improve the **data quality filtering** before training. Currently, all
examples from the three sources are used without any quality check. With one
extra day, I would add a filtering step to remove examples where the output
is too short (less than 5 tokens), in a language other than Wolof, or
duplicated across sources. This would improve training signal quality and
reduce noise, which is especially important for a low-resource language like
Wolof where every training example counts.
