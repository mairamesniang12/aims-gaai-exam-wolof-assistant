# Individual Technical Note
**Student:** Jean  
**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  

---

## 1. What exact part did I implement?

I was responsible for **the project report and the final demo preparation**.

For the project report, I used `templates/PROJECT_REPORT_TEMPLATE.md` as a
base and filled in all required sections:

- **Problem definition**: Wolof low-resource language assistant use case,
  motivation, and target users.
- **Data methodology**: description of the three sources (AYA, Soynade,
  Synthetic), cleaning steps, chat formatting, and split ratios.
- **Training**: LoRA configuration, assistant-only -100 masking explanation,
  training curves (loss from 5.5 → 0.04 over 3 epochs).
- **Evaluation**: automatic metrics (Exact Match, Token F1, BLEU, ROUGE-L)
  and qualitative examples including failure cases.
- **Deployment**: Hugging Face Hub model repository and Gradio Space links.
- **Limitations**: Wolof low-resource challenges, small dataset size,
  English fallback behavior, Whisper ASR incompatibility.

For the demo, I prepared a structured walkthrough of the deployed Space,
selecting 5 representative prompts that show both the model's capabilities
and its documented limitations.

---

## 2. Which technical choice did I make, and what alternative did I reject?

**Choice: Structuring the report around the 7 pipeline steps.**  
An alternative would have been to write the report as a free-form essay.
I chose to follow the 7-step structure (data → training → evaluation →
deployment) because it directly maps to the grading rubric and makes it
easy for the instructor to verify each deliverable.

**Choice: Including training loss curves in the report.**  
An alternative would have been to only report final metric numbers. I
included the full training and validation loss curves (showing loss
decreasing from 5.5 to 0.04 over 3 epochs) because they demonstrate
that the model actually learned and did not overfit, which is a key
requirement for the training quality criterion (15% of the grade).

---

## 3. What problem or failure did I observe, and how did I diagnose it?

**Problem — The report initially lacked a clear limitations section.**  
In the first draft, the limitations section only mentioned dataset size.
During review, I noticed the exam rubric explicitly requires documenting
failure cases and deployment limitations. I diagnosed the gap by comparing
the draft against the rubric checklist.

I expanded the limitations section to include:
- Wolof not being natively supported by Whisper ASR (relevant for future
  voice interfaces).
- The model defaulting to English for some Wolof prompts ("Ana nga?" → "What?").
- The Soynade dataset using non-standard Wolof orthography, which may
  cause inconsistencies in the model's spelling.

---

## 4. How did I verify that my contribution works?

I verified the report by going through the exam submission checklist
from the README one item at a time:

```
✅ configs/wolof_training_config.yml exists
✅ data/splits/eval_all.jsonl exists
✅ Hugging Face model repo is public
✅ Hugging Face Space runs and uses the Hub model
✅ Model card is complete
✅ Report explains data, training, evaluation, deployment, limitations
✅ Each student has written an individual technical note
```

All items were verified before submission.

I verified the demo by running through all 5 prepared prompts on the live
Space at `niangmariame513/wolof-assistant-demo` and confirming that the
model responds and that the guardrail limitation message is visible.

---

## 5. If I had one extra day, what would I improve first and why?

I would improve the **demo script** to include a side-by-side comparison
between the base Qwen3-0.6B model (without the LoRA adapter) and the
fine-tuned model on the same Wolof prompts. This would visually demonstrate
the added value of fine-tuning and make the demo more convincing for the
final presentation. It would also directly support the training quality
criterion by showing qualitative improvement from the adapter.
