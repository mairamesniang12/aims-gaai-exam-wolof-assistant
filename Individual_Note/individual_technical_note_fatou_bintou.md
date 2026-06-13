# Individual Technical Note
**Student:** Fatou Bintou  
**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  

---

## 1. What exact part did I implement?

I was responsible for **model evaluation and the model card**.

For evaluation, I ran `evaluation.py` on the held-out eval set
(`data/splits/eval_all.jsonl`, 315 examples) using the trained LoRA adapter:

```bash
python evaluation.py \
    --data data/splits/eval_all.jsonl \
    --generate \
    --model-choice qwen \
    --adapter auto
```

I collected and interpreted the automatic metrics: Exact Match, Token F1,
BLEU, ROUGE-L, and perplexity. I also collected qualitative examples, noting
cases where the model succeeded (short Wolof responses) and cases where it
failed (responding in English to Wolof prompts).

For the model card, I filled in the `templates/MODEL_CARD_TEMPLATE.md` with
the real training configuration, data sources, evaluation results, and
limitations. The completed model card was uploaded to the Hugging Face model
repository at `niangmariame513/wolof-assistant-qwen3`.

---

## 2. Which technical choice did I make, and what alternative did I reject?

**Choice: Reporting BLEU and ROUGE-L together over BLEU alone.**  
BLEU alone measures n-gram precision but does not capture recall. For a
generative model on short Wolof outputs, ROUGE-L (which measures the longest
common subsequence) is more informative because it captures whether the model
produces the key content of the reference answer, even if the word order
differs slightly. I included both metrics to give a more complete picture.

**Choice: Including qualitative failure cases in the model card.**  
An alternative would have been to only report the numerical metrics. I chose
to include at least one failure case ("Ana nga?" → "What?") because the exam
rubric explicitly requires documenting limitations and failure cases. This also
makes the model card more honest and useful for anyone who wants to reuse the
adapter.

---

## 3. What problem or failure did I observe, and how did I diagnose it?

**Problem — Model generates English responses to Wolof inputs.**  
During qualitative evaluation on the deployed Gradio Space, I tested the
prompt "Ana nga?" (meaning "How are you?" in Wolof). The model responded
"What?" instead of the expected Wolof response "Mangi fi rekk" (I am fine).

I diagnosed this as a data imbalance issue: the synthetic dataset (4770
examples) dominates the training data compared to AYA (450) and Soynade (450).
If many synthetic examples have English outputs, the model learns to default
to English when uncertain. This is a known limitation of training on small,
potentially noisy synthetic data.

---

## 4. How did I verify that my contribution works?

I verified the evaluation by checking that `evaluation.py` ran to completion
on all 315 eval examples without errors and produced metric scores for each
source (aya, soynade, synth) as well as a combined score.

I verified the model card by confirming it is publicly visible at:
`https://huggingface.co/niangmariame513/wolof-assistant-qwen3`

The model card includes all required sections: model description, training
data (3 sources with counts), training configuration (LoRA rank 16, alpha 32,
trainable params 1.67%), evaluation results, usage example, and limitations.

---

## 5. If I had one extra day, what would I improve first and why?

I would improve the **evaluation methodology** by adding a per-source
breakdown of metrics. Currently, the eval set mixes examples from AYA,
Soynade, and Synthetic sources. A per-source evaluation would reveal whether
the model performs better on synthetic examples (which dominate training)
than on real AYA or Soynade examples. This diagnostic information would
directly guide the next data collection and training iteration.
