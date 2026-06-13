---
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- wolof
- senegal
- education
---

# Wolof LoRA Adapter

This repository contains a LoRA adapter fine-tuned for a classroom demo on
Wolof instruction following in Senegalese/African contexts.

## Base Model

- `Qwen/Qwen3-0.6B`

## Intended Use

This adapter is designed for teaching:

- instruction fine-tuning,
- LoRA deployment,
- local inference,
- basic evaluation with Exact Match, F1, BLEU, ROUGE-L, and perplexity.

It is not a production Wolof assistant.

## Loading Example

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen3-0.6B"
adapter_id = "YOUR_USERNAME/YOUR_REPO"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_id)
```

## Training Setup

- LoRA rank: 16
- LoRA alpha: 32
- Target modules: attention and MLP projection layers
- Dataset schema: `instruction`, `input`, `output`
- Chat template rendered without hidden thinking traces when supported.

## Limitations

The dataset is small and classroom-oriented. The model may repeat short Wolof
phrases or fail outside the covered categories. Evaluate before reuse.
