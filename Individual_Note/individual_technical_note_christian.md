# Individual Technical Note
**Student:** Christian  
**Project:** Wolof Educational Assistant — LoRA Fine-tuned Qwen3-0.6B  
**Course:** Applied Generative and Agentic AI — AIMS Sénégal  
**Instructor:** Dr. Papa-Séga WADE  

---

## 1. What exact part did I implement?

I was responsible for **improving `src/context_state_machine.py`** and
**qualitative testing of the deployed Space**.

For the context state machine, I analyzed the existing implementation and
added two improvements:

1. **A confidence score mechanism**: the state machine now estimates a
   confidence level based on the length and coherence of the retrieved
   context. If confidence is below a threshold, it returns a fallback
   message instead of passing low-quality context to the model.

2. **A simple safety filter**: I added a keyword-based check that detects
   prompts asking for harmful, medical, or legal advice and returns a
   guardrail message explaining the model's limitations.

For qualitative testing, I tested the deployed Gradio Space at
`niangmariame513/wolof-assistant-demo` with 10 different prompts covering
different categories: greetings, educational questions, general Wolof
phrases, and out-of-domain requests. I documented which categories worked
well and which failed.

---

## 2. Which technical choice did I make, and what alternative did I reject?

**Choice: Keyword-based safety filter over a classifier-based filter.**  
A more sophisticated alternative would have been to use a small text
classifier to detect unsafe prompts. I rejected this because it would
require an additional model, adding latency and complexity to the Space.
A simple keyword list is sufficient for a classroom demo and can be
expanded later without retraining.

**Choice: Confidence score based on context length over cosine similarity.**  
An alternative confidence measure would have been cosine similarity between
the user query and retrieved context embeddings. I chose a simpler
length-based proxy because the Space runs on CPU and computing embeddings
at inference time would significantly slow down the response.

---

## 3. What problem or failure did I observe, and how did I diagnose it?

**Problem — Out-of-domain prompts produce confident but wrong answers.**  
During qualitative testing, I sent the prompt "What is the capital of France?"
to the Wolof assistant. The model responded with a plausible-sounding but
irrelevant answer. This is a hallucination failure caused by the model having
no mechanism to detect when a question is outside its training distribution.

I diagnosed this by comparing the training data categories with the test
prompt. The training data covers Wolof language and educational content, but
contains no geography questions. The model has no way to know it should
refuse or express uncertainty.

This failure case motivated adding the confidence score and safety filter
to `context_state_machine.py`, so the system can at least warn users when
it may not have reliable information.

---

## 4. How did I verify that my contribution works?

I verified the safety filter by testing 5 prompts that should trigger it
(medical advice, harmful content requests) and confirming they returned
the guardrail message instead of a model response.

I verified the qualitative testing by documenting a table of 10 test
prompts with their expected outputs, actual outputs, and a pass/fail
judgment. This table was included in the project report as the qualitative
evaluation section required by the exam rubric.

---

## 5. If I had one extra day, what would I improve first and why?

I would replace the keyword-based safety filter with a **small embedding-based
out-of-distribution detector**. Using a sentence-transformer model
(e.g. `paraphrase-multilingual-MiniLM-L12-v2`), I would compute the
cosine similarity between each user query and the training data centroids
per category. If the similarity is below 0.4, the system would display:
"This question may be outside my knowledge. Please consult a specialist."
This would directly address the hallucination failure I observed during
testing, without requiring a full model retrain.
