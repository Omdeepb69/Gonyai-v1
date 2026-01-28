# 🌴 Gonyai-v1: A Poetic Konkani Language Model

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/omdeep22/Gonyai-v1)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/omdeep22/Konkani_books_corpus-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Omdeepb69](https://img.shields.io/badge/GitHub-Omdeepb69-black)](https://github.com/Omdeepb69)

**Gonyai-v1** is a 160M parameter Large Language Model (LLM) built from scratch to capture the poetic essence and cultural depth of the **Konkani language (Goan dialect)**. 

Unlike models fine-tuned from generic multilingual bases, Gonyai-v1 uses a custom architecture optimized specifically for Devanagari script nuances and Konkani linguistic patterns.

---

## 🚀 Key Features

* **Custom Architecture:** Built on `KonkanGPT`, featuring **Rotary Positional Embeddings (RoPE)**, **RMSNorm**, and **SwiGLU** activation functions.
* **Pure Konkani Tokenizer:** A custom 32k Byte-Level BPE tokenizer trained exclusively on Konkani corpora for high semantic density.
* **Curated Training Data:** Trained on the [Konkani Books Corpus-v2](https://huggingface.co/datasets/omdeep22/Konkani_books_corpus-v2), comprising literature, poetry, and regional news.
* **Efficiency:** At 160M parameters, it is optimized for low-latency inference and edge deployment.

---

## 🛠️ Installation

```bash
pip install transformers torch accelerate
```

---

## 💻 Usage (Optimized Inference)

For the best poetic and coherent results, use the following configuration. This prevents the model from "drifting" into repetitive or unrelated topics.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "omdeep22/Gonyai-v1"

# 1. Load Tokenizer and Model
# trust_remote_code=True is required for the custom KonkanGPT architecture
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define your prompt using the Chat Template
messages = [
    {"role": "user", "content": "गोंयच्या पावसाचेर एक कविता बरोव."}
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)

# 3. Optimized Inference Settings
outputs = model.generate(
    tokenized_chat,
    max_new_tokens=80,          # Recommended for stability
    temperature=0.3,            # Low temp prevents hallucinations
    top_k=40,                   # Filters out low-probability noise
    top_p=0.9,                  # Nucleus sampling for coherence
    repetition_penalty=1.2,     # Essential for small models to prevent loops
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

# 4. Decode and clean response
generated_tokens = outputs[0][tokenized_chat.shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Post-processing: Clean incomplete sentences
if "।" in response:
    response = response[:response.rfind("।") + 1]

print(f"Assistant: {response}")
```

---

## 📊 Model Specifications

| Component | Specification |
| :--- | :--- |
| **Total Parameters** | 160 Million |
| **Hidden Layers** | 12 |
| **Attention Heads** | 12 |
| **Embedding Dim** | 768 |
| **Context Window** | 2048 Tokens |
| **Vocabulary Size** | 32,000 |

---

## 🗺️ DevOps & Roadmap

Gonyai-v1 is evolving. The current roadmap focuses on high-concurrency deployment and architectural scaling:

1.  **Backend Scaling:** Developing a pipeline using **AWS ECS Fargate** and **Amazon SQS** to handle up to 1M concurrent users.
2.  **Quantization:** Exporting to **GGUF/ONNX** formats for faster CPU-based inference on mobile devices.
3.  **Global Cache:** Implementing **Redis** caching to reduce redundant compute for common Konkani queries.
4.  **UI/UX:** A dedicated Streamlit-based web interface for community testing.

---

## 🤝 Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create.
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m 'Add some NewFeature'`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Credits

* **Author:** Omdeep ([GitHub: @Omdeepb69](https://github.com/Omdeepb69))
* **Model Page:** [omdeep22/Gonyai-v1](https://huggingface.co/omdeep22/Gonyai-v1)
* **Dataset:** [omdeep22/Konkani_books_corpus-v2](https://huggingface.co/datasets/omdeep22/Konkani_books_corpus-v2)

*Built with ❤️ for the Konkani community.*