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

## 📊 Benchmarks (Feb 2026)

Gonyai-v1 was tested against sub-1B global models to evaluate its efficiency in handling the Konkani language.
<img width="1210" height="771" alt="image" src="https://github.com/user-attachments/assets/f1bc5a85-e9e9-4461-9e17-9dce173d39e7" />

| Model | Parameters | Token Efficiency (Lower = Native) | Speed (Tokens/Sec) |
| :--- | :--- | :--- | :--- |
| **Gonyai-v1** | **160M** | **5.00** | **65.96** |
| Qwen2.5-0.5B | 500M | 6.57 | 33.27 |
| SmolLM2-360M | 360M | 7.85 | 27.00 |

### 🔍 Analysis:
- **Efficiency:** Gonyai-v1 is **~35% more efficient** at representing Konkani text than Qwen2.5 due to its native tokenizer.
- **Latency:** It delivers **2x higher throughput** than larger models, making it ideal for edge deployment.
- **Limitations:** As a 160M base model, it focuses on linguistic fluency over world knowledge. It may hallucinate facts or struggle with complex logic.

---

## 💻 Usage (Optimized Inference)

For the best poetic and coherent results, use the following configuration. This prevents the model from "drifting" into repetitive or unrelated topics.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "omdeep22/Gonyai-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# 2. Prepare Prompt (Base Model Format)
prompt = "गोंयच्या पावसाचेर एक कविता बरोव."
full_text = f"<|user|>\n{prompt}\n<|assistant|>\n"

# add_token_type_ids=False is critical for this custom architecture
inputs = tokenizer(full_text, return_tensors="pt", add_token_type_ids=False).to(device)

# 3. Optimized Generation
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.4,           # Balanced for creativity
        repetition_penalty=1.2,    # Prevents loops in small models
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 4. Decode
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
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
