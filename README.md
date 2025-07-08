# KU-RAG: Fine-grained Knowledge Unit Retrieval-Augmented Generation for Visual Question Answering

📌 **Code Release Notice**  
The source code for **KU-RAG** will be released soon.  
If you have any questions or would like to discuss early access, please contact:  
📧 zzhang393(connect(dot)hkust-gz(dot)edu(dot)cn)

## 🔍 Overview

**Visual Question Answering (VQA)** requires models to answer natural language questions based on visual inputs. While cutting-edge multimodal large language models (MLLMs), such as GPT-4o, perform well on VQA benchmarks, they often struggle to access **domain-specific or the latest external knowledge**.

To bridge this gap, we introduce **KU-RAG**, a **Knowledge Unit Retrieval-Augmented Generation** framework designed for **KB-VQA** (knowledge base enhanced VQA). Instead of relying on unimodal retrieval that translates images into text—potentially losing critical visual cues—KU-RAG leverages **fine-grained multimodal knowledge units**, which may include textual facts, entity images, and other structured fragments.

### ✨ Key Contributions

- **Knowledge Unit Construction**: We organize multimodal fragments (text, images, etc.) into structured, retrievable units that preserve semantic richness and visual details.
- **KU-RAG Framework**: Integrates fine-grained knowledge unit retrieval with MLLMs to enhance answer accuracy and explainability.
- **Knowledge Correction Chain**: Boosts reasoning quality by refining and verifying retrieved information in context.

## 📊 Results

KU-RAG outperforms existing KB-VQA baselines across **four widely-used benchmarks**, achieving:

- 🔼 ~3% **average accuracy improvement**
- 🔼 Up to **11% improvement** in the best case

## 📄 Paper and Code

- 📝 Paper on arXiv: [https://arxiv.org/abs/2502.20964](https://arxiv.org/abs/2502.20964)  
- 💻 Code Repository (Coming Soon): [https://github.com/zzhang393/KU-RAG](https://github.com/zzhang393/KU-RAG)  
- 📬 Contact: zzhang393(connect(dot)hkust-gz(dot)edu(dot)cn)

## 🚀 Coming Soon

Stay tuned for:

- 📦 Code and raw data
- 🧪 Usage examples and evaluation scripts
- 📚 Documentation and tutorials

---

Thank you for your interest in KU-RAG!
