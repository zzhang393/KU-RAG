# KU-RAG: Fine-grained Knowledge Unit Retrieval-Augmented Generation for Visual Question Answering

[![arXiv](https://img.shields.io/badge/arXiv-2502.20964-b31b1b.svg)](https://arxiv.org/abs/2502.20964v3)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Official Implementation of "Fine-Grained Knowledge Structuring and Retrieval for Visual Question Answering"**  
> Zhengxuan Zhang, Yin Wu, Yuyu Luo, Nan Tang  
> HKUST, HKUST(GZ)

## 🔍 Overview

**Visual Question Answering (VQA)** requires models to answer natural language questions based on visual inputs. While cutting-edge multimodal large language models (MLLMs), such as GPT-4o, perform well on VQA benchmarks, they often struggle to access **domain-specific or the latest external knowledge**.

To bridge this gap, we introduce **KU-RAG**, a **Knowledge Unit Retrieval-Augmented Generation** framework designed for **KB-VQA** (knowledge base enhanced VQA). Instead of relying on unimodal retrieval that translates images into text—potentially losing critical visual cues—KU-RAG leverages **fine-grained multimodal knowledge units**, which may include textual facts, entity images, and other structured fragments.

### ✨ Key Contributions

- **Knowledge Unit Construction**: We organize multimodal fragments (text, images, etc.) into structured, retrievable units that preserve semantic richness and visual details.
- **Meta-Knowledge Structure**: Organizes knowledge into fine-grained units with explicit entity-knowledge mappings
- **Query-Aware Instance Segmentation**: Identifies and highlights query-relevant visual regions to improve retrieval accuracy
- **KU-RAG Framework**: Integrates fine-grained knowledge unit retrieval with MLLMs to enhance answer accuracy and explainability

## 📊 Results

KU-RAG outperforms existing KB-VQA baselines across **four widely-used benchmarks** (OK-VQA, OVEN, INFOSEEK, E-VQA), achieving:

- 🔼 ~3% **average accuracy improvement**
- 🔼 Up to **11% improvement** in the best case

## 🚀 Key Features

- **Fine-Grained Knowledge Retrieval**: Retrieves specific knowledge units instead of entire passages
- **Meta-Knowledge Management**: Maintains explicit mappings between entities, images, and knowledge
- **Query-Aware Segmentation**: Uses YOLOv8 + LongCLIP to identify query-relevant image regions
- **Multi-Dataset Support**: Works with OK-VQA, OVEN, INFOSEEK, and E-VQA datasets
- **Efficient Vector Search**: FAISS-based indexing for fast retrieval

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 32GB+ RAM recommended
- 100GB+ disk space (for datasets and indices)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/zzhang393/KU-RAG.git
cd KU-RAG

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Download Model Weights

**Required Models:**
- **YOLOv8x-seg** (~130MB): Download from [Ultralytics](https://github.com/ultralytics/assets/releases)
- **LongCLIP-L** (~1.7GB): Download from [HuggingFace](https://huggingface.co/BeichengZhao/LongCLIP-L)

Place downloaded models:
```
models/yolov8x-seg.pt
longclip/longclip-L.pt
```

## 🚦 Quick Start

### Using the Main Program

```bash
# List available datasets
python main.py --list

# Run complete pipeline for OK-VQA
python main.py --dataset okvqa --stage full

# Run specific stage
python main.py --dataset okvqa --stage retrieve

# Skip already completed steps
python main.py --dataset okvqa --stage full --skip-existing
```

### Example: Query-Aware Segmentation

```python
from core.query_segmentation import segment

# Perform query-aware segmentation
query = "How many teeth does this animal have?"
image_path = "./example_image.jpg"
output_path = "./output/segmented.jpg"

success = segment(query, image_path, output_path, device='cuda')
if success:
    print("✓ Segmentation successful!")
```

### Example: Knowledge Retrieval

```python
from common import BaseRetriever, BaseMetaKnowledge

# Initialize retriever
retriever = BaseRetriever(device="cuda")

# Initialize meta-knowledge
mk = BaseMetaKnowledge(
    'mk/mk_data.json',
    'mk/mkid_mapping.json',
    'mk/img_mkid_mapping.json'
)

# Search within entity (fine-grained)
distances, indices = retriever.search_within_entity(
    query="What is this aircraft?",
    entity_name="Boeing 767",
    entity_index=temp_index,
    k=3
)
```

## 📁 Project Structure

```
KU-RAG/
├── main.py                        # Main entry point
├── core/                          # Core functionalities
│   ├── query_segmentation.py     # Query-aware segmentation
│   ├── image_processing.py       # Image processing utilities
│   └── image_merger.py            # Visual passage merging
├── common/                        # Common utilities (shared)
│   ├── base_meta_knowledge.py    # Base meta-knowledge manager
│   ├── base_retriever.py         # Base retriever class
│   └── utils.py                   # Shared utility functions
├── longclip/                      # LongCLIP model
├── datasets/                      # Dataset-specific implementations
│   ├── okvqa/                     # OK-VQA dataset
│   ├── oven/                      # OVEN dataset
│   ├── infoseek/                  # INFOSEEK dataset
│   └── evqa/                      # E-VQA dataset
├── examples/                      # Example scripts
├── scripts/                       # Utility scripts
└── requirements.txt               # Python dependencies
```

## 📊 Supported Datasets

| Dataset | Questions | Images | Knowledge Source | Task Type |
|---------|-----------|--------|------------------|-----------|
| **OK-VQA** | ~14K | COCO val2014 | Wikipedia 6M | Open-ended VQA |
| **OVEN** | ~11K | Wikipedia | Wikipedia 6M | Entity recognition + VQA |
| **INFOSEEK** | ~5K | Wikipedia | Wikipedia 6M | Information seeking |
| **E-VQA** | ~9K | Social media | News articles | Event-oriented VQA |

## 📄 Paper and Code

- 📝 Paper on arXiv: [https://arxiv.org/abs/2502.20964](https://arxiv.org/abs/2502.20964)  
- 💻 Code Repository: [https://github.com/zzhang393/KU-RAG](https://github.com/zzhang393/KU-RAG)  
- 📧 Contact: zzhang393@connect.hkust-gz.edu.cn

## 📝 Citation

If you use KU-RAG in your research, please cite:

```bibtex
@article{zhang2025kurag,
  title={Fine-Grained Knowledge Structuring and Retrieval for Visual Question Answering},
  author={Zhang, Zhengxuan and Wu, Yin and Luo, Yuyu and Tang, Nan},
  journal={arXiv preprint arXiv:2502.20964},
  year={2025}
}
```

## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share**: Copy and redistribute the material
- **Adapt**: Remix, transform, and build upon the material

Under the terms that you must give appropriate credit and indicate if changes were made.

See [LICENSE](LICENSE) for full details.

## 🙏 Acknowledgments

This project builds upon several excellent open-source projects:

- [LongCLIP](https://github.com/beichenzbc/Long-CLIP): Long-text vision-language model
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): Object detection and segmentation
- [FAISS](https://github.com/facebookresearch/faiss): Efficient similarity search
- [OK-VQA](https://okvqa.allenai.org/): Knowledge-based VQA dataset
- [OVEN](https://open-vision-language.github.io/oven/): Open-domain visual entity dataset
- [E-VQA](https://github.com/HITsz-TMG/E-VQA): Event-oriented VQA dataset

---

**Made with ❤️ by the KU-RAG Team**
