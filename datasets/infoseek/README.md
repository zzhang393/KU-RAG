# INFOSEEK Dataset

## About

INFOSEEK (Information Seeking) is a subset of OVEN dataset focusing on information-seeking questions that require specific knowledge about visual entities.

- **Questions**: ~5,000
- **Images**: Wikipedia (subset of OVEN)
- **Knowledge**: Wikipedia 6M

## Download

### 1. Images

INFOSEEK uses a subset of OVEN images.

```bash
# Option 1: Download full OVEN images (recommended)
cd ../oven
bash download_wiki.sh

# Option 2: Download INFOSEEK-specific subset
# The "infoseek_images" folder is a subset of "oven_images"
```

### 2. Annotations

The `infoseek_data` folder contains download scripts for INFOSEEK annotations.

```bash
# Run the bash script to download JSON lines from Google storage
# Follow official OVEN repository instructions for INFOSEEK subset
```

### 3. Knowledge Base

Shares the same Wikipedia 6M knowledge base as OVEN.

## Relationship with OVEN

INFOSEEK is a **subset** of OVEN, specifically curated for:
- Information-seeking questions
- More specific and factual queries
- Fine-grained entity knowledge

## Usage

```bash
# Build FAISS indices
python create_faiss.py

# Run retrieval and passage generation
python search_mk.py
python passage_generator.py

# Or use main.py
cd ../..
python main.py --dataset infoseek --stage full
```

## Citation

```bibtex
@article{hu2023open,
  title={Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
  author={Hu, Hexiang and Zhang, Yi and Sun, Shreyas and Lesica, Nick and Venugopalan, Subhashini},
  journal={ICCV},
  year={2023}
}
```

