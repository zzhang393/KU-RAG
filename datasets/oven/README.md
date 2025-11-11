# OVEN Dataset

## About

OVEN (Open-domain Visual Entity recognitioN) is a dataset for open-domain visual entity recognition and knowledge-based VQA.

- **Questions**: ~11,000
- **Images**: Wikipedia
- **Knowledge**: Wikipedia 6M

## Download

### 1. Wikipedia Images and Knowledge Base

```bash
# Run the download script
bash download_wiki.sh
```

This will download:
- Wikipedia 6M knowledge base (title only / image URLs)
- Image files

### 2. OVEN Annotations

The `oven_data` folder contains download scripts for OVEN annotations.

Run the bash script to download JSON lines from Google storage:
```bash
# Follow official OVEN repository instructions
# https://github.com/open-vision-language/oven
```

### 3. OVEN Entity Data

Download `oven_entity_test.jsonl` and place it in `qa_data/` directory.

## Note

- The "oven_images" folder contains all images for OVEN and InfoSeek
- Wikipedia 6M provides both textual knowledge and image data

## Usage

```bash
# Build FAISS indices
python create_faiss.py

# Run retrieval and passage generation
python search_mk.py
python passage_generator.py

# Or use main.py
cd ../..
python main.py --dataset oven --stage full
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

