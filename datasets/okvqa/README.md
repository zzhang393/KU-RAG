# OK-VQA Dataset

## About

OK-VQA (Outside Knowledge Visual Question Answering) is a dataset that requires external knowledge beyond the image content to answer questions.

- **Questions**: ~14,000
- **Images**: MS COCO val2014
- **Knowledge**: Wikipedia

## Download

### 1. Images (MS COCO)

```bash
# Download COCO val2014 images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d ./val2014/
```

### 2. Annotations

Visit [OK-VQA Official Website](https://okvqa.allenai.org/) to download:
- `OpenEnded_mscoco_val2014_questions.json`
- `mscoco_train2014_annotations.json`

Place them in the `qa_data/` directory.

### 3. Knowledge Base

Download Wikipedia 6M knowledge base:
```bash
# Follow OVEN dataset instructions or contact the authors
```

## Usage

```bash
# Build FAISS indices
python create_faiss.py

# Run retrieval and passage generation
python search_mk.py
python passage_generator.py

# Or use main.py
cd ../..
python main.py --dataset okvqa --stage full
```

## Citation

```bibtex
@inproceedings{marino2019ok,
  title={OK-VQA: A visual question answering benchmark requiring external knowledge},
  author={Marino, Kenneth and Rastegari, Mohammad and Farhadi, Ali and Mottaghi, Roozbeh},
  booktitle={CVPR},
  year={2019}
}
```

