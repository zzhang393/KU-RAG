# E-VQA Dataset

## About

E-VQA (Event-oriented Visual Question Answering) is a dataset for event-oriented VQA, focusing on questions about real-world events in social media images.

- **Questions**: 9,008
- **Images**: 2,690 (social media)
- **Answers**: 5,479
- **Events**: 182 real-world events
- **Knowledge**: 1,157 news media articles

## Topics

The dataset covers a wide range of event topics:
- Armed conflicts and attacks
- Disasters and accidents
- Law and crime
- Politics and elections
- Sports and entertainment
- And more...

## Download

### Access Required

This dataset requires special access. Please contact:

**Dr. Zhenguo Yang**  
Email: zhengyang5-c@my.cityu.edu.hk

### Steps

1. Download and fill up the **Dataset Agreement Form**
2. Send the signed form to the contact email
3. You will receive download links for:
   - `E-VQA_data.csv` - Questions and annotations
   - Image files
   - News article texts (knowledge base)

## Dataset Structure

After downloading, your directory should contain:
```
evqa/
├── E-VQA_data.csv           # Main dataset file
├── image/                    # Social media images
├── Textual Corpus/          # News articles (knowledge)
└── qa_data/                  # Question-answer pairs
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
python main.py --dataset evqa --stage full
```

## Citation

```bibtex
@article{yang2023event,
  title={Event-oriented Visual Question Answering: The E-VQA Dataset and Benchmark},
  author={Yang, Zhenguo and others},
  journal={ACM Multimedia},
  year={2023}
}
```

## Contact

For any questions regarding the dataset, please contact:  
Dr. Zhenguo Yang (zhengyang5-c@my.cityu.edu.hk)

