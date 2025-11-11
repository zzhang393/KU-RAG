import torch
import faiss
import numpy as np
from PIL import Image
import os, re
import csv
import sys

# Get the root directory path of the XRAG project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_processor = longclip.load("../longclip/longclip-L.pt", device=device)

text_data = []
kid_data = []



def split_text(text, kid, max_tokens):
    parts = []
    tokens = longclip.tokenize(text, truncation=False, return_tensors='pt')
    input_ids = tokens['input_ids'][0]
    if len(input_ids) < max_tokens:
        parts = [text]
        text_data.append(text)
        kid_data.append(kid)
        return parts
    sentences = re.split(r'(?<=[.!?])', text)
    current_part = ""
    for sentence in sentences:
        # Check if adding current sentence would exceed max token count
        tokens = longclip.tokenize(current_part + sentence, truncation=False, return_tensors='pt')
        if len(tokens['input_ids'][0]) > max_tokens:
            if current_part:
                parts.append(current_part)
                text_data.append(current_part)
                kid_data.append(kid)
            current_part = sentence
        else:
            current_part += sentence
    if current_part:
        parts.append(current_part)
        text_data.append(current_part)
        kid_data.append(kid)
    return parts



def chunk_text2(text, kid):
    chunks = []
    try:
        inputs = longclip.tokenize(text).to(device)
        chunk = inputs
        chunks.append(chunk)
        text_data.append([text])
        kid_data.append(kid)
        return chunks
    except:
        sentences = re.split(r'(?<=[.!?])', text)
        sent_group = []
        sent_temp = []
        sen_len_sum = 2
        # Pre-tokenize all sentences at once

        tokens = [longclip.tokenize(sentence, truncate=True).to(device) for sentence in sentences]


        for i, token in enumerate(tokens):
            # Calculate the length of the sentence excluding special tokens
            sen_len = int(torch.count_nonzero(token) - 2)
            sen_len_sum += sen_len

            if sen_len_sum < 246:
                sent_temp.append(sentences[i])
            else:
                sent_group.append(''.join(sent_temp))
                sent_temp = [sentences[i]]
                sen_len_sum = sen_len + 2

        if sent_temp:
            sent_group.append(''.join(sent_temp))

        for part in sent_group:
            # if kid==16775:
            #     b = 0

            if len(part) > 5:
                inputs = longclip.tokenize(part, truncate=True).to(device)
                chunks.append(inputs)
                text_data.append([part])
                kid_data.append(kid)

        return chunks


# Function to encode text
def encode_texts(texts, kids, batch_size=64):
    # max_length = 248  # Typically, LongCLIP model's maximum length is 248 tokens
    all_text_features = []
    for i in range(0, len(texts), batch_size):
        print('batch:', i)
        batch_texts = texts[i:i + batch_size]
        batch_kid = kids[i: i + batch_size]
        batch_inputs = []
        for j in range(len(batch_texts)):
            text_chunks = chunk_text2(batch_texts[j], batch_kid[j])
            batch_inputs.extend(text_chunks)
        batch_inputs = torch.cat(batch_inputs).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(batch_inputs)
        all_text_features.append(text_features.cpu().numpy())

    return np.concatenate(all_text_features, axis=0).astype(np.float32)



def encode_images(image_paths, batch_size=32):
    all_image_features = []
    for i in range(0, len(texts), batch_size):
        images = [Image.open(path) for path in image_paths]
        inputs = torch.stack([clip_processor(image) for image in images]).to(device)
        with torch.no_grad():
            images_features = clip_model.encode_image(inputs)
        all_image_features.append(images_features.cpu().numpy())
    # Concatenate all batch feature vectors
    return np.concatenate(all_image_features, axis=0).astype(np.float32)


def load_or_initialize_faiss_index(index_file, dimension):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print(f"Loaded existing FAISS index from {index_file}")
    else:
        index = faiss.IndexFlatL2(dimension)
        print(f"Initialized new FAISS index with dimension {dimension}")
    return index

# Function to append data to a CSV file
def append_to_csv(file):
    file_exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["db_index", "k_id", "content"])  # Write header
        for i in range(len(text_data)):
            writer.writerow([i, kid_data[i], text_data[i]])
        # writer.writerows(data)


import pandas as pd

# Read CSV file
file_path = 'corpus/okvqa_full_corpus.csv'
df = pd.read_csv(file_path)
texts = df['text'].tolist()
kids = df['kid'].tolist()

# Encode texts
text_vectors = encode_texts(texts, kids)

# FAISS index file path
faiss_index_file = 'ok_vqa.vec'

dimension = text_vectors.shape[1]
index = load_or_initialize_faiss_index(faiss_index_file, dimension)
index.add(text_vectors)
faiss.write_index(index, faiss_index_file)

csv_file = 'ok_vqa_index.csv'
append_to_csv(csv_file)

print(f"FAISS index has been successfully updated and saved to {faiss_index_file}")
print(f"Metadata has been successfully appended to {csv_file}")
