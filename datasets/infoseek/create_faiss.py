import torch
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2Model

import faiss
import numpy as np
from PIL import Image
import os, sys, re
import csv
import time
import spacy
nlp = spacy.load("en_core_web_sm")

# Get the root directory path of the XRAG project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import longclip

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:5"
clip_model, clip_processor = longclip.load("../longclip/longclip-L.pt", device=device)


img_data = []
text_data = []
kid_data = []


import json



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
            if len(part) > 5:
                inputs = longclip.tokenize(part, truncate=True).to(device)
                chunks.append(inputs)
                text_data.append([part])
                kid_data.append(kid)

        return chunks


# Function to encode text
import torch
import numpy as np

def encode_texts(texts, kids, batch_size=64):
    all_text_features = []
    all_batch_inputs = []

    for i in range(0, len(texts), batch_size): #len(texts)
        print('batch:', i)

        batch_texts = texts[i:i + batch_size]
        batch_kids = kids[i: i + batch_size]
        batch_inputs = []

        for j in range(len(batch_texts)):
            text_chunks = chunk_text2(batch_texts[j], batch_kids[j])
            batch_inputs.extend(text_chunks)
        all_batch_inputs.append(torch.cat(batch_inputs))

    all_batch_inputs = torch.cat(all_batch_inputs)

    with torch.no_grad():
        for i in range(0, len(all_batch_inputs), batch_size):
            text_features = clip_model.encode_text(all_batch_inputs[i:i+batch_size])
            all_text_features.append(text_features.cpu().numpy())

    return np.concatenate(all_text_features, axis=0).astype(np.float32)



# Function to load or initialize FAISS index
def load_or_initialize_faiss_index(index_file, dimension):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print(f"Loaded existing FAISS index from {index_file}")
    else:
        index = faiss.IndexFlatL2(dimension)
        print(f"Initialized new FAISS index with dimension {dimension}")
    return index


# Append data to CSV file
def append_to_csv_img(csv_file, img_data):
    data_dict = {}
    with open('Wiki6M_ver_1_0_title_only.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            data_dict[entry['wikidata_id']] = entry['wikipedia_title']

    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image ID', 'Wikipedia Title'])  # Write header
        for img_entry in img_data:
            img_id = img_entry[0]
            wiki_title = data_dict.get(img_id, 'Not Found')  # Get corresponding wikipedia title, return 'Not Found' if not found
            writer.writerow([img_id, wiki_title])

# Example image and text data

def append_to_csv_text(file):
    file_exists = os.path.isfile(file)
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["db_index", "k_id", "content"])  # Write header
        for i in range(len(text_data)):
            writer.writerow([i, kid_data[i], text_data[i]])


import os
from PIL import Image
import torch
import numpy as np
import pandas as pd

# Assume clip_processor, clip_model, device are already defined
# Define encode_images function
def encode_images(image_paths, batch_size=32):
    all_image_features = []
    batch = 0
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(path) for path in batch_paths]
        inputs = torch.stack([clip_processor(image) for image in images]).to(device)
        with torch.no_grad():
            images_features = clip_model.encode_image(inputs)
        all_image_features.append(images_features.cpu().numpy())
        batch += 1
    # Concatenate all batch feature vectors
    return np.concatenate(all_image_features, axis=0).astype(np.float32)

# Traverse all images in folders and subfolders
def get_all_image_paths(root_folder):
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_paths.append(os.path.join(root, file))
                img_data.append([file.split('.')[0]])
    return image_paths

# Get all image paths
def constract_img_db():
    root_folder = './wikipedia_images_full'
    image_paths = get_all_image_paths(root_folder)
    image_features = encode_images(image_paths)
    faiss_index_file = 'wiki_img.vec'
    dimension = image_features.shape[1]
    index = load_or_initialize_faiss_index(faiss_index_file, dimension)
    index.add(image_features)
    faiss.write_index(index, faiss_index_file)
    csv_file = 'wiki_img_index.csv'
    append_to_csv_img(csv_file, img_data)
    print(f"FAISS index has been successfully updated and saved to {faiss_index_file}")
    print(f"Metadata has been successfully appended to {csv_file}")

def constract_text_db():
    input_file = './Wiki6M_ver_1_0.jsonl'
    csv_file = 'wiki_text_index.csv'
    json_file = 'wiki_text_clean.json'
    img_csv = 'wiki_img_index.csv'
    df = pd.read_csv(img_csv)
    ids_set = set(df['Image ID'].tolist())

    texts = []
    kids = []
    titles = []
    # Check if csv_file exists
    if os.path.exists(json_file):
        # If exists, directly read wikidata_id and wikipedia_content values
        with open(json_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        kids = result_data['wikidata_id']
        texts = result_data['wikipedia_content']
    else:
        data_list = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line.strip()))

        for data in data_list:
            current_id = data.get('wikidata_id')
            if current_id in ids_set:
                kids.append(current_id)
                texts.append(data.get('wikipedia_content'))
                titles.append(data.get('wikipedia_title'))

        result_data = {'wikidata_id': kids, 'wikipedia_title': titles, 'wikipedia_content': texts}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        result_data2 = []
        for kid, title, text in zip(kids, titles, texts):
            result_data2.append({
                'wikidata_id': kid,
                'wikipedia_title': title,
                'wikipedia_content': text
            })

        # Specify file name
        file_name = 'wiki_full.jsonl'

        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(result_data2, file, ensure_ascii=False, indent=4)


    # print('len_text:', len(texts))
    faiss_index_file = 'wiki_text.vec'
    text_vectors = encode_texts(texts, kids)
    #
    dimension = text_vectors.shape[1]
    index = load_or_initialize_faiss_index(faiss_index_file, dimension)
    index.add(text_vectors)
    faiss.write_index(index, faiss_index_file)
    append_to_csv_text(csv_file)

    # print(f"FAISS index has been successfully updated and saved to {faiss_index_file}")
    print(f"Metadata has been successfully appended to {csv_file}")



if __name__ == '__main__':
    constract_text_db()
    # constract_img_db()