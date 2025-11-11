import torch
import faiss
import numpy as np
from PIL import Image
import os
import csv
import json
import spacy
import re
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sys
from meta_knowledge_manager import MetaKnowledge, get_mk
import shutil
import torch.nn as nn
from query_segment import segment


# Get the root directory path of the XRAG project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import longclip

device = "cuda:5"
clip_model, clip_processor = longclip.load("../longclip/longclip-L.pt", device=device)
nlp = spacy.load("en_core_web_sm")

# Custom Dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        return clip_processor(image).to(device), image_path


def encode_text(text):
    inputs = longclip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(inputs).to(device)
    return text_features.cpu().numpy().astype('float32')


def split_index(cpu_index, num_splits):
    indices = []
    size = cpu_index.ntotal // num_splits
    for i in range(num_splits):
        start = i * size
        end = (i + 1) * size if i < num_splits - 1 else cpu_index.ntotal
        sub_index = faiss.IndexFlatL2(cpu_index.d)
        sub_index.add(cpu_index.reconstruct_n(start, end - start))
        indices.append(sub_index)
    return indices

# Define function to extract keywords
def extract_keywords(sentence):
    # Process sentence using spaCy
    doc = nlp(sentence)
    # Define unwanted parts of speech and question words
    question_words = {'what', 'many', 'which', 'why', 'who', 'whom', 'whose', 'when', 'where', 'how', 'this', 'it', 'that', 'you'}
    pronouns = {'PRP', 'PRP$', 'WP', 'WP$'}

    keywords = []
    pattern = r'\b(?:' + '|'.join(question_words) + r')\b'

    for chunk in doc.noun_chunks:
        if not chunk.root.lower_ in question_words:
            chunk_text = re.sub(pattern, '', chunk.text, flags=re.IGNORECASE)
            keywords.append(chunk_text)
    for token in doc:
        if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} and not token.is_stop and token.dep_ != 'amod':
            if not token.lower_ in question_words and token.tag_ not in pronouns:
                if not element_in_list(token.text, keywords):
                    keywords.append(token.text)

    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    keywords = ', '.join(keywords)
    return keywords


# Function to encode images
def encode_images(image_paths):
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_features = []
    all_image_paths = []

    with torch.no_grad():
        for inputs, paths in dataloader:
            image_features = clip_model.encode_image(inputs)
            all_features.append(image_features.cpu().numpy())
            all_image_paths.extend(paths)

    all_features = np.vstack(all_features).astype('float32')
    return all_features, all_image_paths


def encode_images_single(image):
    image = Image.open(image)
    img_processed = clip_processor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_feature = clip_model.encode_image(img_processed).cpu().numpy()

    return image_feature.astype('float32')



def element_in_list(element, lst):
    return any(element.lower() in item.lower() for item in lst)


def load_faiss_index(index_file):
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file {index_file} not found.")
    index = faiss.read_index(index_file)
    return index


# Function to load metadata from CSV file
def load_metadata(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Metadata file {csv_file} not found.")
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        metadata = list(reader)
    return metadata


# Function to perform a search
def search_faiss_index(index, query_vectors, k=3):
    distances, indices = index.search(query_vectors, k)
    return distances, indices


def get_all_images(root_folder):
    img_file = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                img_file.append(file.split('.')[0])
    return img_file


def load_faiss_index_on_gpus(index_path, gpu_ids):
    # Load the index from file
    cpu_index = faiss.read_index(index_path)

    # Split the index into smaller indices
    num_gpus = len(gpu_ids)
    split_indices = split_index(cpu_index, num_gpus)

    # Create a list of GPU resources
    gpu_resources = []
    for gpu_id in gpu_ids:
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)

    # Create a FAISS IndexShards object
    gpu_index = faiss.IndexShards(cpu_index.d)

    for i, gpu_id in enumerate(gpu_ids):
        gpu_index_i = faiss.index_cpu_to_gpu(gpu_resources[i], gpu_id, split_indices[i])
        gpu_index.add_shard(gpu_index_i)
    return gpu_index


def search_img(img_index, questions_list, seg_list=None, batch_size=16):
    def process_batch(image_paths):
        query_vectors, processed_image_paths = encode_images(image_paths)
        distances, indices = search_faiss_index(img_index, query_vectors)
        return distances, indices

    image_paths_list = []
    query_list = []
    if seg_list == None:
        seg_list = []

    for question in questions_list:
        if question['image_id'] in seg_list:
            image_paths_list.append(f"./image_seg/{question['image_id']}.jpg")
        else:
            image_paths_list.append(f"./image/{question['image_id']}.jpg")
        query_list.append(question['question'])

    all_distances = []
    all_indices = []

    for i in range(0, len(image_paths_list), batch_size):
        batch_image_paths = image_paths_list[i:i + batch_size]
        distances, indices = process_batch(batch_image_paths)
        all_distances.extend(distances)
        all_indices.extend(indices)

    return all_distances, all_indices


def search_img_single(img):
    faiss_index_file = 'wiki_img.vec'
    csv_file = 'wiki_img_index.csv'
    index = load_faiss_index(faiss_index_file)
    metadata = load_metadata(csv_file)

    query_vectors = encode_images_single(img)

    distances, indices = search_faiss_index(index, query_vectors)

    for i in range(len(distances[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        content = metadata[idx]
        result = f"Text: {content}"
        print(f"Rank {i + 1}: Index {idx}, Distance {dist}, {result}")



def search_text2(query_input, mk_name, gpu_index):
    keyword = extract_keywords(query_input)
    text_input = query_input + '[SEP] ' + mk_name + ' [SEP]' + keyword
    query_vector = encode_text(text_input)
    query_vector = torch.tensor(query_vector).numpy()
    distances, indices = gpu_index.search(query_vector, k=3)

    return distances, indices


def search_text(questions_list):
    faiss_index_file = 'wiki_text.vec'
    csv_file = 'wiki_text_index.csv'
    parquet_file = 'wiki_text_index.parquet'

    gpu_ids = [4, 5, 6, 7]  # Assume 4 GPUs are available
    gpu_index = load_faiss_index_on_gpus(faiss_index_file, gpu_ids)

    if os.path.exists(parquet_file):
        metadata = pd.read_parquet(parquet_file).values.tolist()
    else:
        df = pd.read_csv(csv_file)
        df.to_parquet(parquet_file)
        metadata = pd.read_parquet(parquet_file).values.tolist()

    for question in questions_list:
        text_input = question['question']
        keyword = extract_keywords(text_input)
        text_input2 = text_input + '[SEP]' + keyword
        query_vector = encode_text(text_input2)
        query_vector = torch.tensor(query_vector).numpy()
        distances, indices = gpu_index.search(query_vector, k=5)
        print(question['entity_id'], ":", question['entity_text'], "     ", question['data_split'])
        print(question['data_id'], ": ", question['question'])
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            content = metadata[idx]
            result = f"Text: {content}"
            print(f"Rank {rank + 1}: Index {idx}, Distance {dist}, {result}")



def extract_vectors_from_index(index, index_list):
    vectors = [index.reconstruct(idx) for idx in index_list]
    return vectors

def build_temporary_faiss_index(vectors):
    dimension = len(vectors[0])  # Get the dimension of vectors
    temp_index = faiss.IndexFlatL2(dimension)  # Create a flat index with L2 distance
    temp_index.add(vectors)
    return temp_index


if __name__ == "__main__":

    # Load questions
    print('Questions loading...')
    questions_list = []
    file_path = './qa_data/oven_entity_test.jsonl'
    # img_set = set()
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")
    for question_data in data:
        question_info = {
            'data_id': question_data['data_id'],
            'image_id': question_data['image_id'],
            'question': question_data['question'],
            'entity_id': question_data['entity_id'],
            'entity_text': question_data['entity_text'],
            'data_split': question_data['data_split']
        }
        # img_set.add(question_data['image_id'])
        questions_list.append(question_info)


    # Query-aware image segmentation
    print('Image segmenting...')
    seg_result = './results/seg_result.json'
    seg_list = []
    if not os.path.exists(seg_result):
        for question in questions_list:
            query = question['question']
            image = f"./image/{question['image_id']}.jpg"
            need_seg = segment(query, image)
            if need_seg:
                seg_list.append(question['image_id'])
        with open(seg_result, 'w') as file:
            json.dump(seg_list, file, indent=2)
    else:
        with open(seg_result, 'r') as file:
            seg_list = json.load(file)


    # Search for corresponding images
    print('Image searching...')
    img_results_file = './results/img_result.json'
    img_results = []
    if not os.path.exists(img_results_file):
        faiss_index_file = 'wiki_img.vec'
        csv_file = 'wiki_img_index.csv'
        index = load_faiss_index(faiss_index_file)
        metadata = load_metadata(csv_file)
        all_distances, all_indices = search_img(index, questions_list, seg_list=seg_list)

        for i in range(len(all_indices)):
            distances, indices = all_distances[i], all_indices[i]
            temp_list = []
            for rank, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < 150:
                    uid = metadata[idx][0]
                    temp_list.append(uid)
            img_results.append(temp_list)
        with open(img_results_file, 'w') as file:
            json.dump(img_results, file, indent=2)

        del index
        del csv_file
    else:
        with open(img_results_file, 'r') as file:
            img_results = json.load(file)


    # Match images to corresponding meta-knowledge
    print('Meta knowledge searching...')
    mk_result_file = './results/mk_result.json'
    mk_results = []
    if not os.path.exists(mk_result_file):
        mk = MetaKnowledge('./mk/mk_data.json', './mk/mkid_mapping.json', './mk/img_mkid_mapping.json')
        for result in img_results:
            temp_list = []
            if len(result) > 0:
                for sub_result in result:
                    try:
                        mk_result = get_mk(mk, image=sub_result)[0]
                    except:
                        pass
                    if len(mk_result) != 0:
                        temp_list.append(mk_result)
            mk_results.append(temp_list)
        with open(mk_result_file, 'w') as file:
            json.dump(mk_results, file, indent=2)
    else:
        with open(mk_result_file, 'r') as file:
            mk_results = json.load(file)

    # Search based on query
    print('Index searching...')
    idx_results = []
    idx_result_file = './results/idx_result.json'
    if not os.path.exists(idx_result_file):
        faiss_index_file = 'wiki_text.vec'
        csv_file = 'wiki_text_index.csv'
        parquet_file = 'wiki_text_index.parquet'
        original_index = faiss.read_index(faiss_index_file)
        metadata = pd.read_csv(csv_file).values.tolist()

        if os.path.exists(parquet_file):
                metadata = pd.read_parquet(parquet_file).values.tolist()
        else:
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file)
            metadata = pd.read_parquet(parquet_file).values.tolist()

        for i, mk_result in enumerate(mk_results):
            query = questions_list[i]['question']
            question_id = questions_list[i]['data_id']
            idx_result = []
            cnt_idx = 0
            idx_record = {}

            for j, mk_sub_result in enumerate(mk_result):
                mk_img = mk_sub_result['image']
                mk_name = mk_sub_result['name'][0]
                mk_idx = mk_sub_result['knowledge_index']

                # vectors = extract_vectors_from_index(original_index, mk_idx)
                vectors = np.array([original_index.reconstruct(idx) for idx in mk_idx])
                for k, idx in enumerate(mk_idx):
                    idx_record[cnt_idx] = int(idx)
                    cnt_idx += 1
                temp_index = faiss.IndexFlatL2(vectors.shape[-1])
                temp_index.add(vectors)

                dists, idxs = search_text2(query, mk_name, temp_index)
                for dist, idx in zip(dists, idxs):
                    for sub_dist, sub_idx in zip(dist, idx):
                        if sub_idx.astype(int) != -1:
                            idx_result.append(
                                {'question_id': str(question_id), 'name': str(mk_name), 'image': str(img_results[i][j]),
                                 'dist': round(sub_dist.astype(float), 2), 'idx': idx_record[sub_idx.astype(int)]})
            sorted_data = sorted(idx_result, key=lambda x: x['dist'], reverse=False)
            idx_results.append(sorted_data[:3])
        with open(idx_result_file, 'w') as file:
            json.dump(idx_results, file, indent=2)

    else:
        with open(idx_result_file, 'r') as file:
            idx_results = json.load(file)



