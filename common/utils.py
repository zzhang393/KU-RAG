"""
Common Utilities

Shared utility functions for FAISS indexing, text processing, etc.
"""

import os
import csv
import re
from typing import List, Tuple, Any
import faiss
import spacy


# Load spaCy model (lazy loading)
_nlp = None

def get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_faiss_index(index_file: str) -> faiss.Index:
    """
    Load FAISS index from file.
    
    Args:
        index_file: Path to FAISS index file (.vec)
        
    Returns:
        FAISS index object
        
    Raises:
        FileNotFoundError: If index file doesn't exist
    """
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file {index_file} not found.")
    index = faiss.read_index(index_file)
    return index


def load_metadata(csv_file: str) -> List[List[str]]:
    """
    Load metadata from CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        List of metadata rows
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Metadata file {csv_file} not found.")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        metadata = list(reader)
    return metadata


def search_faiss_index(
    index: faiss.Index, 
    query_vectors: Any, 
    k: int = 10
) -> Tuple[Any, Any]:
    """
    Search in FAISS index.
    
    Args:
        index: FAISS index object
        query_vectors: Query vectors (numpy array)
        k: Number of nearest neighbors to return
        
    Returns:
        Tuple of (distances, indices)
    """
    distances, indices = index.search(query_vectors, k)
    return distances, indices


def extract_keywords(sentence: str) -> str:
    """
    Extract keywords from a sentence using spaCy NLP.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Comma-separated keywords string
    """
    nlp = get_nlp()
    doc = nlp(sentence)
    
    # Define words to filter out
    question_words = {
        'what', 'many', 'which', 'why', 'who', 'whom', 'whose', 
        'when', 'where', 'how', 'this', 'it', 'that', 'you'
    }
    pronouns = {'PRP', 'PRP$', 'WP', 'WP$'}
    
    keywords = []
    
    # Build regex pattern for question words
    pattern = r'\b(?:' + '|'.join(question_words) + r')\b'
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        if not chunk.root.lower_ in question_words:
            chunk_text = re.sub(pattern, '', chunk.text, flags=re.IGNORECASE)
            keywords.append(chunk_text)
    
    # Extract individual keywords
    for token in doc:
        if (token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} and 
            not token.is_stop and 
            token.dep_ != 'amod'):
            if (not token.lower_ in question_words and 
                token.tag_ not in pronouns):
                if not element_in_list(token.text, keywords):
                    keywords.append(token.text)
    
    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    keywords_str = ', '.join(keywords)
    
    return keywords_str


def element_in_list(element: str, lst: List[str]) -> bool:
    """
    Check if element exists in list (case-insensitive substring match).
    
    Args:
        element: Element to search for
        lst: List to search in
        
    Returns:
        True if element found, False otherwise
    """
    return any(element.lower() in item.lower() for item in lst)


def split_faiss_index(cpu_index: faiss.Index, num_splits: int) -> List[faiss.Index]:
    """
    Split a FAISS index into multiple smaller indices for multi-GPU usage.
    
    Args:
        cpu_index: Original FAISS index
        num_splits: Number of splits
        
    Returns:
        List of split indices
    """
    indices = []
    size = cpu_index.ntotal // num_splits
    
    for i in range(num_splits):
        start = i * size
        end = (i + 1) * size if i < num_splits - 1 else cpu_index.ntotal
        sub_index = faiss.IndexFlatL2(cpu_index.d)
        sub_index.add(cpu_index.reconstruct_n(start, end - start))
        indices.append(sub_index)
    
    return indices


def load_faiss_index_on_gpus(
    index_path: str, 
    gpu_ids: List[int]
) -> faiss.Index:
    """
    Load FAISS index on multiple GPUs.
    
    Args:
        index_path: Path to FAISS index file
        gpu_ids: List of GPU IDs to use
        
    Returns:
        Multi-GPU FAISS index
    """
    # Load CPU index
    cpu_index = faiss.read_index(index_path)
    
    # Split index
    num_gpus = len(gpu_ids)
    split_indices = split_faiss_index(cpu_index, num_gpus)
    
    # Create GPU resources
    gpu_resources = []
    for gpu_id in gpu_ids:
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)
    
    # Create sharded index
    gpu_index = faiss.IndexShards(cpu_index.d)
    
    for i, gpu_id in enumerate(gpu_ids):
        gpu_index_i = faiss.index_cpu_to_gpu(
            gpu_resources[i], 
            gpu_id, 
            split_indices[i]
        )
        gpu_index.add_shard(gpu_index_i)
    
    return gpu_index

