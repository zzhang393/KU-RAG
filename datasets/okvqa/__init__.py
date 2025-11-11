"""
OK-VQA Dataset Module

Implementation of KU-RAG for OK-VQA (Open-ended Knowledge-based VQA) dataset.

Note: Uses common.BaseMetaKnowledge and common.BaseRetriever
"""

# Import from common module
from common import BaseMetaKnowledge as MetaKnowledge
from common import BaseRetriever
from common.utils import extract_keywords, load_faiss_index, load_metadata

__all__ = [
    'MetaKnowledge',
    'BaseRetriever',
    'extract_keywords',
    'load_faiss_index',
    'load_metadata',
]
