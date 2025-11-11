"""
KU-RAG Core Module

This module contains core functionalities for image processing, 
query-aware segmentation, and visual passage generation.
"""

from .image_processing import add_caption, text_reshaper, resize_width
from .image_merger import merge_images
from .query_segmentation import QueryAwareSegmentation, segment

# OVEN-specific segmentation (if needed separately)
try:
    from . import oven_segmentation
except ImportError:
    oven_segmentation = None

__all__ = [
    'add_caption',
    'text_reshaper',
    'resize_width',
    'merge_images',
    'QueryAwareSegmentation',
    'segment',
    'oven_segmentation',
]
