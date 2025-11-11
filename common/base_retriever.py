"""
Base Retriever

Abstract base class for knowledge retrieval across datasets.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """Dataset for batch image encoding."""
    
    def __init__(self, image_paths: List[str], clip_processor):
        self.image_paths = image_paths
        self.clip_processor = clip_processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        return self.clip_processor(image), image_path


class BaseRetriever:
    """
    Base class for knowledge retrieval operations.
    
    Provides common methods for:
    - Text encoding (LongCLIP)
    - Image encoding (LongCLIP)
    - Keyword extraction
    - FAISS search operations
    """
    
    def __init__(
        self,
        longclip_model_path: str = "../longclip/longclip-L.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize retriever.
        
        Args:
            longclip_model_path: Path to LongCLIP model weights
            device: Device to run models on
        """
        self.device = device
        self.longclip_model_path = longclip_model_path
        
        # Lazy load models
        self.clip_model = None
        self.clip_processor = None
        
    def _load_clip_model(self):
        """Lazy load LongCLIP model."""
        if self.clip_model is None:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import longclip
            
            self.clip_model, self.clip_processor = longclip.load(
                self.longclip_model_path,
                device=self.device
            )
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using LongCLIP.
        
        Args:
            text: Input text string
            
        Returns:
            Text feature vector (numpy array)
        """
        self._load_clip_model()
        import longclip
        
        inputs = longclip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(inputs).to(self.device)
        return text_features.cpu().numpy().astype('float32')
    
    def encode_images(
        self, 
        image_paths: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode multiple images using LongCLIP (batched).
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for encoding
            
        Returns:
            Tuple of (feature_vectors, image_paths)
        """
        self._load_clip_model()
        
        dataset = ImageDataset(image_paths, self.clip_processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_features = []
        all_image_paths = []
        
        with torch.no_grad():
            for inputs, paths in dataloader:
                inputs = inputs.to(self.device)
                image_features = self.clip_model.encode_image(inputs)
                all_features.append(image_features.cpu().numpy())
                all_image_paths.extend(paths)
        
        all_features = np.vstack(all_features).astype('float32')
        return all_features, all_image_paths
    
    def encode_single_image(self, image_path: str) -> np.ndarray:
        """
        Encode a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image feature vector
        """
        self._load_clip_model()
        
        image = Image.open(image_path)
        img_processed = self.clip_processor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_feature = self.clip_model.encode_image(img_processed).cpu().numpy()
        
        return image_feature.astype('float32')
    
    def search_within_entity(
        self,
        query: str,
        entity_name: str,
        entity_index: Any,
        k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search within an entity's knowledge units (fine-grained retrieval).
        
        This is the main retrieval method (formerly search_text2).
        
        Args:
            query: Question text
            entity_name: Name of the entity
            entity_index: FAISS index of entity's knowledge units
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        from .utils import extract_keywords
        
        # Extract keywords
        keywords = extract_keywords(query)
        
        # Enhanced query: question + entity + keywords
        enhanced_query = f"{query} [SEP] {entity_name} [SEP] {keywords}"
        
        # Encode and search
        query_vector = self.encode_text(enhanced_query)
        query_vector = torch.tensor(query_vector).numpy()
        distances, indices = entity_index.search(query_vector, k=k)
        
        return distances, indices
    
    def search_global(
        self,
        query: str,
        global_index: Any,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in global knowledge base (coarse-grained retrieval).
        
        This is for testing/comparison (formerly search_text).
        
        Args:
            query: Question text
            global_index: Global FAISS index
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        from .utils import extract_keywords
        
        keywords = extract_keywords(query)
        enhanced_query = f"{query} [SEP] {keywords}"
        
        query_vector = self.encode_text(enhanced_query)
        query_vector = torch.tensor(query_vector).numpy()
        distances, indices = global_index.search(query_vector, k=k)
        
        return distances, indices

