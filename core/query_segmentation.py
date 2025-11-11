"""
Query-Aware Instance Segmentation Module

Performs instance segmentation guided by query text to identify relevant objects.
"""

import os
from typing import List, Optional, Tuple
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from ultralytics import YOLO


class QueryAwareSegmentation:
    """
    Query-aware instance segmentation using YOLOv8 and LongCLIP.
    
    This class performs instance segmentation on images and filters segments
    based on their relevance to a query text using CLIP similarity.
    """
    
    def __init__(
        self, 
        yolo_model_path: str = 'yolov8x-seg.pt',
        longclip_model_path: str = './longclip/longclip-L.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        similarity_threshold: float = 0.19,
        min_box_size: int = 50
    ):
        """
        Initialize QueryAwareSegmentation.
        
        Args:
            yolo_model_path: Path to YOLOv8 segmentation model weights
            longclip_model_path: Path to LongCLIP model weights
            device: Device to run models on ('cuda' or 'cpu')
            similarity_threshold: Minimum CLIP similarity for segment selection
            min_box_size: Minimum bounding box size (width or height)
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.min_box_size = min_box_size
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize LongCLIP model (lazy loading)
        self.clip_model = None
        self.clip_processor = None
        self.longclip_model_path = longclip_model_path
        
    def _load_clip_model(self):
        """Lazy load LongCLIP model when needed."""
        if self.clip_model is None:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import longclip
            
            self.clip_model, self.clip_processor = longclip.load(
                self.longclip_model_path, 
                device=self.device
            )
    
    def instance_segment(self, image_path: str) -> List[np.ndarray]:
        """
        Perform instance segmentation on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of segmented image arrays (CHW format)
        """
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        segmented_images = []
        
        if results[0].masks is None:
            return segmented_images
        
        for result in results:
            masks = result.masks.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            # Check minimum box size
            if len(boxes) > 0:
                if (boxes[0][2] - boxes[0][0] < self.min_box_size or 
                    boxes[0][3] - boxes[0][1] < self.min_box_size):
                    return segmented_images
            
            for mask, box in zip(masks, boxes):
                try:
                    # Resize mask to image dimensions
                    mask_resized = cv2.resize(
                        mask.data.astype(np.uint8)[0],
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception:
                    return segmented_images
                
                mask_bool = mask_resized.astype(bool)
                segmented_image = np.zeros_like(image)
                segmented_image[mask_bool] = image[mask_bool]
                
                # Convert to CHW format
                segmented_image_chw = np.transpose(segmented_image, (2, 0, 1))
                segmented_images.append(segmented_image_chw)
        
        return segmented_images
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using LongCLIP.
        
        Args:
            text: Input text string
            
        Returns:
            Text feature vector
        """
        self._load_clip_model()
        import longclip
        
        inputs = longclip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(inputs)
        return text_features
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image using LongCLIP.
        
        Args:
            image: Input image array (CHW format)
            
        Returns:
            Image feature vector
        """
        self._load_clip_model()
        
        to_pil_image = ToPILImage()
        tensor_image = torch.tensor(image)
        pil_image = to_pil_image(tensor_image)
        
        img_processed = self.clip_processor(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.clip_model.encode_image(img_processed)
        return image_feature
    
    def segment(
        self, 
        query: str, 
        image_path: str, 
        output_path: Optional[str] = None
    ) -> bool:
        """
        Perform query-aware segmentation and save result.
        
        Args:
            query: Query text to guide segmentation
            image_path: Path to input image
            output_path: Path to save segmented image (optional)
            
        Returns:
            True if segmentation successful, False otherwise
        """
        seg_imgs = self.instance_segment(image_path)
        if not seg_imgs:
            return False
        
        query_vec = self.encode_text(query)
        selected_segments = []
        
        for seg_img in seg_imgs:
            img_vec = self.encode_image(seg_img)
            cross_sim = torch.cosine_similarity(query_vec, img_vec).item()
            
            if cross_sim >= self.similarity_threshold:
                selected_segments.append(seg_img)
        
        # Only save if exactly one segment matches
        if len(selected_segments) == 1:
            if output_path is None:
                image_name = os.path.basename(image_path)
                output_path = f'./image_seg/{image_name}'
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert from CHW to HWC and save
            segmented_image = np.transpose(selected_segments[0], (1, 2, 0))
            cv2.imwrite(output_path, segmented_image)
            return True
        
        return False


def segment(
    query: str, 
    image_path: str, 
    output_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> bool:
    """
    Convenience function for query-aware segmentation.
    
    Args:
        query: Query text to guide segmentation
        image_path: Path to input image
        output_path: Path to save segmented image (optional)
        device: Device to run on
        **kwargs: Additional arguments for QueryAwareSegmentation
        
    Returns:
        True if segmentation successful, False otherwise
    """
    segmenter = QueryAwareSegmentation(device=device, **kwargs)
    return segmenter.segment(query, image_path, output_path)

