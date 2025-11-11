"""
Image Merger Module

Provides functionality to merge multiple images vertically with labels.
"""

from typing import List
from PIL import Image, ImageDraw, ImageFont


def merge_images(
    images: List[Image.Image], 
    save_path: str,
    font_path: str = 'DejaVuSerif.ttf',
    font_size: int = 15,
    spacing: int = 50,
    separator_height: int = 5
) -> None:
    """
    Merge multiple PIL Images vertically with labels and separators.
    
    Args:
        images: List of PIL Image objects to merge
        save_path: Output path for merged image
        font_path: Path to TrueType font file
        font_size: Font size for image labels
        spacing: Vertical spacing between images (pixels)
        separator_height: Height of black separator line (pixels)
    """
    if not images:
        raise ValueError("Image list cannot be empty")
        
    max_width = max(image.width for image in images)
    total_height = (
        sum(image.height for image in images) + 
        spacing * len(images) + 
        separator_height * len(images) + 
        25  # Initial offset
    )
    
    # Create blank canvas
    merged_image = Image.new("RGB", (max_width, total_height), "white")
    font = ImageFont.truetype(font_path, font_size)
    
    y_offset = 0
    
    for i, image in enumerate(images):
        draw = ImageDraw.Draw(merged_image)
        
        # Add spacing
        blank = 25 if i == 0 else spacing
        draw.rectangle(
            [0, y_offset, max_width, y_offset + blank], 
            fill="white"
        )
        y_offset += blank
        text_y = y_offset - 20
        
        # Paste image
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height
        
        # Add separator
        draw.rectangle(
            [0, y_offset, max_width, y_offset + separator_height], 
            fill="black"
        )
        y_offset += separator_height
        
        # Add label
        label_text = f'Image {i}'
        draw.text((2, text_y), label_text, fill='red', font=font)
    
    # Save merged image
    merged_image.save(save_path)

