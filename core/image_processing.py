"""
Image Processing Module

Provides utilities for adding captions to images and resizing images.
"""

from typing import Optional
from PIL import Image, ImageDraw, ImageFont


def text_reshaper(words: str, max_chars_per_line: int = 48) -> tuple[int, str]:
    """
    Reshape text to fit within a specified width by adding line breaks.
    
    Args:
        words: Input text string
        max_chars_per_line: Maximum characters per line before breaking
        
    Returns:
        Tuple of (number_of_lines, reshaped_text)
    """
    count = 0
    result_text = ""
    line = 0
    
    for word in words:
        result_text += word
        count += 1
        if count > max_chars_per_line and word == ' ':
            result_text += "\n"
            count = 0
            line += 1
            
    return line, result_text


def resize_width(image: Image.Image, target_width: int) -> Image.Image:
    """
    Resize image to target width while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_width: Target width in pixels
        
    Returns:
        Resized PIL Image object
    """
    aspect_ratio = image.height / image.width
    new_height = int(target_width * aspect_ratio)
    resized_image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def add_caption(
    image_path: Optional[str], 
    caption: str, 
    font_path: str = "DejaVuSans.ttf",
    font_size: int = 15,
    image_width: int = 224,
    caption_width: int = 500
) -> Image.Image:
    """
    Add caption text to an image or create a text-only image.
    
    Args:
        image_path: Path to input image, or empty string for text-only
        caption: Caption text to add
        font_path: Path to TrueType font file
        font_size: Font size for caption text
        image_width: Width to resize input image
        caption_width: Width for caption text area
        
    Returns:
        PIL Image object with caption added
    """
    font = ImageFont.truetype(font_path, font_size)
    
    # Text-only mode (no image)
    if image_path == '':
        line, text = text_reshaper(caption)
        new_height = 10
        stand_line = int(new_height / 22)
        
        if line > stand_line:
            map_height = (line - stand_line) * 22 + new_height
        else:
            map_height = new_height
            
        output_image = Image.new('RGB', (caption_width, map_height), color='white')
        draw = ImageDraw.Draw(output_image)
        draw.text((5, 5), text, fill='black', font=font)
        
    # Image with caption
    elif caption != '':
        input_image = Image.open(image_path)
        input_image = resize_width(input_image, image_width)
        
        line, text = text_reshaper(caption)
        new_height = input_image.size[1]
        stand_line = int(new_height / 22)
        
        if line > stand_line:
            map_height = (line - stand_line) * 22 + new_height
        else:
            map_height = new_height
            
        output_image = Image.new(
            'RGB', 
            (input_image.size[0] + caption_width, map_height), 
            color='white'
        )
        output_image.paste(input_image, (0, 0))
        
        draw = ImageDraw.Draw(output_image)
        draw.text((230, 5), text, fill='black', font=font)
        
    # Image only (no caption)
    else:
        input_image = Image.open(image_path)
        input_image = resize_width(input_image, 500)
        new_height = input_image.size[1]
        output_image = Image.new(
            'RGB', 
            (input_image.size[0] + image_width, new_height), 
            color='white'
        )
        output_image.paste(input_image, (0, 0))
        
    return output_image

