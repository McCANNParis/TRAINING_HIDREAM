#!/usr/bin/env python3
"""
Dataset preparation script for HiDream-I1 finetuning.
Prepares images and captions in the format expected by ai-toolkit.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import shutil
from typing import List, Dict, Tuple

def resize_and_center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize and center crop image to target size while maintaining aspect ratio.
    """
    target_width, target_height = target_size
    
    # Calculate scaling factor to fill the target size
    scale = max(target_width / image.width, target_height / image.height)
    
    # Resize image
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return image.crop((left, top, right, bottom))

def prepare_image(
    image_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (1024, 1024),
    quality: int = 95
) -> Path:
    """
    Process and save image for training.
    """
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize and crop
    img = resize_and_center_crop(img, target_size)
    
    # Save processed image
    output_path = output_dir / f"{image_path.stem}.jpg"
    img.save(output_path, 'JPEG', quality=quality)
    
    return output_path

def create_caption_file(
    image_path: Path,
    caption: str,
    output_dir: Path,
    trigger_word: str = "hidream_style"
) -> Path:
    """
    Create caption file for image.
    """
    # Add trigger word if not present
    if trigger_word and trigger_word not in caption:
        caption = f"{caption}, {trigger_word}"
    
    caption_path = output_dir / f"{image_path.stem}.txt"
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(caption)
    
    return caption_path

def prepare_dataset_from_json(
    json_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (1024, 1024),
    trigger_word: str = "hidream_style"
):
    """
    Prepare dataset from JSON file containing image paths and captions.
    
    JSON format:
    [
        {"image": "path/to/image1.jpg", "caption": "description of image 1"},
        {"image": "path/to/image2.jpg", "caption": "description of image 2"}
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(data, 1):
        print(f"Processing image {i}/{len(data)}: {item['image']}")
        
        image_path = Path(item['image'])
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Process image
            output_image = prepare_image(image_path, output_dir, target_size)
            
            # Create caption file
            create_caption_file(
                output_image,
                item['caption'],
                output_dir,
                trigger_word
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def prepare_dataset_from_folder(
    input_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (1024, 1024),
    trigger_word: str = "hidream_style",
    default_caption: str = "a high quality image"
):
    """
    Prepare dataset from folder of images.
    Looks for existing .txt files with captions, otherwise uses default.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            # Process image
            output_image = prepare_image(image_path, output_dir, target_size)
            
            # Check for existing caption
            caption_path = image_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = default_caption
            
            # Create caption file
            create_caption_file(
                output_image,
                caption,
                output_dir,
                trigger_word
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for HiDream-I1 finetuning"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input directory with images or JSON file with image-caption pairs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input/dataset"),
        help="Output directory for processed dataset (default: input/dataset)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Target image size (default: 1024)"
    )
    parser.add_argument(
        "--trigger-word",
        default="hidream_style",
        help="Trigger word to add to captions (default: hidream_style)"
    )
    parser.add_argument(
        "--default-caption",
        default="a high quality image",
        help="Default caption for images without captions"
    )
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    if args.input.suffix == '.json':
        print(f"Preparing dataset from JSON: {args.input}")
        prepare_dataset_from_json(
            args.input,
            args.output,
            target_size,
            args.trigger_word
        )
    elif args.input.is_dir():
        print(f"Preparing dataset from folder: {args.input}")
        prepare_dataset_from_folder(
            args.input,
            args.output,
            target_size,
            args.trigger_word,
            args.default_caption
        )
    else:
        print(f"Error: {args.input} is not a directory or JSON file")
        return 1
    
    print(f"\nDataset prepared successfully in: {args.output}")
    print(f"Total images: {len(list(args.output.glob('*.jpg')))}")
    
    return 0

if __name__ == "__main__":
    exit(main())