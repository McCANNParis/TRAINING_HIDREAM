#!/usr/bin/env python3
"""
Training script for HiDream-I1 finetuning using ai-toolkit.
This script provides a wrapper around ai-toolkit with HiDream-specific optimizations.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch
import json
from typing import Dict, Any

def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. Training requires CUDA.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        gpu_name = props.name
        print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        
        # L40S detection and optimization note
        if "L40S" in gpu_name or "L40" in gpu_name:
            print(f"  ✓ NVIDIA L40S detected - Excellent for training with 48GB VRAM")
        elif memory_gb < 24:
            print(f"  WARNING: GPU {i} has less than 24GB VRAM. Training may be slow or fail.")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required = [
        "transformers",
        "diffusers",
        "accelerate",
        "flash_attn",
        "bitsandbytes"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please run: pip install " + " ".join(missing))
        return False
    
    return True

def optimize_config_for_vram(config_path: Path, vram_gb: float, gpu_name: str = "", dataset_path: Path = None) -> Dict[str, Any]:
    """
    Optimize training configuration based on available VRAM.
    Special optimizations for L40S GPU (48GB VRAM).
    Updates dataset path if provided.
    """
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Handle new config structure
    if 'config' in config and 'process' in config['config']:
        # New structure
        train_config = config['config']['process'][0]['train']
        model_config = config['config']['process'][0]['model']
        
        # Update dataset path if provided
        if dataset_path:
            config['config']['process'][0]['dataset']['datasets'][0]['folder_path'] = str(dataset_path)
    else:
        # Old structure fallback
        train_config = config.get('job', {}).get('extension_args', {}).get('process', [{}])[0].get('train', {})
        model_config = config.get('job', {}).get('extension_args', {}).get('process', [{}])[0].get('model', {})
        
        # Update dataset path if provided
        if dataset_path and 'job' in config:
            config['job']['extension_args']['process'][0]['datasets'][0]['folder_path'] = str(dataset_path)
    
    # L40S specific optimizations (48GB VRAM)
    if "L40S" in gpu_name or "L40" in gpu_name or vram_gb >= 40:
        print("Optimizing for L40S GPU (48GB VRAM)")
        train_config['batch_size'] = 4
        train_config['gradient_accumulation_steps'] = 1
        train_config['gradient_checkpointing'] = False  # Not needed with 48GB
        model_config['quantize'] = False  # Full precision for best quality
        model_config['name_or_path'] = "HiDream-ai/HiDream-I1-Full"  # Full model
        # Can increase training steps for better quality
        train_config['steps'] = 3000
        # Enable mixed precision for speed
        train_config['dtype'] = 'bf16'
        
    elif vram_gb < 16:
        print("Optimizing for low VRAM (<16GB)")
        train_config['batch_size'] = 1
        train_config['gradient_accumulation_steps'] = 8
        train_config['gradient_checkpointing'] = True
        model_config['quantize'] = True
        # Use smaller model variant
        model_config['name_or_path'] = "HiDream-ai/HiDream-I1-Fast"
        
    elif vram_gb < 24:
        print("Optimizing for medium VRAM (16-24GB)")
        train_config['batch_size'] = 1
        train_config['gradient_accumulation_steps'] = 4
        train_config['gradient_checkpointing'] = True
        model_config['quantize'] = True
        model_config['name_or_path'] = "HiDream-ai/HiDream-I1-Fast"
        
    else:
        print("Using standard high VRAM configuration (24-40GB)")
        train_config['batch_size'] = 2
        train_config['gradient_accumulation_steps'] = 2
        train_config['gradient_checkpointing'] = False
        model_config['quantize'] = False
        # Can use full model
        model_config['name_or_path'] = "HiDream-ai/HiDream-I1-Dev"
    
    return config

def validate_dataset(dataset_path: Path) -> bool:
    """
    Validate that dataset is properly formatted.
    """
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    captions = list(dataset_path.glob("*.txt"))
    
    if not images:
        print(f"ERROR: No images found in {dataset_path}")
        return False
    
    print(f"Found {len(images)} images and {len(captions)} caption files")
    
    # Check if each image has a caption
    missing_captions = []
    for img in images:
        caption_file = img.with_suffix('.txt')
        if not caption_file.exists():
            missing_captions.append(img.name)
    
    if missing_captions:
        print(f"WARNING: {len(missing_captions)} images missing captions:")
        for name in missing_captions[:5]:
            print(f"  - {name}")
        if len(missing_captions) > 5:
            print(f"  ... and {len(missing_captions) - 5} more")
    
    return True

def run_training(config_path: Path, resume: bool = False, dataset_path: Path = None):
    """
    Run the training process using ai-toolkit.
    Updates config with dataset path if provided.
    """
    # Update config file with dataset path if needed
    if dataset_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update dataset path in config - handle new structure
        if 'config' in config and 'process' in config['config']:
            # New structure: config.process[0].dataset.datasets[0].folder_path
            config['config']['process'][0]['dataset']['datasets'][0]['folder_path'] = str(dataset_path)
        else:
            # Fallback for old structure if it exists
            if 'job' in config and config['job'] == 'extension':
                if 'extension_args' in config:
                    config['extension_args']['process'][0]['datasets'][0]['folder_path'] = str(dataset_path)
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Updated config with dataset path: {dataset_path}")
    
    # Check multiple possible ai-toolkit locations
    ai_toolkit_paths = [
        Path("/workspace/ai-toolkit"),  # Main workspace location
        Path("/workspace/hidream_finetune/ai-toolkit"),  # Alternative location
        Path("./ai-toolkit"),  # Local directory
    ]
    
    ai_toolkit_path = None
    for path in ai_toolkit_paths:
        if path.exists() and (path / "run.py").exists():
            ai_toolkit_path = path
            print(f"Found ai-toolkit at: {ai_toolkit_path}")
            break
    
    if ai_toolkit_path:
        # Use ai-toolkit installation
        cmd = ["python", str(ai_toolkit_path / "run.py"), str(config_path)]
    else:
        # Try to use ai-toolkit directly if it's installed as a package
        try:
            import ai_toolkit
            # Use ai-toolkit module directly
            cmd = ["python", "-m", "ai_toolkit.run", str(config_path)]
        except ImportError:
            # Fallback to diffusers training script
            print("ai-toolkit not found. Using direct diffusers training.")
            return run_diffusers_training(config_path, resume)
    
    if resume:
        cmd.append("--resume")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def run_diffusers_training(config_path: Path, resume: bool = False):
    """
    Run training directly with diffusers if ai-toolkit is not available.
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract training parameters from config
    train_args = config['job']['extension_args']['process'][0]
    
    # Build accelerate launch command
    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "train_lora_diffusers.py",  # We'll create this script
        "--model_name", train_args['train']['model']['name_or_path'],
        "--dataset_path", train_args['datasets'][0]['folder_path'],
        "--output_dir", train_args['training_folder'],
        "--batch_size", str(train_args['train']['batch_size']),
        "--num_train_steps", str(train_args['train']['steps']),
        "--learning_rate", str(train_args['train']['lr']),
        "--lora_rank", str(train_args['network']['linear']),
        "--lora_alpha", str(train_args['network']['linear_alpha']),
    ]
    
    if resume:
        cmd.append("--resume_from_checkpoint")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        # If diffusers training script doesn't exist, provide instructions
        if not Path("train_lora_diffusers.py").exists():
            print("\nERROR: Neither ai-toolkit nor train_lora_diffusers.py found.")
            print("\nTo fix this, you have two options:")
            print("\n1. Install ai-toolkit:")
            print("   git clone https://github.com/ostris/ai-toolkit.git /workspace/hidream_finetune/ai-toolkit")
            print("   cd /workspace/hidream_finetune/ai-toolkit")
            print("   pip install -r requirements.txt")
            print("\n2. Or create a custom training script (we can generate one if needed)")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train HiDream-I1 model using ai-toolkit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hidream_i1_finetune.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset directory (defaults to workspace/ai-toolkit/assets or input/dataset)"
    )
    parser.add_argument(
        "--use-workspace-dataset",
        action="store_true",
        help="Use dataset from workspace/ai-toolkit/assets"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically optimize config for available VRAM"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup without training"
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset:
        # User specified a custom path
        dataset_path = args.dataset
    elif args.use_workspace_dataset:
        # Use workspace/ai-toolkit/assets
        dataset_path = Path("/workspace/ai-toolkit/assets")
        if not dataset_path.exists():
            # Try alternative workspace paths
            alt_paths = [
                Path("workspace/ai-toolkit/assets"),
                Path("../workspace/ai-toolkit/assets"),
                Path("../../workspace/ai-toolkit/assets")
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    dataset_path = alt_path.resolve()
                    break
    else:
        # Default behavior: check workspace first, then fall back to input/dataset
        workspace_path = Path("/workspace/ai-toolkit/assets")
        local_path = Path("input/dataset")
        
        if workspace_path.exists():
            dataset_path = workspace_path
            print(f"Using workspace dataset: {dataset_path}")
        elif local_path.exists():
            dataset_path = local_path
            print(f"Using local dataset: {dataset_path}")
        else:
            # Try to find workspace path relative to current location
            alt_paths = [
                Path("workspace/ai-toolkit/assets"),
                Path("../workspace/ai-toolkit/assets"),
                Path("../../workspace/ai-toolkit/assets")
            ]
            dataset_path = local_path  # Default
            for alt_path in alt_paths:
                if alt_path.exists():
                    dataset_path = alt_path.resolve()
                    print(f"Found workspace dataset at: {dataset_path}")
                    break
    
    print("="*50)
    print("HiDream-I1 Finetuning Setup")
    print("="*50)
    print(f"Dataset path: {dataset_path}")
    
    # Check GPU
    if not check_gpu():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Validate dataset
    if not validate_dataset(dataset_path):
        return 1
    
    # Auto-optimize if requested
    if args.auto_optimize:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        gpu_name = props.name
        config = optimize_config_for_vram(args.config, vram_gb, gpu_name, dataset_path)
        
        # Save optimized config
        optimized_path = args.config.parent / f"{args.config.stem}_optimized.yaml"
        import yaml
        with open(optimized_path, 'w') as f:
            yaml.dump(config, f)
        args.config = optimized_path
        print(f"Saved optimized config to: {optimized_path}")
    
    if args.validate_only:
        print("\nValidation complete. Ready to train!")
        print(f"Run training with: python train_hidream.py --config {args.config}")
        return 0
    
    # Start training
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    success = run_training(args.config, args.resume, dataset_path)
    
    if success:
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Check output in: output/hidream_i1_finetune/")
        print("="*50)
    else:
        print("\nTraining failed or was interrupted")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())