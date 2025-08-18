# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a HiDream-I1 finetuning toolkit for training custom LoRA models on the HiDream-I1 17B parameter text-to-image generation model. The toolkit wraps the ai-toolkit framework and is optimized for RunPod cloud deployment.

## Common Commands

### Setup
```bash
# RunPod environment setup
chmod +x runpod_setup.sh
./runpod_setup.sh

# Hugging Face authentication (required for Llama models)
huggingface-cli login
```

### Dataset Preparation
```bash
# From image folder
python prepare_dataset.py /path/to/images --output input/dataset

# From JSON metadata
python prepare_dataset.py dataset.json --output input/dataset
```

### Training
```bash
# Validate setup before training
python train_hidream.py --validate-only

# Auto-optimized training (detects GPU and adjusts settings - RECOMMENDED for L40S)
python train_hidream.py --auto-optimize

# Custom configuration training
python train_hidream.py --config config/hidream_i1_finetune.yaml

# Resume from checkpoint
python train_hidream.py --resume
```

### Monitoring
```bash
# TensorBoard visualization
tensorboard --logdir output/hidream_i1_finetune

# Check generated samples
ls output/hidream_i1_finetune/samples/
```

## Architecture

The codebase consists of three main components:

1. **Dataset Preparation Pipeline** (`prepare_dataset.py`)
   - Processes images from folders or JSON metadata files
   - Automatically resizes images to 1024x1024 resolution
   - Handles format conversion (RGBA to RGB)
   - Adds trigger words to captions for style control
   - Implements center-cropping and quality optimization

2. **Training Orchestration** (`train_hidream.py`)
   - Detects GPU capabilities and optimizes VRAM usage automatically
   - Special optimization for NVIDIA L40S GPUs (48GB VRAM)
   - Validates all dependencies and environment setup
   - Adjusts batch size and gradient accumulation based on available memory:
     - <16GB VRAM: HiDream-I1-Fast model, batch_size=1, gradient_accumulation=8
     - 16-24GB VRAM: HiDream-I1-Fast model, batch_size=1, gradient_accumulation=4
     - 24-40GB VRAM: HiDream-I1-Dev model, batch_size=2, gradient_accumulation=2
     - L40S (48GB): HiDream-I1-Dev model, batch_size=4, gradient_accumulation=1, 3000 steps
   - Manages checkpoint saving and recovery
   - Integrates with WandB and TensorBoard for monitoring

3. **Configuration System** (`config/hidream_i1_finetune.yaml`)
   - YAML-based configuration for training parameters
   - LoRA settings: rank 16, alpha 16 for efficiency/quality balance
   - FlowMatch noise scheduler for diffusion training
   - BF16 precision for memory optimization
   - AdamW 8-bit optimizer configuration

## Key Technical Details

- **Framework**: Built on ai-toolkit with PyTorch 2.3.0 and CUDA 12.4
- **Model Architecture**: LoRA adaptation of HiDream-I1 17B parameter model
- **Training Method**: FlowMatch diffusion with EMA (decay 0.99)
- **Optimization**: Flash Attention and XFormers for memory efficiency
- **Precision**: BF16 mixed precision training
- **Default Trigger Word**: "hidream_style" (customizable in config)

## RunPod Deployment

The project is optimized for RunPod L40S GPUs with the following configurations:
- `runpod_dockerfile`: Docker container with CUDA 12.4 optimized for L40S
- `runpod_setup.sh`: Automated environment setup with L40S detection
- `runpod_template.json`: L40S-specific deployment template with 48GB VRAM settings

### L40S GPU Optimizations
- Automatic detection of L40S hardware
- Optimized memory allocation with `expandable_segments`
- Increased batch size (4) for faster training
- Full HiDream-I1-Dev model support
- Extended training to 3000 steps for better quality

## Directory Structure

- `input/dataset/`: Prepared training datasets
- `output/hidream_i1_finetune/`: Training outputs, checkpoints, and samples
- `config/`: YAML configuration files
- Root scripts: `prepare_dataset.py`, `train_hidream.py`