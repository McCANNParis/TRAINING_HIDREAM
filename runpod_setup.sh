#!/bin/bash

echo "==================================="
echo "HiDream-I1 Finetuning Setup on RunPod"
echo "==================================="

# Detect GPU type
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40"; then
    echo "✓ NVIDIA L40S GPU detected - Optimized for 48GB VRAM"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export CUDA_LAUNCH_BLOCKING=0
fi

# Update system packages
apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Set up working directory
WORK_DIR="/workspace/hidream_finetune"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clone ai-toolkit repository
echo "Cloning ai-toolkit repository..."
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support for Flash Attention (L40S compatible)
echo "Installing PyTorch with CUDA 12.4 (L40S optimized)..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention
echo "Installing Flash Attention..."
pip install flash-attn --no-build-isolation

# Install ai-toolkit dependencies
echo "Installing ai-toolkit dependencies..."
pip install -r requirements.txt

# Install additional dependencies for HiDream-I1
echo "Installing HiDream-I1 specific dependencies..."
pip install \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    omegaconf \
    einops \
    xformers \
    bitsandbytes \
    wandb

# Login to Hugging Face (required for Llama model access)
echo "Please login to Hugging Face to access Llama models:"
echo "Run: huggingface-cli login"
echo "You'll need your HF token with access to meta-llama/Llama-3.2-1B-Instruct"

# Create necessary directories
mkdir -p input/dataset
mkdir -p output/hidream_i1_finetune
mkdir -p configs

# Copy helper scripts if they exist in /workspace
if [ -f "/workspace/train_hidream.py" ]; then
    cp /workspace/train_hidream.py .
    echo "✓ Copied train_hidream.py"
fi

if [ -f "/workspace/prepare_dataset.py" ]; then
    cp /workspace/prepare_dataset.py .
    echo "✓ Copied prepare_dataset.py"
fi

# Copy configuration file
echo "Setting up configuration..."

# Check if L40S GPU is present and create optimized config
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40"; then
    echo "Creating L40S optimized configuration..."
    cat > configs/hidream_i1_finetune.yaml << 'EOF'
job:
  extension: ai_toolkit.extensions.sd_trainer
  extension_args:
    process:
      - type: "train"
        training_folder: "output/hidream_i1_finetune"
        device: "cuda:0"
        trigger_word: "hidream_style"
        network:
          type: "lora"
          linear: 16
          linear_alpha: 16
        save:
          dtype: "float16"
          save_every: 500
          max_step_saves_to_keep: 4
        datasets:
          - folder_path: "input/dataset"
            caption_ext: "txt"
            caption_dropout_rate: 0.05
            shuffle_tokens: false
            cache_latents_to_disk: true
            resolution: [1024, 1024]
        train:
          batch_size: 4
          steps: 3000
          gradient_accumulation_steps: 1
          train_unet: true
          train_text_encoder: false
          gradient_checkpointing: false
          noise_scheduler: "flowmatch"
          optimizer: "adamw8bit"
          lr: 0.0001
          ema_config:
            use_ema: true
            ema_decay: 0.99
          dtype: "bf16"
          model:
            name_or_path: "HiDream-ai/HiDream-I1-Dev"
            is_flux: false
            quantize: false
          # Optional: Generate sample images during training to monitor progress
          # Uncomment and customize the prompts based on your training data
          # sample:
          #   sampler: "flowmatch"
          #   sample_every: 250
          #   width: 1024
          #   height: 1024
          #   prompts:
          #     - "your test prompt here, hidream_style"
          #   neg: ""
          #   seed: 42
          #   walk_seed: true
          #   guidance_scale: 3.5
          #   sample_steps: 16
EOF
else
    echo "Creating standard configuration..."
    cat > configs/hidream_i1_finetune.yaml << 'EOF'
job:
  extension: ai_toolkit.extensions.sd_trainer
  extension_args:
    process:
      - type: "train"
        training_folder: "output/hidream_i1_finetune"
        device: "cuda:0"
        trigger_word: "hidream_style"
        network:
          type: "lora"
          linear: 16
          linear_alpha: 16
        save:
          dtype: "float16"
          save_every: 500
          max_step_saves_to_keep: 4
        datasets:
          - folder_path: "input/dataset"
            caption_ext: "txt"
            caption_dropout_rate: 0.05
            shuffle_tokens: false
            cache_latents_to_disk: true
            resolution: [1024, 1024]
        train:
          batch_size: 1
          steps: 2000
          gradient_accumulation_steps: 4
          train_unet: true
          train_text_encoder: false
          gradient_checkpointing: true
          noise_scheduler: "flowmatch"
          optimizer: "adamw8bit"
          lr: 0.0001
          ema_config:
            use_ema: true
            ema_decay: 0.99
          dtype: "bf16"
          model:
            name_or_path: "HiDream-ai/HiDream-I1-Fast"
            is_flux: false
            quantize: true
          # Optional: Generate sample images during training to monitor progress
          # Uncomment and customize the prompts based on your training data
          # sample:
          #   sampler: "flowmatch"
          #   sample_every: 250
          #   width: 1024
          #   height: 1024
          #   prompts:
          #     - "your test prompt here, hidream_style"
          #   neg: ""
          #   seed: 42
          #   walk_seed: true
          #   guidance_scale: 3.5
          #   sample_steps: 16
EOF
fi

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""

# Check if L40S GPU is present
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40"; then
    echo "L40S GPU Configuration:"
    echo "- 48GB VRAM available"
    echo "- Optimized for batch_size=4"
    echo "- Full HiDream-I1-Dev model enabled"
    echo "- Use --auto-optimize flag for automatic configuration"
    echo ""
fi

echo "Next steps:"
echo "1. Login to Hugging Face: huggingface-cli login"
echo "2. Upload your dataset to input/dataset/"
echo "3. For L40S optimized training: python train_hidream.py --auto-optimize"
echo "   Or standard training: python run.py configs/hidream_i1_finetune.yaml"
echo ""
echo "Dataset format:"
echo "- Place images in input/dataset/"
echo "- Create .txt files with same name as images for captions"
echo "- Example: image001.jpg and image001.txt"