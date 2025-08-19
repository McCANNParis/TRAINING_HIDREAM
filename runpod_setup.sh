#!/bin/bash

echo "==================================================="
echo "HiDream-I1 Finetuning Setup"
echo "==================================================="

# Function to check if running in RunPod
check_runpod() {
    if [ -n "$RUNPOD_POD_ID" ] || [ -d "/workspace" ]; then
        echo "✓ RunPod environment detected"
        return 0
    else
        echo "⚠ Not running in RunPod - adjusting paths for local setup"
        return 1
    fi
}

# Detect and display GPU information
echo ""
echo "GPU Detection:"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=',' read -r index name memory; do
        # Convert memory from MiB to GB
        memory_gb=$(echo "scale=1; ${memory%MiB}/1024" | bc 2>/dev/null || echo "${memory}")
        echo "  GPU $index: $name ($memory_gb GB)"
    done
    
    # Check for L40S and set optimizations
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40"; then
        echo "  ✓ NVIDIA L40S detected - Excellent for training with 48GB VRAM"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        export CUDA_LAUNCH_BLOCKING=0
        IS_L40S=true
    else
        IS_L40S=false
    fi
else
    echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Update system packages only if we have sudo/root access
if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo ""
    echo "Installing system dependencies..."
    if [ "$EUID" -eq 0 ]; then
        apt-get update && apt-get install -y \
            git \
            wget \
            curl \
            python3-pip \
            python3-dev \
            python3-venv \
            build-essential \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            bc
    else
        sudo apt-get update && sudo apt-get install -y \
            git \
            wget \
            curl \
            python3-pip \
            python3-dev \
            python3-venv \
            build-essential \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            bc
    fi
else
    echo "⚠ No sudo access - skipping system package installation"
    echo "  Assuming packages are pre-installed in container"
fi

# Set up working directory based on environment
if check_runpod; then
    WORK_DIR="/workspace/TRAINING_HIDREAM"
else
    WORK_DIR="$(pwd)"
fi

echo ""
echo "Working directory: $WORK_DIR"

# Check if we're already in the project directory
if [ -f "$WORK_DIR/train_hidream.py" ]; then
    echo "✓ Already in project directory"
    cd "$WORK_DIR"
else
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Check Python version
echo ""
echo "Python environment check:"
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✓ Using Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✓ Using Python 3.10"
else
    echo "✓ Using system Python 3"
fi

$PYTHON_CMD --version

# Upgrade pip first
echo ""
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA version
echo ""
echo "Installing PyTorch..."

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "CUDA compiler not found, using runtime version"
    CUDA_VERSION="12.1"
fi

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "12.4"* ]]; then
    echo "Installing PyTorch for CUDA 12.4..."
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "12.1"* ]] || [[ "$CUDA_VERSION" == "12.2"* ]] || [[ "$CUDA_VERSION" == "12.3"* ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch for CUDA 11.8 (fallback)..."
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Install core dependencies first
echo ""
echo "Installing core ML dependencies..."
pip install --no-cache-dir \
    transformers>=4.44.0 \
    diffusers>=0.30.0 \
    accelerate>=0.34.0 \
    safetensors>=0.4.5 \
    huggingface-hub>=0.24.0 \
    datasets>=2.20.0

# Install training dependencies
echo ""
echo "Installing training dependencies..."
pip install --no-cache-dir \
    omegaconf>=2.3.0 \
    einops>=0.8.0 \
    tensorboard>=2.17.0 \
    wandb>=0.17.0 \
    Pillow>=10.4.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0

# Install optimization libraries
echo ""
echo "Installing optimization libraries..."

# Install bitsandbytes
echo "Installing bitsandbytes..."
pip install --no-cache-dir bitsandbytes>=0.43.0

# Install xformers (memory efficient attention)
echo "Installing xformers..."
pip install --no-cache-dir xformers>=0.0.27

# Try to install Flash Attention (may fail on some systems)
echo ""
echo "Attempting Flash Attention installation..."
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "✓ Flash Attention installed successfully"
else
    echo "⚠ Flash Attention installation failed (optional - training will still work)"
fi

# Install ai-toolkit if we have a requirements.txt
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing project requirements..."
    pip install --no-cache-dir -r requirements.txt
fi

# Install additional useful tools
echo ""
echo "Installing additional tools..."
pip install --no-cache-dir \
    ipython \
    jupyter \
    matplotlib \
    scipy

# Validation check
echo ""
echo "==================================================="
echo "Validating Installation"
echo "==================================================="

# Function to check if a Python package is installed
check_package() {
    if python -c "import $1" 2>/dev/null; then
        version=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo "  ✓ $1 ($version)"
        return 0
    else
        echo "  ✗ $1 - NOT INSTALLED"
        return 1
    fi
}

# Check required packages
REQUIRED_PACKAGES=("torch" "transformers" "diffusers" "accelerate" "bitsandbytes" "safetensors" "einops" "omegaconf" "PIL" "tqdm")
OPTIONAL_PACKAGES=("flash_attn" "xformers" "wandb")

echo "Required packages:"
MISSING_REQUIRED=false
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_package "$pkg"; then
        MISSING_REQUIRED=true
    fi
done

echo ""
echo "Optional packages:"
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    check_package "$pkg"
done

# Check CUDA availability in PyTorch
echo ""
echo "PyTorch CUDA check:"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "  ✓ CUDA is available"
    echo "  PyTorch version: $TORCH_VERSION"
    echo "  CUDA version: $CUDA_VERSION"
    
    # Show GPU info from PyTorch
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null
    python -c "import torch; print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null
else
    echo "  ✗ CUDA is NOT available in PyTorch"
    echo "  This will prevent GPU training!"
    MISSING_REQUIRED=true
fi

if [ "$MISSING_REQUIRED" = true ]; then
    echo ""
    echo "ERROR: Missing required packages. Please run:"
    echo "pip install transformers diffusers accelerate bitsandbytes safetensors einops omegaconf pillow tqdm"
    echo ""
    echo "If installation fails, try:"
    echo "pip install --upgrade pip setuptools wheel"
    echo "pip install --no-cache-dir <package_name>"
    exit 1
fi

# Hugging Face login reminder
echo ""
echo "==================================================="
echo "Hugging Face Authentication"
echo "==================================================="
echo "To access HiDream models, you need to login to Hugging Face:"
echo "Run: huggingface-cli login"
echo "Get your token from: https://huggingface.co/settings/tokens"

# Create necessary directories
echo ""
echo "Setting up project directories..."
mkdir -p dataset
mkdir -p input/dataset
mkdir -p output/hidream_i1_finetune
mkdir -p config

# Check for existing scripts
if [ -f "train_hidream.py" ]; then
    echo "✓ train_hidream.py found"
else
    echo "⚠ train_hidream.py not found - please ensure it's in the current directory"
fi

if [ -f "prepare_dataset.py" ]; then
    echo "✓ prepare_dataset.py found"
else
    echo "⚠ prepare_dataset.py not found - please ensure it's in the current directory"
fi

# Copy or update configuration file
echo ""
echo "Setting up configuration..."

# Check if L40S GPU is present and create optimized config
if [ "$IS_L40S" = true ]; then
    echo "Creating L40S optimized configuration..."
    cat > config/hidream_i1_finetune.yaml << 'EOF'
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
    cat > config/hidream_i1_finetune.yaml << 'EOF'
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

echo ""
echo "==================================================="
echo "Setup Complete!"
echo "==================================================="
echo ""

# GPU-specific recommendations
if [ "$IS_L40S" = true ]; then
    echo "L40S GPU Configuration:"
    echo "  - 48GB VRAM available"
    echo "  - Optimized for batch_size=4"
    echo "  - Full HiDream-I1-Dev model enabled"
    echo "  - Extended training to 3000 steps"
    echo ""
fi

echo "Next steps:"
echo "1. Login to Hugging Face (if not already done):"
echo "   huggingface-cli login"
echo ""
echo "2. Prepare your dataset in the 'dataset' folder:"
echo "   - Place images in dataset/"
echo "   - Create .txt files with same name as images for captions"
echo "   - Example: dataset/image001.jpg and dataset/image001.txt"
echo ""
echo "3. Validate your setup:"
echo "   python train_hidream.py --validate-only"
echo ""
echo "4. Start training:"
if [ "$IS_L40S" = true ]; then
    echo "   python train_hidream.py --auto-optimize --dataset-path dataset"
else
    echo "   python train_hidream.py --auto-optimize --dataset-path dataset"
fi
echo ""
echo "5. Monitor training:"
echo "   tensorboard --logdir output/hidream_i1_finetune"
echo ""
echo "For help and documentation:"
echo "  - See HOW_TO.md for detailed instructions"
echo "  - Check README.md for project overview"