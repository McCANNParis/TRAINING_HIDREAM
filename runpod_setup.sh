#!/bin/bash

echo "==================================================="
echo "HiDream-I1 Finetuning Setup - Optimized for RunPod"
echo "==================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track what needs to be installed
declare -a MISSING_SYSTEM_PACKAGES=()
declare -a MISSING_PYTHON_PACKAGES=()
NEEDS_PYTORCH_UPDATE=false

# Function to check if running in RunPod
check_runpod() {
    if [ -n "$RUNPOD_POD_ID" ] || [ -d "/workspace" ]; then
        echo -e "${GREEN}✓${NC} RunPod environment detected (Pod ID: ${RUNPOD_POD_ID:-unknown})"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Not running in RunPod - adjusting paths for local setup"
        return 1
    fi
}

# Function to check if a system package is installed
check_system_package() {
    local package=$1
    if dpkg -l "$package" 2>/dev/null | grep -q "^ii"; then
        return 0
    else
        return 1
    fi
}

# Function to check Python package with version
check_python_package() {
    local package=$1
    local min_version=$2
    local import_name=${3:-$package}
    
    # Handle special import names
    if [ "$import_name" = "PIL" ]; then
        import_name="PIL"
        version_check="import PIL; print(PIL.__version__)"
    else
        version_check="import $import_name; print($import_name.__version__ if hasattr($import_name, '__version__') else 'unknown')"
    fi
    
    if python -c "import $import_name" 2>/dev/null; then
        local installed_version=$(python -c "$version_check" 2>/dev/null || echo "unknown")
        
        if [ -n "$min_version" ] && [ "$installed_version" != "unknown" ]; then
            # Simple version comparison (may need refinement for complex cases)
            if python -c "from packaging import version; exit(0 if version.parse('$installed_version') >= version.parse('$min_version') else 1)" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $package ($installed_version)"
                return 0
            else
                echo -e "  ${YELLOW}⚠${NC} $package ($installed_version) - needs update to >=$min_version"
                return 1
            fi
        else
            echo -e "  ${GREEN}✓${NC} $package ($installed_version)"
            return 0
        fi
    else
        echo -e "  ${RED}✗${NC} $package - not installed"
        return 1
    fi
}

# Detect and display GPU information
echo ""
echo -e "${BLUE}GPU Detection:${NC}"
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
        echo -e "  ${GREEN}✓${NC} NVIDIA L40S detected - Excellent for training with 48GB VRAM"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        export CUDA_LAUNCH_BLOCKING=0
        IS_L40S=true
    else
        IS_L40S=false
    fi
else
    echo -e "${RED}ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed.${NC}"
    exit 1
fi

# Check system packages
echo ""
echo -e "${BLUE}Checking system packages:${NC}"
SYSTEM_PACKAGES=("git" "wget" "curl" "python3-pip" "python3-dev" "python3-venv" "build-essential" "libgl1-mesa-glx" "libglib2.0-0" "libsm6" "libxext6" "libxrender-dev" "libgomp1" "bc")

for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if check_system_package "$pkg"; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg - will install"
        MISSING_SYSTEM_PACKAGES+=("$pkg")
    fi
done

# Install missing system packages if any
if [ ${#MISSING_SYSTEM_PACKAGES[@]} -gt 0 ]; then
    if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Installing ${#MISSING_SYSTEM_PACKAGES[@]} missing system packages...${NC}"
        if [ "$EUID" -eq 0 ]; then
            apt-get update && apt-get install -y "${MISSING_SYSTEM_PACKAGES[@]}"
        else
            sudo apt-get update && sudo apt-get install -y "${MISSING_SYSTEM_PACKAGES[@]}"
        fi
    else
        echo -e "${YELLOW}⚠ No sudo access - cannot install system packages${NC}"
        echo "Missing packages: ${MISSING_SYSTEM_PACKAGES[*]}"
    fi
else
    echo -e "${GREEN}All system packages already installed!${NC}"
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
    echo -e "${GREEN}✓${NC} Already in project directory"
    cd "$WORK_DIR"
else
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Check Python version
echo ""
echo -e "${BLUE}Python environment check:${NC}"
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✓${NC} Using Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✓${NC} Using Python 3.10"
else
    echo -e "${GREEN}✓${NC} Using system Python 3"
fi

$PYTHON_CMD --version

# First install packaging for version comparison
pip install -q packaging 2>/dev/null

# Check CUDA and PyTorch status
echo ""
echo -e "${BLUE}PyTorch and CUDA check:${NC}"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "  ${GREEN}✓${NC} PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION"
    
    # Check if we need correct CUDA version
    if command -v nvcc &> /dev/null; then
        SYSTEM_CUDA=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        if [[ "$SYSTEM_CUDA" == "12.4"* ]] && [[ "$CUDA_VERSION" != "12."* ]]; then
            echo -e "  ${YELLOW}⚠${NC} PyTorch CUDA version mismatch - will reinstall"
            NEEDS_PYTORCH_UPDATE=true
        fi
    fi
else
    echo -e "  ${RED}✗${NC} PyTorch not installed or CUDA not available"
    NEEDS_PYTORCH_UPDATE=true
fi

# Check Python packages
echo ""
echo -e "${BLUE}Checking Python packages:${NC}"

# Define required packages with versions and import names
declare -A PYTHON_PACKAGES=(
    ["transformers>=4.44.0"]="transformers"
    ["diffusers>=0.30.0"]="diffusers"
    ["accelerate>=0.34.0"]="accelerate"
    ["safetensors>=0.4.5"]="safetensors"
    ["huggingface-hub>=0.24.0"]="huggingface_hub"
    ["datasets>=2.20.0"]="datasets"
    ["omegaconf>=2.3.0"]="omegaconf"
    ["einops>=0.8.0"]="einops"
    ["tensorboard>=2.17.0"]="tensorboard"
    ["wandb>=0.17.0"]="wandb"
    ["Pillow>=10.4.0"]="PIL"
    ["tqdm>=4.66.0"]="tqdm"
    ["pyyaml>=6.0"]="yaml"
    ["oyaml>=1.0"]="oyaml"
    ["python-dotenv>=1.0.0"]="dotenv"
    ["lycoris-lora>=2.2.0"]="lycoris"
    ["prodigyopt>=1.0"]="prodigyopt"
    ["bitsandbytes>=0.43.0"]="bitsandbytes"
    ["xformers>=0.0.27"]="xformers"
)

for pkg_spec in "${!PYTHON_PACKAGES[@]}"; do
    import_name="${PYTHON_PACKAGES[$pkg_spec]}"
    pkg_name=$(echo "$pkg_spec" | cut -d'>' -f1 | cut -d'=' -f1)
    min_version=$(echo "$pkg_spec" | grep -oP '(?<=>)=[\d.]+' | cut -d'=' -f2)
    
    if ! check_python_package "$pkg_name" "$min_version" "$import_name"; then
        MISSING_PYTHON_PACKAGES+=("$pkg_spec")
    fi
done

# Upgrade pip if needed
echo ""
if [ ${#MISSING_PYTHON_PACKAGES[@]} -gt 0 ] || [ "$NEEDS_PYTORCH_UPDATE" = true ]; then
    echo -e "${YELLOW}Upgrading pip...${NC}"
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel
fi

# Install PyTorch if needed
if [ "$NEEDS_PYTORCH_UPDATE" = true ]; then
    echo ""
    echo -e "${YELLOW}Installing/Updating PyTorch...${NC}"
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "System CUDA Version: $CUDA_VERSION"
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
fi

# Install missing Python packages
if [ ${#MISSING_PYTHON_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Installing ${#MISSING_PYTHON_PACKAGES[@]} missing Python packages...${NC}"
    for pkg in "${MISSING_PYTHON_PACKAGES[@]}"; do
        echo "  Installing $pkg..."
        pip install --no-cache-dir "$pkg"
    done
else
    echo -e "${GREEN}All Python packages already installed!${NC}"
fi

# Try to install Flash Attention (optional but with version check)
echo ""
echo -e "${BLUE}Checking Flash Attention:${NC}"
if python -c "import flash_attn" 2>/dev/null; then
    FLASH_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
    echo "  Current Flash Attention version: $FLASH_VERSION"
    
    # Check if version is compatible with xformers (2.7.1-2.8.2)
    if python -c "from packaging import version; v = version.parse('$FLASH_VERSION'); exit(0 if version.parse('2.7.1') <= v <= version.parse('2.8.2') else 1)" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Flash Attention $FLASH_VERSION is compatible with xformers"
    else
        echo -e "  ${YELLOW}⚠${NC} Flash Attention $FLASH_VERSION is incompatible with xformers"
        echo "  Installing compatible version (2.8.2)..."
        pip uninstall flash-attn -y 2>/dev/null
        if pip install flash-attn==2.8.2 --no-build-isolation 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Flash Attention 2.8.2 installed successfully"
        else
            echo -e "  ${YELLOW}⚠${NC} Failed to install Flash Attention 2.8.2"
            echo "  Training will work but may be slower without Flash Attention"
        fi
    fi
else
    echo "  Flash Attention not installed, attempting installation..."
    # Install specific version compatible with xformers
    if pip install flash-attn==2.8.2 --no-build-isolation 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Flash Attention 2.8.2 installed successfully"
    else
        echo -e "  ${YELLOW}⚠${NC} Flash Attention installation failed (optional - training will still work)"
        echo "  You can try: pip install flash-attn==2.8.2 --no-build-isolation"
    fi
fi

# Install ai-toolkit if we have a requirements.txt
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing project requirements..."
    pip install --no-cache-dir -r requirements.txt
fi

# Install ai-toolkit dependencies if ai-toolkit exists
echo ""
if [ -d "/workspace/ai-toolkit" ]; then
    echo "Installing ai-toolkit dependencies..."
    cd /workspace/ai-toolkit
    if [ -f "requirements.txt" ]; then
        pip install --no-cache-dir -r requirements.txt
    fi
    cd "$WORK_DIR"
fi

# Final validation check
echo ""
echo "==================================================="
echo -e "${BLUE}Final Validation${NC}"
echo "==================================================="

# Check CUDA availability in PyTorch
echo ""
echo "PyTorch CUDA Status:"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "  ${GREEN}✓${NC} CUDA is available"
    echo "  PyTorch version: $TORCH_VERSION"
    echo "  CUDA version: $CUDA_VERSION"
    
    # Show GPU info from PyTorch
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null
    python -c "import torch; print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null
else
    echo -e "  ${RED}✗${NC} CUDA is NOT available in PyTorch"
    echo "  This will prevent GPU training!"
fi

# Quick final package check
echo ""
echo "Core package status:"
CORE_PACKAGES=("torch" "transformers" "diffusers" "accelerate" "bitsandbytes")
ALL_GOOD=true
for pkg in "${CORE_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo ""
    echo -e "${RED}ERROR: Some core packages are still missing!${NC}"
    echo "Please check the installation logs above for errors."
    exit 1
fi

# Hugging Face login reminder
echo ""
echo "==================================================="
echo -e "${BLUE}Hugging Face Authentication${NC}"
echo "==================================================="
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo -e "${GREEN}✓${NC} Hugging Face token found"
else
    echo -e "${YELLOW}⚠${NC} No Hugging Face token found"
    echo "To access HiDream models, you need to login:"
    echo "Run: huggingface-cli login"
    echo "Get your token from: https://huggingface.co/settings/tokens"
fi

# Create necessary directories
echo ""
echo "Setting up project directories..."
mkdir -p dataset
mkdir -p input/dataset
mkdir -p output/hidream_i1_finetune
mkdir -p config

# Check for existing scripts
if [ -f "train_hidream.py" ]; then
    echo -e "${GREEN}✓${NC} train_hidream.py found"
else
    echo -e "${YELLOW}⚠${NC} train_hidream.py not found"
fi

if [ -f "prepare_dataset.py" ]; then
    echo -e "${GREEN}✓${NC} prepare_dataset.py found"
else
    echo -e "${YELLOW}⚠${NC} prepare_dataset.py not found"
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
echo -e "${GREEN}Setup Complete!${NC}"
echo "==================================================="
echo ""

# Installation summary
echo -e "${BLUE}Installation Summary:${NC}"
echo "  System packages installed: ${#MISSING_SYSTEM_PACKAGES[@]}"
echo "  Python packages installed: ${#MISSING_PYTHON_PACKAGES[@]}"
if [ "$NEEDS_PYTORCH_UPDATE" = true ]; then
    echo "  PyTorch: Updated"
fi

# GPU-specific recommendations
if [ "$IS_L40S" = true ]; then
    echo ""
    echo -e "${BLUE}L40S GPU Configuration:${NC}"
    echo "  - 48GB VRAM available"
    echo "  - Optimized for batch_size=4"
    echo "  - Full HiDream-I1-Dev model enabled"
    echo "  - Extended training to 3000 steps"
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
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