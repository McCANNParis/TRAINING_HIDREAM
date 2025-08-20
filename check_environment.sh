#!/bin/bash

# HiDream Training Environment Check Script
# This script verifies all requirements for HiDream-I1 finetuning

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HiDream-I1 Environment Configuration Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track overall status
OVERALL_STATUS=0

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $2 found: $(command -v $1)"
        return 0
    else
        echo -e "${RED}✗${NC} $2 not found"
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__ if hasattr($1, '__version__') else 'installed')" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓${NC} $2: $VERSION"
        return 0
    else
        echo -e "${RED}✗${NC} $2 not installed"
        return 1
    fi
}

# 1. System Information
echo -e "${YELLOW}1. System Information${NC}"
echo "   OS: $(uname -s) $(uname -r)"
echo "   Hostname: $(hostname)"
echo "   Date: $(date)"
echo ""

# 2. GPU and CUDA Check
echo -e "${YELLOW}2. GPU and CUDA Configuration${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA Driver detected"
    
    # Get GPU info
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unable to query")
    echo "   GPU: $GPU_INFO"
    
    # Check for L40S specifically
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40S"; then
        echo -e "${GREEN}✓${NC} NVIDIA L40S GPU detected (48GB VRAM) - Optimal configuration available!"
    fi
    
    # CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}✓${NC} CUDA Version: $CUDA_VERSION"
        
        # Check if CUDA 12.4 (optimal for HiDream)
        if [[ "$CUDA_VERSION" == "12.4"* ]]; then
            echo -e "${GREEN}✓${NC} CUDA 12.4 detected - Optimal for HiDream-I1"
        else
            echo -e "${YELLOW}⚠${NC}  CUDA $CUDA_VERSION detected - CUDA 12.4 recommended"
        fi
    else
        echo -e "${YELLOW}⚠${NC}  CUDA compiler (nvcc) not found in PATH"
    fi
    
    # Check VRAM
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
    if [ "$VRAM_MB" -gt 0 ]; then
        VRAM_GB=$((VRAM_MB / 1024))
        echo "   Total VRAM: ${VRAM_GB}GB"
        
        # Recommend configuration based on VRAM
        if [ "$VRAM_GB" -lt 16 ]; then
            echo -e "${YELLOW}⚠${NC}  Low VRAM - Will use HiDream-I1-Fast model"
        elif [ "$VRAM_GB" -lt 24 ]; then
            echo -e "${GREEN}✓${NC} Sufficient VRAM for HiDream-I1-Fast model"
        elif [ "$VRAM_GB" -lt 40 ]; then
            echo -e "${GREEN}✓${NC} Good VRAM for HiDream-I1-Dev model"
        else
            echo -e "${GREEN}✓${NC} Excellent VRAM for full HiDream-I1-Dev model with larger batch sizes"
        fi
    fi
else
    echo -e "${RED}✗${NC} NVIDIA driver not found - GPU training not available"
    OVERALL_STATUS=1
fi
echo ""

# 3. Python Environment
echo -e "${YELLOW}3. Python Environment${NC}"

if check_command python "Python"; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "   Version: $PYTHON_VERSION"
    
    # Check if Python 3.10 or 3.11 (recommended)
    if [[ "$PYTHON_VERSION" == "3.10"* ]] || [[ "$PYTHON_VERSION" == "3.11"* ]]; then
        echo -e "${GREEN}✓${NC} Python version optimal for HiDream"
    else
        echo -e "${YELLOW}⚠${NC}  Python 3.10 or 3.11 recommended"
    fi
else
    echo -e "${RED}✗${NC} Python not found"
    OVERALL_STATUS=1
fi

# Check virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment active: $VIRTUAL_ENV"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}✓${NC} Conda environment active: $CONDA_DEFAULT_ENV"
else
    echo -e "${YELLOW}⚠${NC}  No virtual environment detected"
fi
echo ""

# 4. Essential Python Packages
echo -e "${YELLOW}4. Essential Python Packages${NC}"

# PyTorch and related
check_python_package "torch" "PyTorch" || OVERALL_STATUS=1
check_python_package "torchvision" "TorchVision" || OVERALL_STATUS=1

# Check PyTorch CUDA availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyTorch CUDA support enabled"
    TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "   PyTorch CUDA version: $TORCH_CUDA_VERSION"
else
    echo -e "${YELLOW}⚠${NC}  PyTorch CUDA support not available"
fi

# AI Toolkit and dependencies
check_python_package "transformers" "Transformers" || OVERALL_STATUS=1
check_python_package "diffusers" "Diffusers" || OVERALL_STATUS=1
check_python_package "accelerate" "Accelerate" || OVERALL_STATUS=1
check_python_package "safetensors" "SafeTensors" || OVERALL_STATUS=1
check_python_package "peft" "PEFT (LoRA support)" || OVERALL_STATUS=1

# Memory optimization
check_python_package "xformers" "XFormers (memory optimization)" || echo -e "${YELLOW}⚠${NC}  XFormers recommended for memory efficiency"
check_python_package "bitsandbytes" "BitsAndBytes (8-bit optimizer)" || echo -e "${YELLOW}⚠${NC}  BitsAndBytes recommended for optimizer efficiency"

# Monitoring tools
check_python_package "tensorboard" "TensorBoard" || echo -e "${YELLOW}⚠${NC}  TensorBoard recommended for monitoring"
check_python_package "wandb" "Weights & Biases" || echo -e "${YELLOW}⚠${NC}  WandB optional for advanced monitoring"

echo ""

# 5. HuggingFace Authentication
echo -e "${YELLOW}5. HuggingFace Authentication${NC}"

if [ -f ~/.cache/huggingface/token ] || [ -f ~/.huggingface/token ]; then
    echo -e "${GREEN}✓${NC} HuggingFace token found"
    
    # Try to verify token validity
    if python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami()['name'])" 2>/dev/null; then
        USERNAME=$(python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami()['name'])" 2>/dev/null)
        echo -e "${GREEN}✓${NC} Authenticated as: $USERNAME"
    else
        echo -e "${YELLOW}⚠${NC}  Token found but validation failed"
    fi
else
    echo -e "${YELLOW}⚠${NC}  HuggingFace token not found"
    echo "   Run: huggingface-cli login"
fi
echo ""

# 6. Project Structure
echo -e "${YELLOW}6. Project Structure${NC}"

# Check required directories
DIRS=("input" "input/dataset" "output" "config")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} Directory exists: $dir/"
    else
        echo -e "${YELLOW}⚠${NC}  Directory missing: $dir/ (will be created when needed)"
    fi
done

# Check required scripts
SCRIPTS=("train_hidream.py" "prepare_dataset.py" "runpod_setup.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "${GREEN}✓${NC} Script found: $script"
    else
        echo -e "${RED}✗${NC} Script missing: $script"
        OVERALL_STATUS=1
    fi
done

# Check config file
if [ -f "config/hidream_i1_finetune.yaml" ]; then
    echo -e "${GREEN}✓${NC} Configuration file found"
else
    echo -e "${YELLOW}⚠${NC}  Configuration file not found: config/hidream_i1_finetune.yaml"
fi
echo ""

# 7. Git Repository
echo -e "${YELLOW}7. Git Repository${NC}"

if [ -d .git ]; then
    echo -e "${GREEN}✓${NC} Git repository initialized"
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "   Current branch: $BRANCH"
    
    # Check for uncommitted changes
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠${NC}  Uncommitted changes detected"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Not a git repository"
fi
echo ""

# 8. Memory and Disk Space
echo -e "${YELLOW}8. System Resources${NC}"

# Memory
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
    echo "   Total RAM: $TOTAL_MEM"
    echo "   Available RAM: $AVAIL_MEM"
fi

# Disk space
DISK_USAGE=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available disk space: $DISK_USAGE"

# Check if enough space for training (recommend at least 50GB)
DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_GB" -lt 50 ]; then
    echo -e "${YELLOW}⚠${NC}  Low disk space - at least 50GB recommended for training"
fi
echo ""

# 9. AI Toolkit Check
echo -e "${YELLOW}9. AI Toolkit Installation${NC}"

# Check if ai-toolkit is installed via pip
if pip show ai-toolkit &>/dev/null; then
    echo -e "${GREEN}✓${NC} ai-toolkit installed via pip"
elif [ -d "ai-toolkit" ]; then
    echo -e "${GREEN}✓${NC} ai-toolkit directory found"
else
    echo -e "${YELLOW}⚠${NC}  ai-toolkit not detected - may need installation"
    echo "   Run: ./runpod_setup.sh to install"
fi
echo ""

# Final Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Environment Check Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Environment is ready for HiDream-I1 training!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Prepare your dataset: python prepare_dataset.py /path/to/images --output input/dataset"
    echo "2. Validate setup: python train_hidream.py --validate-only"
    echo "3. Start training: python train_hidream.py --auto-optimize"
else
    echo -e "${RED}✗ Some requirements are missing${NC}"
    echo ""
    echo "To fix missing dependencies:"
    echo "1. Run: ./runpod_setup.sh"
    echo "2. Activate virtual environment if needed"
    echo "3. Run this check again: ./check_environment.sh"
fi

exit $OVERALL_STATUS