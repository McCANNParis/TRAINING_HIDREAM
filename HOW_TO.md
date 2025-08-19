# How to Fine-tune HiDream-I1 Model

This guide walks you through the complete process of fine-tuning the HiDream-I1 17B parameter text-to-image model using this toolkit.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training Process](#training-process)
- [RunPod Deployment](#runpod-deployment)
- [Monitoring & Evaluation](#monitoring--evaluation)
- [Using Your Trained Model](#using-your-trained-model)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM
  - Recommended: NVIDIA L40S (48GB) for optimal performance
  - Minimum: RTX 3090/4090 (24GB) for basic training
- **RAM**: 32GB system RAM minimum
- **Storage**: 100GB free space for models and datasets
- **OS**: Linux (Ubuntu 20.04+) or RunPod cloud environment

### Software Requirements
- Python 3.10 or 3.11
- CUDA 12.1+ with cuDNN
- Git and Git LFS
- Hugging Face account (for model access)

## Quick Start

For experienced users who want to get started immediately:

```bash
# 1. Clone the repository
git clone https://github.com/McCANNParis/TRAINING_HIDREAM.git
cd TRAINING_HIDREAM

# 2. Set up environment (RunPod)
chmod +x runpod_setup.sh
./runpod_setup.sh

# 3. Authenticate with Hugging Face
huggingface-cli login

# 4. Your dataset is already prepared in the 'dataset' folder
# with paired images and captions - ready to use!

# 5. Start training with auto-optimization
python train_hidream.py --auto-optimize --dataset-path dataset
```

## Detailed Setup

### Step 1: Environment Setup

#### Option A: RunPod (Recommended)
```bash
# Deploy using the RunPod template
# 1. Go to RunPod.io and create a new pod
# 2. Select NVIDIA L40S GPU
# 3. Use the provided Docker image or run setup script

# Once connected to your pod:
cd /workspace
git clone https://github.com/McCANNParis/TRAINING_HIDREAM.git
cd TRAINING_HIDREAM

# Run automated setup
chmod +x runpod_setup.sh
./runpod_setup.sh
```

#### Option B: Local Setup
```bash
# Clone repository
git clone https://github.com/McCANNParis/TRAINING_HIDREAM.git
cd TRAINING_HIDREAM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Hugging Face Authentication

The HiDream models require authentication:

```bash
# Login to Hugging Face
huggingface-cli login

# Enter your access token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

### Step 3: Validate Installation

Before proceeding, validate your setup:

```bash
python train_hidream.py --validate-only
```

This checks:
- CUDA availability and version
- GPU memory
- Required packages
- Model access permissions

## Dataset Preparation

### Using Your Pre-Prepared Dataset

Your dataset is already prepared and located in the `dataset` folder at the root of the workspace. This folder contains:
- Image files (JPEG/PNG)
- Corresponding caption files (.txt) with the same name as each image

Example structure:
```
dataset/
├── image001.jpg
├── image001.txt  # Caption: "a beautiful landscape painting"
├── image002.jpg
├── image002.txt  # Caption: "portrait in dramatic lighting"
└── ...
```

To use this dataset directly:
```bash
# Training with your existing dataset
python train_hidream.py --auto-optimize --dataset-path dataset

# Or specify in the config file
python train_hidream.py --config config/hidream_i1_finetune.yaml
```

### Optional: Preparing Additional Datasets

If you need to prepare additional images:

```bash
# Process new images and add to existing dataset
python prepare_dataset.py /path/to/new/images --output dataset --append

# With custom trigger word for style consistency
python prepare_dataset.py /path/to/images --output dataset --trigger-word "my_style" --append
```

### Dataset Requirements

- **Image Format**: JPEG, PNG, WebP supported
- **Resolution**: Images will be automatically resized to 1024x1024
- **Quantity**: 
  - Minimum: 10-20 images for basic style transfer
  - Recommended: 50-100 images for robust results
  - Maximum: 500-1000 images (more may overfit)

### Caption Best Practices

1. **Be Descriptive**: Include details about style, composition, colors
2. **Use Trigger Words**: Add a unique identifier for your style
3. **Consistency**: Use similar caption structure across your dataset
4. **Variety**: Include diverse descriptions to improve generalization

Example captions:
- "hidream_style, a serene landscape with mountains and lake"
- "hidream_style, portrait photography with soft lighting"
- "hidream_style, abstract art with vibrant colors"

## Training Process

### Automatic Training (Recommended)

The easiest way to start training with GPU-optimized settings using your prepared dataset:

```bash
# Use the dataset folder in your workspace
python train_hidream.py --auto-optimize --dataset-path dataset
```

This automatically:
- Detects your GPU and available VRAM
- Selects appropriate model variant
- Configures batch size and gradient accumulation
- Sets optimal training steps

### Custom Configuration Training

For fine-grained control, modify `config/hidream_i1_finetune.yaml`:

```yaml
model:
  name_or_path: "TencentARC/HiDream-I1-Dev"  # or HiDream-I1-Fast for less VRAM
  revision: "main"
  is_hidream: true

dataset:
  name_or_path: "dataset"          # Your prepared dataset folder
  caption_column: "text"           # Column name for captions
  
train:
  batch_size: 2                    # Adjust based on VRAM
  gradient_accumulation_steps: 2   # Increase if batch_size is small
  steps: 2000                       # Total training steps
  lr: 1e-4                         # Learning rate
  
lora:
  rank: 16                         # LoRA rank (16 is good balance)
  alpha: 16                        # LoRA alpha (usually same as rank)
```

Then run:
```bash
python train_hidream.py --config config/hidream_i1_finetune.yaml
```

### GPU-Specific Settings

#### NVIDIA L40S (48GB)
```bash
python train_hidream.py --auto-optimize --dataset-path dataset  # Automatically uses optimal L40S settings
# Or manually:
python train_hidream.py --batch-size 4 --steps 3000 --model HiDream-I1-Dev --dataset-path dataset
```

#### RTX 4090/3090 (24GB)
```bash
python train_hidream.py --batch-size 1 --gradient-accumulation 4 --model HiDream-I1-Fast --dataset-path dataset
```

#### Lower VRAM GPUs (16GB)
```bash
python train_hidream.py --batch-size 1 --gradient-accumulation 8 --model HiDream-I1-Fast --use-8bit --dataset-path dataset
```

### Resume Training from Checkpoint

If training is interrupted:

```bash
# Automatically finds latest checkpoint and continues with your dataset
python train_hidream.py --resume --dataset-path dataset

# Or specify checkpoint directory
python train_hidream.py --resume --checkpoint-dir output/hidream_i1_finetune/checkpoint-1000 --dataset-path dataset
```

## RunPod Deployment

### Using the RunPod Template

1. **Deploy Template**:
   ```bash
   # The template is in runpod_template.json
   # Upload this to RunPod or use their API
   ```

2. **Connect to Pod**:
   ```bash
   ssh root@[your-pod-ip] -p [port]
   # Or use RunPod's web terminal
   ```

3. **Run Setup**:
   ```bash
   cd /workspace
   git clone https://github.com/McCANNParis/TRAINING_HIDREAM.git
   cd TRAINING_HIDREAM
   ./runpod_setup.sh
   ```

### Building Custom Docker Image

```bash
# Build the Docker image
docker build -f runpod_dockerfile -t hidream-training .

# Push to your registry
docker tag hidream-training your-registry/hidream-training
docker push your-registry/hidream-training
```

## Monitoring & Evaluation

### TensorBoard

Monitor training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir output/hidream_i1_finetune

# Access at http://localhost:6006
```

Metrics to watch:
- **Loss**: Should decrease steadily
- **Learning Rate**: Check schedule is correct
- **Gradient Norm**: Should remain stable

### Sample Generation

The trainer automatically generates samples during training:

```bash
# View generated samples
ls output/hidream_i1_finetune/samples/

# Samples are saved every 100 steps by default
```

### Weights & Biases (Optional)

Enable W&B logging in config:

```yaml
tracker:
  name: "wandb"
  project: "hidream-finetune"
  run_name: "my-training-run"
```

## Using Your Trained Model

### Finding Your Model

After training completes:

```bash
# Your LoRA weights are saved here:
ls output/hidream_i1_finetune/

# Key files:
# - pytorch_lora_weights.safetensors  (main LoRA weights)
# - samples/                           (generated samples)
# - config.yaml                        (training configuration)
```

### Loading in Diffusers

```python
from diffusers import DiffusionPipeline
import torch

# Load base model
pipe = DiffusionPipeline.from_pretrained(
    "TencentARC/HiDream-I1-Dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

# Load your LoRA weights
pipe.load_lora_weights("output/hidream_i1_finetune")

# Generate images
prompt = "hidream_style, a beautiful sunset over mountains"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("generated_image.png")
```

### Merging LoRA Weights (Optional)

To create a standalone model:

```python
# Merge LoRA weights with base model
pipe.fuse_lora(lora_scale=1.0)

# Save merged model
pipe.save_pretrained("my-hidream-model")
```

## Troubleshooting

### Common Issues and Solutions

#### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python train_hidream.py --batch-size 1 --gradient-accumulation 8 --dataset-path dataset

# Solution 2: Use smaller model
python train_hidream.py --model HiDream-I1-Fast --dataset-path dataset

# Solution 3: Enable 8-bit optimization
python train_hidream.py --use-8bit --dataset-path dataset

# Solution 4: Clear GPU memory
nvidia-smi
# Kill any hanging processes
```

#### Hugging Face Authentication Error
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login

# Check token permissions
# Ensure token has 'read' access to gated models
```

#### Training Loss Not Decreasing
- Check your dataset quality and captions
- Reduce learning rate: `--lr 5e-5`
- Increase training steps: `--steps 3000`
- Ensure images are diverse and high-quality

#### Slow Training Speed
```bash
# Enable optimizations with your dataset
python train_hidream.py --auto-optimize --enable-xformers --dataset-path dataset

# Check GPU utilization
nvidia-smi -l 1  # Monitor GPU usage

# Ensure you're using the right GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

1. **Check Logs**: 
   ```bash
   # Training logs are in:
   cat output/hidream_i1_finetune/training.log
   ```

2. **Validate Setup**:
   ```bash
   python train_hidream.py --validate-only
   ```

3. **Community Support**:
   - Open an issue on [GitHub](https://github.com/McCANNParis/TRAINING_HIDREAM/issues)
   - Check existing issues for solutions

## Advanced Tips

### Multi-GPU Training
```bash
# Use multiple GPUs with your dataset (experimental)
torchrun --nproc_per_node=2 train_hidream.py --auto-optimize --dataset-path dataset
```

### Custom Schedulers
Modify in config:
```yaml
noise_scheduler:
  name: "flowmatch"  # or "ddpm", "euler"
  num_train_timesteps: 1000
```

### Experimenting with LoRA Ranks
- **Lower rank (8)**: Faster training, less expressive
- **Default (16)**: Good balance
- **Higher rank (32)**: More parameters, risk of overfitting

### Working with Your Dataset

```bash
# Check your dataset structure
ls -la dataset/ | head -20

# Count image-caption pairs
echo "Total images: $(ls dataset/*.jpg dataset/*.png 2>/dev/null | wc -l)"
echo "Total captions: $(ls dataset/*.txt 2>/dev/null | wc -l)"

# Verify caption content
for file in dataset/*.txt; do
    echo "$(basename $file): $(cat $file)"
done | head -5
```

## Performance Benchmarks

| GPU | Model Variant | Batch Size | Training Time (1000 steps) |
|-----|--------------|------------|---------------------------|
| L40S (48GB) | HiDream-I1-Dev | 4 | ~2 hours |
| RTX 4090 (24GB) | HiDream-I1-Fast | 2 | ~3 hours |
| RTX 3090 (24GB) | HiDream-I1-Fast | 1 | ~4 hours |
| RTX 3080 (10GB) | Not recommended | - | - |

## Best Practices Summary

1. **Start Simple**: Begin with auto-optimize mode
2. **Quality over Quantity**: 50 high-quality images > 500 poor ones
3. **Monitor Progress**: Use TensorBoard to track training
4. **Save Checkpoints**: Enable frequent checkpointing
5. **Test Regularly**: Generate samples during training
6. **Document Settings**: Save your config for reproducibility

## Conclusion

You're now ready to fine-tune HiDream-I1! Remember:
- Start with `--auto-optimize` for hassle-free training
- Use RunPod with L40S GPUs for best performance
- Monitor your training with TensorBoard
- Experiment with different datasets and settings

Happy training! 🚀