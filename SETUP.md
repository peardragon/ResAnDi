# ResAnDi Setup Guide

## Prerequisites

1. **Python 3.8 or higher**
2. **UV package manager** (recommended) or pip
3. **CUDA-capable GPU** (optional, for faster training)

## Installation with UV (Recommended)

UV is a fast Python package installer and resolver. If you don't have UV installed:

```bash
pip install uv
```

### Option 1: Using pyproject.toml (Recommended)

```bash
# Navigate to the ResAnDi directory
cd /path/to/ResAnDi

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# On Windows: .venv\Scripts\activate

# Verify activation (optional - should show (.venv) in prompt)
which python  # Should show: /path/to/ResAnDi/.venv/bin/python

# Install the project in development mode
uv pip install -e .
```

### Option 2: Using requirements.txt

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from requirements file
uv pip install -r requirements.txt
```

## Installation with Pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## CUDA Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

Test your installation by running:

```python
import torch
import numpy as np
from utils._resnet import resnet18_8

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model creation
model = resnet18_8()
print("ResNet model created successfully!")
```

## Directory Setup

Ensure the following directories exist (they will be created automatically by the notebooks):

```bash
mkdir -p dataset_noiseless/{train,test,eval,train_1000,test_1000,eval_1000}
mkdir -p Grad-CAM
mkdir -p model_ckpt
mkdir -p saved_models
mkdir -p analysis_results/{model_results,statistical_relevance,erasing_method,augmentation_results,feature_ablation,tsne_results,statistical_relevance_block}
mkdir -p figures
```

## VS Code Jupyter Notebook Setup

To run Jupyter notebooks (.ipynb files) directly in VS Code like an IDE:

### 1. Install Required VS Code Extensions

Install the following extensions in VS Code:
- **Python** (ms-python.python)
- **Jupyter** (ms-toolsai.jupyter)


### 2. Select Kernel in VS Code

1. Open any `.ipynb` file in VS Code
2. Click on the **kernel selector** in the top-right corner of the notebook
3. Choose **"ResAnDi"** from the list
4. VS Code will now use your project environment for all notebook cells

### 3. Configure Jupyter Kernel (Manual)

After activating your virtual environment, register it as a Jupyter kernel:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # UV environment
# OR
source venv/bin/activate   # Standard venv

# Install ipykernel in your environment (should already be installed via requirements.txt)
pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name ResAnDi --display-name "ResAnDi"
```