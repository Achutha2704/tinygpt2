# TinyGPT2

A lightweight implementation of the GPT-2 language model with training, QLoRA fine-tuning, and text generation capabilities.

## Overview

TinyGPT2 is a compact GPT-2-style language model designed for research and experimentation with transformer-based architectures. The project includes:

- A custom TinyGPT2 model implementation
- Training pipeline on the WikiText dataset
- QLoRA (Quantized Low-Rank Adaptation) fine-tuning
- Text generation utilities

The model architecture follows the principles from the GPT-2 paper but with a smaller parameter footprint, making it more accessible for experimentation on consumer hardware.

## Repository Structure

- `model.py`: Core model architecture implementation (TinyGPT2)
- `trainer.py`: Model training script with WikiText dataset
- `finetune.py`: QLoRA fine-tuning implementation
- `generate.py`: Text generation utilities
- `main.py`: Command-line interface for all functionality

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Achutha2704/tinygpt2.git
   cd tinygpt2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

TinyGPT2 provides a simple command-line interface for all major operations.

### Training a New Model

Train a TinyGPT2 model from scratch on the WikiText dataset:

```bash
python main.py train --output-dir ./tiny_gpt2_pretrained --epochs 5 --batch-size 8
```

Options:
- `--output-dir`: Directory to save the trained model
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per device
- `--gradient-accumulation`: Number of gradient accumulation steps
- `--learning-rate`: Learning rate for training

### Fine-tuning with QLoRA

Fine-tune a pre-trained model using QLoRA:

```bash
python main.py finetune --model-path ./tiny_gpt2_pretrained --output-dir ./tiny_gpt2_qlora --epochs 10
```

Options:
- `--model-path`: Path to the pre-trained model
- `--output-dir`: Directory to save the adapter weights
- `--epochs`: Number of fine-tuning epochs
- `--batch-size`: Batch size for fine-tuning
- `--learning-rate`: Learning rate for fine-tuning

### Generating Text

Generate text using the pre-trained model or the fine-tuned QLoRA model:

```bash
python main.py generate --base-model-path ./tiny_gpt2_pretrained --adapter-path ./tiny_gpt2_qlora/final
```

Options:
- `--base-model-path`: Path to the base pre-trained model
- `--adapter-path`: Path to the QLoRA adapter weights (optional)
- `--prompts`: List of prompts to generate from
- `--max-length`: Maximum length of generated text
- `--temperature`: Sampling temperature (higher = more random)

## File Details

### model.py

This file contains the complete implementation of the TinyGPT2 model architecture:
- `TinyGPT2Config`: Configuration class for the model
- `TinyGPT2`: Base implementation of the transformer model
- `TinyGPT2ForCausalLM`: Language modeling head implementation
- Utility functions: `get_config()`, `get_tokenizer()`

The architecture follows the transformer design with:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Transformer decoder blocks
- Support for generation and caching

### trainer.py

The trainer script handles the process of training the TinyGPT2 model from scratch:
- Data loading and processing from the WikiText-103 dataset
- Training loop with metrics tracking
- GPU monitoring and visualization
- Model checkpointing and evaluation

Key features include:
- Integration with the Hugging Face Trainer API
- Memory-efficient training with gradient accumulation
- Visualization of training metrics and GPU usage

### finetune.py

The fine-tuning script implements QLoRA, a parameter-efficient fine-tuning method:
- Loads and quantizes a pre-trained model to 4-bit precision
- Applies Low-Rank Adaptation (LoRA) to specific layers
- Fine-tunes on the WikiText-2 dataset
- Tracks and visualizes training metrics

QLoRA enables fine-tuning large models with minimal memory requirements while maintaining performance.

### generate.py

The generation script provides utilities for text generation:
- Loading the base pre-trained model
- Loading a fine-tuned model with QLoRA adapters
- Generating text from various prompts
- Comparing output from different model versions

## Model Size and Performance

TinyGPT2 has approximately 82M parameters with the following configuration:
- 8 transformer layers
- 8 attention heads
- 512 embedding dimension
- 512 maximum sequence length

This provides a good balance between performance and resource requirements, making it suitable for research and experimentation.