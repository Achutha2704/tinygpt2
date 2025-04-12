"""
TinyGPT2 QLoRA Fine-tuning Script

This script implements Quantized Low-Rank Adaptation (QLoRA) fine-tuning
for a pre-trained TinyGPT2 language model on the WikiText-2 dataset.
It handles data processing, model quantization, LoRA adapter setup, and training.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the model components
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

# Import PEFT components
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import model components 
from model import TinyGPT2ForCausalLM, TinyGPT2Config, get_config, get_tokenizer


def load_and_prepare_data(batch_size=4, max_length=128):
    """
    Load and prepare the WikiText-2 dataset for language modeling.
    
    Args:
        batch_size: Number of samples per batch
        max_length: Maximum sequence length for model input
        
    Returns:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
    """
    print("Loading and preparing dataset...")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", name="wikitext-2-v1")
    
    def tokenize_function(examples):
        """Tokenize text examples with truncation."""
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    block_size = max_length

    def group_texts(examples):
        """
        Group tokenized texts into blocks of specified size.
        Creates input_ids and labels for causal language modeling.
        """
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling, not masked
    )
    
    train_dataloader = DataLoader(
        lm_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    val_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    return train_dataloader, val_dataloader


def load_and_prepare_model(model_path='tiny_gpt2_pretrained'):
    """
    Load a pre-trained TinyGPT2 model and set up QLoRA adapters.
    
    Args:
        model_path: Path to the pre-trained model
        
    Returns:
        peft_model: Quantized model with LoRA adapters attached
    """
    print("Setting up QLoRA model...")
    
    # Define quantization configuration using BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load the base model configuration
    config = TinyGPT2Config(**get_config())
    
    # Load model with quantization
    model = TinyGPT2ForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,  # rank dimension
        lora_alpha=16,  # scaling factor (typically 2*r)
        target_modules=["c_attn", "c_proj", "c_fc"],  # Using actual module names from the model
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA adapters to the model
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters to verify PEFT setup
    peft_model.print_trainable_parameters()
    
    return peft_model


def evaluate(model, val_dataloader, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The model to evaluate
        val_dataloader: DataLoader for validation data
        device: Device to run evaluation on (cuda/cpu)
        
    Returns:
        avg_loss: Average loss on validation set
        perplexity: Perplexity metric (exp(avg_loss))
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, device, save_path):
    """
    Train the model using QLoRA fine-tuning.
    
    Args:
        model: Model with LoRA adapters to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for parameter updates
        num_epochs: Number of training epochs
        device: Device to run training on (cuda/cpu)
        save_path: Directory to save adapter weights and plots
        
    Returns:
        model: Trained model
    """
    print(f"Starting training on {device}...")
    
    # Initialize tracking lists
    train_losses = []
    eval_losses = []
    eval_perplexities = []
    
    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move data to device
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass through the PEFT model
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            
            # Backward pass and optimization
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update parameters
            optimizer.zero_grad()  # Reset gradients
            
            # Track metrics
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            
            # Update progress bar with current loss
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Free up memory
            del outputs, loss
            torch.cuda.empty_cache()
        
        # Evaluate at the end of each epoch
        val_loss, val_perplexity = evaluate(model, val_dataloader, device)
        eval_losses.append(val_loss)
        eval_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1} - Avg Train Loss: {sum(epoch_losses)/len(epoch_losses):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
        
        # Save adapter weights after each epoch
        adapter_save_path = os.path.join(save_path, f"epoch_{epoch+1}")
        model.save_pretrained(adapter_save_path)
        print(f"Adapters saved to {adapter_save_path}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, eval_losses, eval_perplexities, save_path)
    
    # Save final adapter weights
    final_save_path = os.path.join(save_path, "final")
    model.save_pretrained(final_save_path)
    print(f"Final adapters saved to {final_save_path}")
    
    return model


def plot_training_metrics(train_losses, eval_losses, eval_perplexities, save_path):
    """
    Plot and save training metrics.
    
    Args:
        train_losses: List of training losses
        eval_losses: List of evaluation losses
        eval_perplexities: List of perplexity values
        save_path: Directory to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Training loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_losses)), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    # Evaluation loss
    plt.subplot(2, 2, 2)
    plt.plot(range(len(eval_losses)), eval_losses, 'o-')
    plt.title('Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Perplexity
    plt.subplot(2, 2, 3)
    plt.plot(range(len(eval_perplexities)), eval_perplexities, 'o-')
    plt.title('Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_metrics.png"))
    plt.close()


def run_qlora_finetuning(
    model_path='tiny_gpt2_pretrained',
    adapter_save_path='tiny_gpt2_qlora',
    batch_size=4, 
    learning_rate=1e-4,
    num_epochs=10
):
    """
    Run the complete QLoRA fine-tuning pipeline.
    
    Args:
        model_path: Path to the pre-trained model
        adapter_save_path: Directory to save adapter weights
        batch_size: Number of samples per batch
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        
    Returns:
        peft_model: Fine-tuned model
        final_perplexity: Final perplexity on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare dataset
    train_dataloader, val_dataloader = load_and_prepare_data(batch_size=batch_size)
    
    # Load and prepare model
    peft_model = load_and_prepare_model(model_path=model_path)
    peft_model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    
    # Train model
    peft_model = train(
        model=peft_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path=adapter_save_path
    )
    
    # Perform final evaluation
    final_loss, final_perplexity = evaluate(peft_model, val_dataloader, device)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Perplexity: {final_perplexity:.2f}")
    
    print("QLoRA fine-tuning completed!")
    
    return peft_model, final_perplexity


if __name__ == "__main__":
    # Configuration
    model_path = 'tiny_gpt2_pretrained'  
    adapter_save_path = 'tiny_gpt2_qlora'  
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 10
    
    # Run QLoRA fine-tuning
    model, perplexity = run_qlora_finetuning(
        model_path=model_path,
        adapter_save_path=adapter_save_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )