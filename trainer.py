"""
TinyGPT2 Model Training Script

This script trains a small GPT-2 model (TinyGPT2) on the WikiText-103 dataset.
It handles data processing, model training, and visualization of training metrics.
"""

import os
import math
import torch
import pynvml
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

from model import TinyGPT2Config, TinyGPT2ForCausalLM, get_tokenizer, get_config


def initialize_gpu_monitoring():
    """Initialize NVIDIA GPU monitoring tools."""
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def load_and_process_dataset(tokenizer, block_size=128):
    """
    Load and process the WikiText dataset for language modeling.
    
    Args:
        tokenizer: Tokenizer to use for text processing
        block_size: Maximum sequence length for model input
        
    Returns:
        Processed dataset ready for training
    """
    dataset = load_dataset("wikitext", name="wikitext-103-v1")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        """Tokenize text examples with truncation."""
        return tokenizer(examples["text"], truncation=True, max_length=block_size)
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    
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
        mlm=False,  # Use causal language modeling (not masked)
    )
    
    return lm_datasets, data_collator


def setup_model():
    """
    Initialize the TinyGPT2 model with the configuration.
    
    Returns:
        Configured TinyGPT2 model for causal language modeling
    """
    config = TinyGPT2Config(**get_config())
    return TinyGPT2ForCausalLM(config)


def get_training_args(model_path):
    """
    Configure training arguments for the Trainer.
    
    Args:
        model_path: Path to save the trained model
        
    Returns:
        TrainingArguments object with specified parameters
    """
    return TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_dir=f"{model_path}/logs",
        logging_steps=50,
        save_total_limit=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=1e-1,   
        learning_rate=2e-4,
        num_train_epochs=5,  
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,  # Simulate larger batch size (effective batch size = 8 * 4 = 32)
        warmup_steps=500, 
    )


class LossTrackerCallback(TrainerCallback):
    """
    Custom callback to track training metrics and GPU usage during training.
    
    Tracks:
    - Training loss
    - Evaluation loss
    - Perplexity
    - GPU memory usage
    """
    
    def __init__(self, gpu_handle):
        """
        Initialize the callback with GPU handle for memory monitoring.
        
        Args:
            gpu_handle: NVIDIA GPU device handle for memory tracking
        """
        self.train_losses = []
        self.eval_losses = []
        self.eval_perplexities = []
        self.gpu_usages = []
        self.gpu_handle = gpu_handle

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Process and store logs during training.
        
        Args:
            args: Training arguments
            state: Current training state
            control: Training control object
            logs: Dictionary of current logs
        """
        if logs:
            if "loss" in logs:
                self.train_losses.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                self.eval_losses.append((state.epoch, eval_loss))
                self.eval_perplexities.append((state.epoch, math.exp(eval_loss)))

        # Track GPU memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.gpu_usages.append((state.global_step, mem_info.used / 1024**2))  # in MB


def plot_training_metrics(loss_tracker, save_path="training_metrics.png"):
    """
    Plot training loss and validation perplexity.
    
    Args:
        loss_tracker: LossTrackerCallback with collected metrics
        save_path: Path to save the generated plot
    """
    train_steps, train_losses = zip(*loss_tracker.train_losses)
    eval_epochs_ppl, perplexities = zip(*loss_tracker.eval_perplexities)

    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)

    # Plot Perplexity
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs_ppl, perplexities, label="Eval Perplexity", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_gpu_usage(loss_tracker, save_path="gpu_usage_curve.png"):
    """
    Plot GPU memory usage during training.
    
    Args:
        loss_tracker: LossTrackerCallback with collected metrics
        save_path: Path to save the generated plot
    """
    peak_gpu_usage = max(usage for _, usage in loss_tracker.gpu_usages)
    print(f"\nPeak GPU Memory Usage: {peak_gpu_usage:.2f} MB\n")
    
    steps, usages = zip(*loss_tracker.gpu_usages)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, usages)
    plt.xlabel("Training Steps")
    plt.ylabel("GPU Usage (MB)")
    plt.title("GPU Memory Usage During Training")
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def train_model(model_path='tiny_gpt2_pretrained'):
    """
    Main function to train the TinyGPT2 model.
    
    Args:
        model_path: Directory to save the trained model
        
    Returns:
        Trained model, evaluation results, and tracking metrics
    """
    # Initialize GPU monitoring
    gpu_handle = initialize_gpu_monitoring()
    
    # Set up tokenizer and dataset
    tokenizer = get_tokenizer()
    lm_datasets, data_collator = load_and_process_dataset(tokenizer)
    
    # Initialize model
    model = setup_model()
    
    # Configure training arguments
    training_args = get_training_args(model_path)
    
    # Set up metrics tracker
    loss_tracker = LossTrackerCallback(gpu_handle)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        callbacks=[loss_tracker],
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    return model, eval_results, loss_tracker


if __name__ == "__main__":
    # Train the model
    model_path = 'tiny_gpt2_pretrained'
    model, eval_results, loss_tracker = train_model(model_path)
    
    # Generate and save plots
    plot_training_metrics(loss_tracker)
    plot_gpu_usage(loss_tracker)