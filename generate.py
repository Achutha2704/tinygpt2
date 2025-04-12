"""
TinyGPT2 Text Generation Script

This script demonstrates text generation using two versions of a TinyGPT2 model:
1. The base pre-trained model
2. The model fine-tuned with QLoRA adapters

It loads both models and generates text from a set of test prompts for comparison.
"""

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Import required modules from your model implementation
from model import get_tokenizer, TinyGPT2ForCausalLM


def load_base_model(model_path):
    """
    Load the base pre-trained TinyGPT2 model.
    
    Args:
        model_path: Path to the pre-trained model
        
    Returns:
        model: Loaded model
        tokenizer: Corresponding tokenizer
    """
    print(f"Loading base model from {model_path}...")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = TinyGPT2ForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU")
    else:
        print("Running on CPU")
        
    return model, tokenizer


def load_qlora_model(base_model_path, adapter_path):
    """
    Load the TinyGPT2 model with QLoRA adapters.
    
    Args:
        base_model_path: Path to the base pre-trained model
        adapter_path: Path to the QLoRA adapter weights
        
    Returns:
        model: Model with adapters
        tokenizer: Corresponding tokenizer
    """
    print(f"Loading QLoRA model with adapter from {adapter_path}...")
    
    # Load the adapter config
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    
    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=128, temperature=0.9, top_p=0.9):
    """
    Generate text from a prompt using the provided model.
    
    Args:
        model: Language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        
    Returns:
        generated_text: The complete generated text including the prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


def test_generation(model, tokenizer, prompts, model_name="Model"):
    """
    Test text generation with a set of prompts.
    
    Args:
        model: Language model to use
        tokenizer: Tokenizer corresponding to the model
        prompts: List of text prompts to test
        model_name: Name to display for the model
    """
    print(f"\n----- Generated Text Examples ({model_name}) -----")
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")


def compare_models(base_model_path, qlora_adapter_path, test_prompts):
    """
    Compare text generation between base model and QLoRA-enhanced model.
    
    Args:
        base_model_path: Path to the base pre-trained model
        qlora_adapter_path: Path to the QLoRA adapter weights
        test_prompts: List of prompts to test with both models
    """
    # Load base model
    base_model, base_tokenizer = load_base_model(base_model_path)
    
    # Test base model
    test_generation(base_model, base_tokenizer, test_prompts, "Base Pre-trained Model")
    
    # Load QLoRA model
    qlora_model, qlora_tokenizer = load_qlora_model(base_model_path, qlora_adapter_path)
    
    # Test QLoRA model
    test_generation(qlora_model, qlora_tokenizer, test_prompts, "QLoRA Fine-tuned Model")

# Add these parameter options to the generate_text function in generate.py

def generate_text(model, tokenizer, prompt, max_length=128, temperature=0.9, top_p=0.9):
    """
    Generate text from a prompt using the provided model.
    
    Args:
        model: Language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        
    Returns:
        generated_text: The complete generated text including the prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


def test_generation(model, tokenizer, prompts, model_name="Model", max_length=128, temperature=0.9):
    """
    Test text generation with a set of prompts.
    
    Args:
        model: Language model to use
        tokenizer: Tokenizer corresponding to the model
        prompts: List of text prompts to test
        model_name: Name to display for the model
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
    """
    print(f"\n----- Generated Text Examples ({model_name}) -----")
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=max_length, temperature=temperature)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")

if __name__ == "__main__":
    # Define model paths
    pretrained_path = '/kaggle/input/nlp-assignment2-tinygpt/tiny_gpt2_pretrained'
    qlora_path = "/kaggle/input/nlp-assignment2-tinygpt/tiny_gpt2_qlora/final"
    
    # Define test prompts
    test_prompts = [
        "The quick brown fox",
        "In a galaxy far far away",
        "Once upon a time in",
    ]
    
    # Compare text generation between models
    compare_models(pretrained_path, qlora_path, test_prompts)
    
    # If you want to test individual models, uncomment these sections:
    """
    # Test just the base model
    base_model, base_tokenizer = load_base_model(pretrained_path)
    test_generation(base_model, base_tokenizer, test_prompts, "Base Model")
    
    # Test just the QLoRA model
    qlora_model, qlora_tokenizer = load_qlora_model(pretrained_path, qlora_path)
    test_generation(qlora_model, qlora_tokenizer, test_prompts, "QLoRA Model")
    """