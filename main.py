#!/usr/bin/env python3
"""
TinyGPT2 - A lightweight implementation of GPT-2 with training, fine-tuning, and generation capabilities

This script serves as the main entry point, parsing arguments and running the appropriate
modules based on user commands.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Parse arguments and run the appropriate module"""
    parser = argparse.ArgumentParser(
        description="TinyGPT2 - A lightweight implementation of GPT-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new TinyGPT2 model from scratch
  python main.py train --output-dir ./tiny_gpt2_pretrained --epochs 5 --batch-size 8
        
  # Fine-tune a pre-trained model using QLoRA
  python main.py finetune --model-path ./tiny_gpt2_pretrained --output-dir ./tiny_gpt2_qlora --epochs 10
        
  # Generate text using pre-trained and fine-tuned models
  python main.py generate --base-model-path ./tiny_gpt2_pretrained --adapter-path ./tiny_gpt2_qlora/final
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a TinyGPT2 model from scratch")
    train_parser.add_argument("--output-dir", default="tiny_gpt2_pretrained", 
                              help="Directory to save the trained model")
    train_parser.add_argument("--epochs", type=int, default=5,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8,
                              help="Training batch size per device")
    train_parser.add_argument("--gradient-accumulation", type=int, default=4,
                              help="Number of gradient accumulation steps")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4,
                              help="Learning rate for training")
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a pre-trained model using QLoRA")
    finetune_parser.add_argument("--model-path", default="tiny_gpt2_pretrained",
                                 help="Path to the pre-trained model")
    finetune_parser.add_argument("--output-dir", default="tiny_gpt2_qlora",
                                 help="Directory to save the QLoRA adapter weights")
    finetune_parser.add_argument("--epochs", type=int, default=10,
                                 help="Number of fine-tuning epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=4,
                                 help="Fine-tuning batch size")
    finetune_parser.add_argument("--learning-rate", type=float, default=1e-4,
                                 help="Learning rate for fine-tuning")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text using trained models")
    generate_parser.add_argument("--base-model-path", default="tiny_gpt2_pretrained",
                                 help="Path to the base pre-trained model")
    generate_parser.add_argument("--adapter-path", default=None,
                                 help="Path to the QLoRA adapter weights (optional)")
    generate_parser.add_argument("--prompts", nargs="+", 
                                 default=["The quick brown fox", "Once upon a time"],
                                 help="List of prompts for text generation")
    generate_parser.add_argument("--max-length", type=int, default=128,
                                 help="Maximum length of generated text")
    generate_parser.add_argument("--temperature", type=float, default=0.9,
                                 help="Sampling temperature (higher = more random)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    # Execute the appropriate module based on the command
    if args.command == "train":
        from trainer import train_model
        
        print(f"Training TinyGPT2 model with {args.epochs} epochs and batch size {args.batch_size}")
        os.environ["TRAINING_ARGS"] = f"""{{
            "num_train_epochs": {args.epochs},
            "per_device_train_batch_size": {args.batch_size},
            "gradient_accumulation_steps": {args.gradient_accumulation},
            "learning_rate": {args.learning_rate}
        }}"""
        
        train_model(model_path=args.output_dir)
        
    elif args.command == "finetune":
        from finetune import run_qlora_finetuning
        
        print(f"Fine-tuning model at {args.model_path} using QLoRA")
        model, perplexity = run_qlora_finetuning(
            model_path=args.model_path,
            adapter_save_path=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs
        )
        print(f"Final perplexity: {perplexity:.2f}")
        
    elif args.command == "generate":
        from generate import load_base_model, load_qlora_model, test_generation
        
        print(f"Generating text from prompts: {args.prompts}")
        
        # Check if the model paths exist
        base_model_path = Path(args.base_model_path)
        if not base_model_path.exists():
            print(f"Error: Base model path {args.base_model_path} does not exist.")
            return
            
        # Load base model
        base_model, tokenizer = load_base_model(args.base_model_path)
        test_generation(
            base_model, 
            tokenizer, 
            args.prompts, 
            model_name="Base Pre-trained Model",
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # If adapter path is provided, also test with fine-tuned model
        if args.adapter_path:
            adapter_path = Path(args.adapter_path)
            if not adapter_path.exists():
                print(f"Warning: Adapter path {args.adapter_path} does not exist.")
            else:
                qlora_model, _ = load_qlora_model(args.base_model_path, args.adapter_path)
                test_generation(
                    qlora_model, 
                    tokenizer, 
                    args.prompts, 
                    model_name="QLoRA Fine-tuned Model",
                    max_length=args.max_length,
                    temperature=args.temperature
                )

if __name__ == "__main__":
    main()