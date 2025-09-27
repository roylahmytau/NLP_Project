#!/usr/bin/env python3
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
import os
from transformers import TrainerCallback
import logging

class LossLoggingCallback(TrainerCallback):
    """Custom callback to log training loss after each step"""
    
    def __init__(self):
        self.step_count = 0
        self.epoch_count = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch_count += 1
        print(f"\n--- Starting Epoch {self.epoch_count} ---")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are created"""
        if logs is not None and 'loss' in logs:
            self.step_count += 1
            # epoch_progress = (state.global_step % (len(state.log_history) // args.num_train_epochs)) if args.num_train_epochs > 1 else state.global_step
            print(f"Step {state.global_step} (Epoch {self.epoch_count}): Training Loss = {logs['loss']:.4f}")
            
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if state.log_history:
            last_log = state.log_history[-1]
            if 'loss' in last_log and 'learning_rate' in last_log:
                print(f"Step {state.global_step}: Loss = {last_log['loss']:.4f}, LR = {last_log['learning_rate']:.2e}")

class SimpleDataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Extract text from features
        texts = [f["text"] for f in features]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Add labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized

def load_qa_data(file_path):
    """Load and format QA data for training"""
    data = []
    with open(file_path, 'r') as f:
        # Parse as JSONL (one JSON object per line)
        data = json.load(f)
        
        print(f"Successfully loaded {len(data)} items from JSON file")

    # Format for Qwen instruction tuning
    formatted_data = []
    for item in data:
        instruction = item['instruction']
        doc = item['doc']
        question = item['question']
        answer = item['outputs'][0] if item['outputs'] else ""
        
        # Create Qwen-compatible instruction format
        text = f"<|im_start|>user\n{instruction}\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        formatted_data.append({"text": text})
    
    return formatted_data


def load_generated_qa_pairs(generated_dir: str):
    """Load generated Q&A text files (record_*.txt) and build few-shot snippets.

    Returns a dict: { index (int) : prompt_snippet (str) }
    where index corresponds to the record index used during generation.
    """
    if not os.path.isdir(generated_dir):
        print(f"Generated QA directory not found: {generated_dir}")
        return {}

    index_to_prompt = {}
    for name in os.listdir(generated_dir):
        if not name.startswith("record_") or not name.endswith(".txt"):
            continue
        stem = name[len("record_"):-len(".txt")]
        try:
            rec_index = int(stem)
        except ValueError:
            continue
        path = os.path.join(generated_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Wrap as a system-style preface the assistant can condition on
            snippet = (
                "Here are related questions and concise answers that pertain to the document.\n"
                + content
                + "\n"
            )
            index_to_prompt[rec_index] = snippet
        except Exception as e:
            print(f"Failed reading {path}: {e}")
    print(f"Loaded {len(index_to_prompt)} generated QA snippets from {generated_dir}")
    return index_to_prompt

def create_lora_model(model_name, adapter_config, use_4bit=True):
    """Create model with LoRA adapter"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization 
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_config['r'],
        lora_alpha=adapter_config['lora_alpha'],
        lora_dropout=adapter_config['lora_dropout'],
        target_modules=adapter_config['target_modules'],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    return model, tokenizer

def get_data_file_path(needle_size, needle_type):
    """Construct the data file path based on needle size and type"""
    valid_sizes = ["2048", "32768", "131072"]
    valid_types = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", 
                   "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"]
    
    if needle_size not in valid_sizes:
        raise ValueError(f"Invalid needle size: {needle_size}. Must be one of {valid_sizes}")
    if needle_type not in valid_types:
        raise ValueError(f"Invalid needle type: {needle_type}. Must be one of {valid_types}")
    
    return f"needles/{needle_size}/{needle_type}_{needle_size}.json"

def main():
    parser = argparse.ArgumentParser()
    # Model selection - using Qwen model
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B", 
                       help="Base model name (Qwen model)")
    parser.add_argument("--needle_size", required=True, choices=["2048", "32768", "131072"],
                       help="Needle size (required): 2048, 32768, or 131072")
    parser.add_argument("--needle_type", required=True, 
                       choices=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", 
                               "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"],
                       help="Needle type (required): niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_single_1, niah_single_2, niah_single_3, or qa_1")
    parser.add_argument("--use_generated_qa", action="store_true", help="Augment prompts with generated Q&A")
    parser.add_argument("--generated_qa_dir", type=str, help="Directory with generated Q&A files (auto-generated if not provided)")
    
    # LoRA configuration
    parser.add_argument("--adapter_r", type=int, default=8,
                       help="LoRA rank (reduced for memory efficiency)")
    parser.add_argument("--adapter_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--adapter_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--adapter_layers", type=int, default=3,
                       help="Number of adapter layers")
    parser.add_argument("--adapter_type", default="linear",
                       help="Adapter type")
    
    # Training configuration
    parser.add_argument("--output_dir", default="adapter",
                       help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data file path from needle size and type
    data_file = get_data_file_path(args.needle_size, args.needle_type)
    print(f"Using data file: {data_file}")
    
    # Auto-generate generated_qa_dir if not provided but use_generated_qa is True
    if args.use_generated_qa and not args.generated_qa_dir:
        args.generated_qa_dir = f"outputs/{args.needle_type}_{args.needle_size}"
        print(f"Auto-generated QA directory: {args.generated_qa_dir}")
    
    # Load data
    print("Loading data...")
    data = load_qa_data(data_file)
    index_to_snippet = load_generated_qa_pairs(args.generated_qa_dir) if args.use_generated_qa else {}
    print(f"Loaded {len(data)} examples")
    
    # Create dataset
    # If using generated QA, augment each formatted training example with the matching snippet
    if args.use_generated_qa and index_to_snippet:
        augmented = []
        for i, item in enumerate(data):
            prefix = index_to_snippet.get(i)
            if prefix:
                augmented_text = (
                    "<|im_start|>system\n"
                    + prefix
                    + "<|im_end|>\n"
                    + item["text"]
                )
                augmented.append({"text": augmented_text})
            else:
                augmented.append(item)
        data = augmented

    dataset = Dataset.from_list(data)
    
    # Configure adapter for Qwen models
    adapter_config = {
        'r': args.adapter_r,
        'lora_alpha': args.adapter_alpha,
        'lora_dropout': args.adapter_dropout,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen uses these module names
    }
    
    # Create model
    print("Creating model...")
    model, tokenizer = create_lora_model(args.model_name, adapter_config, args.use_4bit)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Use raw dataset - tokenization will be handled by data collator
    tokenized_dataset = dataset
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Increased for effective larger batch
        warmup_steps=50,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=1,  # Log after every step
        save_steps=200,
        eval_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Reduce memory usage
        dataloader_num_workers=0,     # Reduce memory usage
        gradient_checkpointing=True,   # Save memory
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = SimpleDataCollator(tokenizer, args.max_length)
    
    # Create loss logging callback
    loss_callback = LossLoggingCallback()
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback],  # Add the loss logging callback
    )
    
    # Train
    print("Starting training...")
    print(f"Total training steps: {len(tokenized_dataset) // (args.batch_size * 8) * args.epochs}")
    print("=" * 50)
    trainer.train()
    print("=" * 50)
    
    # Save adapter
    print("Saving adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete! Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()

