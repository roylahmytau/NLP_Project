#!/usr/bin/env python3
"""
Quick training script with optimized settings for RTX 3090
"""
import subprocess
import sys
import argparse
import shutil
import os

def main():
    parser = argparse.ArgumentParser(description="Quick LoRA training with RTX 3090 optimized settings")
    parser.add_argument("--needle_size", required=True, choices=["2048", "32768", "131072"],
                       help="Needle size (required): 2048, 32768, or 131072")
    parser.add_argument("--needle_type", required=True, 
                       choices=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", 
                               "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"],
                       help="Needle type (required): niah_multikey_1, niah_multikey_2, niah_multikey_3, niah_single_1, niah_single_2, niah_single_3, or qa_1")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--adapter_r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--adapter_layers", type=int, default=10, help="Number of adapter layers (default: 10)")
    parser.add_argument("--output_dir", type=str, default="adapter", help="Output directory for adapter")
    parser.add_argument("--keep_adapter", action="store_true", help="Keep adapter directory after benchmarking")
    parser.add_argument("--use_generated_qa", action="store_true", help="If set, augment training with generated Q&A prompts")
    parser.add_argument("--generated_qa_dir", type=str, help="Directory containing generated Q&A files (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Optimized settings for RTX 3090
    cmd = [
        "python", "train_lora_optimized.py",
        "--model_name", "Qwen/Qwen3-8B",
        "--needle_size", args.needle_size,
        "--needle_type", args.needle_type,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--adapter_r", str(args.adapter_r),
        "--adapter_layers", str(args.adapter_layers),
        "--output_dir", args.output_dir,
    ]

    if args.use_generated_qa:
        cmd += ["--use_generated_qa"]
        if args.generated_qa_dir:
            cmd += ["--generated_qa_dir", args.generated_qa_dir]
    
    print("Starting LoRA training with RTX 3090 optimized settings...")
    print(f"Needle size: {args.needle_size}, Needle type: {args.needle_type}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        # run benchmark against the produced adapter
        subprocess.run(["python", "benchmark_model.py", "--adapter_path", args.output_dir], check=True)
        # optionally delete the adapter to save space
        adapter_dir = args.output_dir
        if not args.keep_adapter and os.path.exists(adapter_dir):
            try:
                shutil.rmtree(adapter_dir)
                print(f"Deleted adapter directory: {adapter_dir}")
            except Exception as e:
                print(f"Failed to delete adapter directory '{adapter_dir}': {e}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

