#!/usr/bin/env python3
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os
from tqdm import tqdm
import re
import numpy as np
import math
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr


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

def load_qa_data(file_path):
    """Load and format QA data for training"""
    data = []
    print(f"Loading QA data from {file_path}")
    with open(file_path, 'r') as f:
        # Parse as JSONL (one JSON object per line)
        data = json.load(f)
        
        print(f"Successfully loaded {len(data)} items from JSON file")
    
    return data

def load_model_and_tokenizer(model_name, adapter_path):
    """Load base model and LoRA adapter"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure tokenizer has proper special tokens
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter from local path
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def generate_answer(model, tokenizer, instruction, doc, question, max_length=100):
    """Generate answer for a given question (single prompt)"""
    # Use Qwen-compatible format
    prompt = f"<|im_start|>user\n{instruction}\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,  # Low temperature for stable generation
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract answer - get only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up the answer - remove any remaining special tokens
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0].strip()
            
            # Remove trailing punctuation if it's just a period
            if answer.endswith('.') and len(answer) > 1:
                answer = answer[:-1]
            
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error generating answer"

def generate_answers_batch(model, tokenizer, instructions, docs, questions, max_length=100, batch_size=8):
    """Generate answers for a batch of questions"""
    all_answers = []
    
    # Process in batches
    for i in range(0, len(questions), batch_size):
        batch_instructions = instructions[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        # Create batch prompts
        batch_prompts = []
        for instruction, doc, question in zip(batch_instructions, batch_docs, batch_questions):
            prompt = f"<|im_start|>user\n{instruction}\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n"
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Process each output in the batch
                for j, output in enumerate(outputs):
                    # Extract answer - get only new tokens
                    input_length = inputs['input_ids'][j].shape[0]
                    new_tokens = output[input_length:]
                    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    # Clean up the answer
                    if "<|im_end|>" in answer:
                        answer = answer.split("<|im_end|>")[0].strip()
                    
                    # Remove trailing punctuation if it's just a period
                    if answer.endswith('.') and len(answer) > 1:
                        answer = answer[:-1]
                    
                    all_answers.append(answer)
                    
            except Exception as e:
                print(f"Batch generation error: {e}")
                # Add error responses for this batch
                for _ in range(len(batch_prompts)):
                    all_answers.append("Error generating answer")
    
    return all_answers

def evaluate_answers(predictions, ground_truths):
    """Simple evaluation metrics"""
    exact_matches = 0
    partial_matches = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.lower().strip()
        gt_clean = gt.lower().strip()
        
        if pred_clean == gt_clean:
            exact_matches += 1
        elif any(word in pred_clean for word in gt_clean.split()):
            partial_matches += 1
    
    total = len(predictions)
    exact_accuracy = exact_matches / total
    partial_accuracy = (exact_matches + partial_matches) / total
    
    return {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "total": total
    }

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """Calculate perplexity for a list of texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Calculate number of tokens (excluding padding)
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "average_loss": avg_loss,
        "total_tokens": total_tokens
    }

def run_mmlu_benchmark(model, tokenizer, subset="all", max_samples=100, batch_size=8):
    """Run MMLU (Massive Multitask Language Understanding) benchmark with batch processing"""
    try:
        # Load MMLU dataset
        if subset == "all":
            # Load a representative subset of MMLU tasks
            tasks = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", 
                    "clinical_knowledge", "college_biology", "college_chemistry", 
                    "college_computer_science", "college_mathematics", "college_medicine",
                    "college_physics", "computer_security", "conceptual_physics",
                    "econometrics", "electrical_engineering", "elementary_mathematics",
                    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry"]
        else:
            tasks = [subset]
        
        all_results = {}
        
        for task in tasks:
            try:
                print(f"Running MMLU task: {task}")
                dataset = load_dataset("cais/mmlu", task, split="test")
                
                # Limit samples for faster evaluation
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                
                # Prepare batch data
                questions = []
                correct_answers = []
                
                for item in dataset:
                    # Format question
                    question = f"{item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}"
                    questions.append(question)
                    correct_answers.append(item['answer'])
                
                # Generate answers in batches
                predictions = []
                for i in range(0, len(questions), batch_size):
                    batch_questions = questions[i:i+batch_size]
                    
                    # Create batch prompts
                    batch_prompts = []
                    for question in batch_questions:
                        prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
                        batch_prompts.append(prompt)
                    
                    # Tokenize batch
                    inputs = tokenizer(
                        batch_prompts, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=1024,
                        padding=True
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        # Process each output in the batch
                        for j, output in enumerate(outputs):
                            input_length = inputs['input_ids'][j].shape[0]
                            response = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
                            
                            # Extract answer choice
                            answer_choice = None
                            for choice in ['A', 'B', 'C', 'D']:
                                if choice in response.upper():
                                    answer_choice = choice
                                    break
                            
                            predictions.append(answer_choice)
                
                # Calculate accuracy
                correct = sum(1 for pred, gt in zip(predictions, correct_answers) if pred == gt)
                total = len(predictions)
                accuracy = correct / total if total > 0 else 0
                
                all_results[task] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total
                }
                
            except Exception as e:
                print(f"Error in MMLU task {task}: {e}")
                all_results[task] = {"accuracy": 0, "correct": 0, "total": 0}
        
        # Calculate overall accuracy
        total_correct = sum(result["correct"] for result in all_results.values())
        total_questions = sum(result["total"] for result in all_results.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "task_results": all_results
        }
        
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        return {"overall_accuracy": 0, "total_correct": 0, "total_questions": 0, "task_results": {}}

def run_glue_benchmark(model, tokenizer, task="all", max_samples=100, batch_size=8):
    """Run GLUE (General Language Understanding Evaluation) benchmark with batch processing"""
    try:
        # Define GLUE tasks and their configurations - limited to SST-2, MRPC, and QNLI
        glue_tasks = {
            "sst2": {"dataset": "glue", "subset": "sst2", "metric": "accuracy"},
            "mrpc": {"dataset": "glue", "subset": "mrpc", "metric": "f1"},
            "qnli": {"dataset": "glue", "subset": "qnli", "metric": "accuracy"}
        }
        
        if task != "all":
            if task in glue_tasks:
                glue_tasks = {task: glue_tasks[task]}
            else:
                print(f"Warning: Task '{task}' not available. Available tasks: {list(glue_tasks.keys())}")
                return {"average_score": 0, "task_results": {}}
        
        all_results = {}
        
        for task_name, config in glue_tasks.items():
            try:
                print(f"Running GLUE task: {task_name}")
                dataset = load_dataset(config["dataset"], config["subset"], split="validation")
                
                # Limit samples for faster evaluation
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                
                # Prepare batch data
                prompts = []
                labels = []
                
                for item in dataset:
                    # Format input based on task
                    if task_name in ["cola", "sst2"]:
                        # Single sentence tasks
                        text = item["sentence"]
                        prompt = f"<|im_start|>user\nClassify the following sentence as positive or negative:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
                    elif task_name in ["mrpc", "qqp"]:
                        # Paraphrase tasks
                        text1 = item["sentence1"]
                        text2 = item["sentence2"]
                        prompt = f"<|im_start|>user\nAre these two sentences equivalent?\nSentence 1: {text1}\nSentence 2: {text2}\nAnswer yes or no.\n<|im_end|>\n<|im_start|>assistant\n"
                    elif task_name == "stsb":
                        # Similarity task
                        text1 = item["sentence1"]
                        text2 = item["sentence2"]
                        prompt = f"<|im_start|>user\nRate the similarity between these sentences on a scale of 0-5:\nSentence 1: {text1}\nSentence 2: {text2}\n<|im_end|>\n<|im_start|>assistant\n"
                    elif task_name in ["mnli", "qnli", "rte", "wnli"]:
                        # NLI tasks
                        premise = item["premise"]
                        hypothesis = item["hypothesis"]
                        prompt = f"<|im_start|>user\nDoes the premise entail the hypothesis?\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer yes, no, or maybe.\n<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        continue
                    
                    prompts.append(prompt)
                    labels.append(item["label"])
                
                # Generate predictions in batches
                predictions = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i+batch_size]
                    
                    # Tokenize batch
                    inputs = tokenizer(
                        batch_prompts, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=1024,
                        padding=True
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        # Process each output in the batch
                        for j, output in enumerate(outputs):
                            input_length = inputs['input_ids'][j].shape[0]
                            response = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
                            
                            # Parse prediction based on task
                            if task_name in ["cola", "sst2"]:
                                pred = 1 if "positive" in response.lower() else 0
                            elif task_name in ["mrpc", "qqp"]:
                                pred = 1 if "yes" in response.lower() else 0
                            elif task_name == "stsb":
                                # Extract numeric score
                                import re
                                numbers = re.findall(r'\d+\.?\d*', response)
                                pred = float(numbers[0]) if numbers else 0.0
                            elif task_name in ["mnli", "qnli", "rte", "wnli"]:
                                if "yes" in response.lower():
                                    pred = 0  # entailment
                                elif "no" in response.lower():
                                    pred = 1  # contradiction
                                else:
                                    pred = 2  # neutral
                            else:
                                pred = 0
                            
                            predictions.append(pred)
                
                # Calculate metrics
                if task_name == "stsb":
                    # Pearson correlation for STS-B
                    correlation, _ = pearsonr(predictions, labels)
                    score = correlation
                elif task_name == "cola":
                    # Matthews correlation for CoLA
                    score = matthews_corrcoef(labels, predictions)
                elif task_name in ["mrpc", "qqp"]:
                    # F1 score for MRPC and QQP
                    score = f1_score(labels, predictions)
                else:
                    # Accuracy for other tasks
                    score = accuracy_score(labels, predictions)
                
                all_results[task_name] = {
                    "score": score,
                    "predictions": predictions,
                    "labels": labels
                }
                
            except Exception as e:
                print(f"Error in GLUE task {task_name}: {e}")
                all_results[task_name] = {"score": 0, "predictions": [], "labels": []}
        
        # Calculate average score
        valid_scores = [result["score"] for result in all_results.values() if isinstance(result["score"], (int, float))]
        avg_score = np.mean(valid_scores) if valid_scores else 0
        
        return {
            "average_score": avg_score,
            "task_results": all_results
        }
        
    except Exception as e:
        print(f"Error loading GLUE dataset: {e}")
        return {"average_score": 0, "task_results": {}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B",
                       help="Base model name")
    parser.add_argument("--adapter_path", default="adapter",
                       help="Path to LoRA adapter")
    parser.add_argument("--needle_size", default="2048", choices=["2048", "32768", "131072"],
                       help="Needle size (default: 2048)")
    parser.add_argument("--needle_type", default="qa_1", 
                       choices=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", 
                               "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"],
                       help="Needle type (default: qa_1)")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to evaluate")
    parser.add_argument("--output_file", default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--benchmark", default="qa", 
                       choices=["success", "perplexity", "mmlu", "glue", "all"],
                       help="Type of benchmark to run")
    parser.add_argument("--perplexity_texts", default=None,
                       help="Path to text file for perplexity evaluation (one text per line)")
    parser.add_argument("--mmlu_subset", default="all",
                       help="MMLU subset to evaluate (or 'all' for all tasks)")
    parser.add_argument("--glue_task", default="all",
                       choices=["sst2", "mrpc", "qnli", "all"],
                       help="GLUE task to evaluate: sst2, mrpc, qnli, or 'all' for all three tasks")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for GPU processing (default: 8)")
    
    args = parser.parse_args()
    
    print("Starting benchmark...")
    # print args
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Generate data file path from needle size and type
    data_file = get_data_file_path(args.needle_size, args.needle_type)
    print(f"Using data file: {data_file}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_path)
    
    all_results = {}
    
    # Run success benchmark
    if args.benchmark in ["success", "all"]: 
        print("\n=== Running QA Benchmark ===")
        # Load data
        print("Loading test data...")
        data = load_qa_data(data_file)
        print(f"Evaluating on {len(data)} samples")
        
        # Prepare batch data
        instructions = [item['instruction'] for item in data]
        docs = [item['doc'] for item in data]
        questions = [item['question'] for item in data]
        ground_truths = [item['outputs'][0] if item['outputs'] else "" for item in data]
        
        # Generate predictions in batches
        print(f"Generating predictions with batch size {args.batch_size}...")
        predictions = generate_answers_batch(
            model, tokenizer, instructions, docs, questions, 
            max_length=100, batch_size=args.batch_size
        )
        
        # Print sample results
        print("\n--- Sample Results ---")
        for i in range(min(3, len(predictions))):
            print(f"\nSample {i+1}:")
            print(f"Question: {questions[i]}")
            print(f"Expected: {ground_truths[i]}")
            print(f"Predicted: {predictions[i]}")
        
        # Evaluate
        print("Evaluating...")
        qa_results = evaluate_answers(predictions, ground_truths)
        all_results["qa"] = qa_results
        
        # Print results
        print("\n=== QA Benchmark Results ===")
        print(f"Exact Accuracy: {qa_results['exact_accuracy']:.3f}")
        print(f"Partial Accuracy: {qa_results['partial_accuracy']:.3f}")
        print(f"Exact Matches: {qa_results['exact_matches']}/{qa_results['total']}")
        print(f"Partial Matches: {qa_results['partial_matches']}/{qa_results['total']}")
    
    # Run Perplexity benchmark
    if args.benchmark in ["perplexity", "all"]:
        print("\n=== Running Perplexity Benchmark ===")
        
        if args.perplexity_texts:
            # Load texts from file
            with open(args.perplexity_texts, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Use default sample texts
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Natural language processing is a fascinating field of artificial intelligence.",
                "Machine learning models can understand and generate human-like text.",
                "Transformers have revolutionized the field of natural language processing.",
                "Large language models demonstrate remarkable capabilities in various tasks."
            ]
        
        print(f"Calculating perplexity on {len(texts)} texts...")
        perplexity_results = calculate_perplexity(model, tokenizer, texts)
        all_results["perplexity"] = perplexity_results
        
        print(f"Perplexity: {perplexity_results['perplexity']:.3f}")
        print(f"Average Loss: {perplexity_results['average_loss']:.3f}")
        print(f"Total Tokens: {perplexity_results['total_tokens']}")
    
    # Run MMLU benchmark
    if args.benchmark in ["mmlu"]: # , "all"
        print("\n=== Running MMLU Benchmark ===")
        mmlu_results = run_mmlu_benchmark(model, tokenizer, args.mmlu_subset, args.max_samples, args.batch_size)
        all_results["mmlu"] = mmlu_results
        
        print(f"Overall MMLU Accuracy: {mmlu_results['overall_accuracy']:.3f}")
        print(f"Total Correct: {mmlu_results['total_correct']}/{mmlu_results['total_questions']}")
        
        # Print per-task results
        for task, result in mmlu_results['task_results'].items():
            print(f"{task}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
    
    # Run GLUE benchmark
    if args.benchmark in ["glue", "all"]:
        print("\n=== Running GLUE Benchmark ===")
        glue_results = run_glue_benchmark(model, tokenizer, args.glue_task, args.max_samples, args.batch_size)
        all_results["glue"] = glue_results
        
        print(f"Average GLUE Score: {glue_results['average_score']:.3f}")
        
        # Print per-task results
        for task, result in glue_results['task_results'].items():
            print(f"{task}: {result['score']:.3f}")
    
    # Save detailed results
    detailed_results = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "needle_size": args.needle_size,
        "needle_type": args.needle_type,
        "data_file": data_file,
        "benchmark_type": args.benchmark,
        "results": all_results
    }
    
    # Add QA predictions if available
    if "qa" in all_results and args.benchmark in ["qa", "all"]:
        detailed_results["qa_predictions"] = [
            {"question": data[i]['question'], 
             "predicted": predictions[i], 
             "expected": ground_truths[i]}
            for i in range(len(predictions))
        ]
    
    with open(args.output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()

