#!/usr/bin/env python3
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os
from tqdm import tqdm
from datetime import datetime
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
    # Prefer left padding for causal LMs during generation
    try:
        tokenizer.padding_side = 'left'
    except Exception:
        pass
    
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

def load_penn_treebank_texts(split="validation", max_samples=1000):
    """Load sentences from the Penn Treebank via Hugging Face datasets.

    Attempts several dataset identifiers. If PTB is unavailable, falls back to WikiText-2 raw.
    Returns a list of strings.
    """
    dataset = None
    last_error = None
    # Try canonical ids first
    for ds_id in [
        "penn_treebank",
        "ptb_text_only",
        "tner/penn_treebank",
    ]:
        try:
            dataset = load_dataset(ds_id, split=split)
            print(f"Loaded Penn Treebank from '{ds_id}' split='{split}'")
            break
        except Exception as e:
            last_error = e
            dataset = None

    # Fallback: try config variants for ptb_text_only
    if dataset is None:
        try:
            dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
            print(f"Loaded Penn Treebank from 'ptb_text_only/penn_treebank' split='{split}'")
        except Exception as e:
            last_error = e
            dataset = None

    # Final fallback to WikiText-2 raw if PTB not available
    if dataset is None:
        print("Warning: Penn Treebank not available (" + str(last_error) + ") — falling back to WikiText-2 raw.")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation" if split == "validation" else split)
            print(f"Loaded fallback dataset 'wikitext-2-raw-v1' split='{split}'")
        except Exception as e:
            raise RuntimeError(f"Failed to load Penn Treebank and fallback WikiText-2: {e}")

    texts = []
    for item in dataset:
        sent = None
        if isinstance(item, dict):
            # Try common text fields
            for key in ["sentence", "text", "line", "content"]:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    sent = val.strip()
                    break
        if isinstance(sent, str) and sent.strip():
            texts.append(sent)
    # Filter out empty lines
    texts = [t for t in texts if t]
    if max_samples is not None and len(texts) > max_samples:
        texts = texts[:max_samples]
    return texts

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
                        prompt = (
                            "<|im_start|>user\n"
                            "Choose the correct answer to the multiple-choice question. "
                            "Respond with only the letter A, B, C, or D.\n\n"
                            f"{question}\n"
                            "<|im_end|>\n<|im_start|>assistant\n"
                        )
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
                            # Robustly slice new tokens: prefer full padded length
                            padded_len = inputs['input_ids'][j].shape[0]
                            decoded = tokenizer.decode(output[padded_len:], skip_special_tokens=True).strip()
                            response = decoded if decoded else tokenizer.decode(output, skip_special_tokens=True).strip()

                            # Extract answer choice robustly
                            text_up = response.upper()
                            answer_choice = None
                            # Direct letter
                            import re as _re
                            m = _re.search(r"\b([ABCD])\b", text_up)
                            if m:
                                answer_choice = m.group(1)
                            else:
                                # Common patterns like 'Answer: B', '(C)', 'Option D'
                                m2 = _re.search(r"(ANSWER\s*[:\-]?\s*|OPTION\s+)([ABCD])\b", text_up)
                                if m2:
                                    answer_choice = m2.group(2)
                                else:
                                    # Map numbers to letters if present
                                    m3 = _re.search(r"\b([1-4])\b", text_up)
                                    if m3:
                                        num = int(m3.group(1))
                                        answer_choice = ['A','B','C','D'][num-1]
                            predictions.append(answer_choice)
                
                # Calculate accuracy
                # Normalize gold answers to letters
                def _norm_gold(x):
                    if isinstance(x, str):
                        xu = x.strip().upper()
                        if xu in ['A','B','C','D']:
                            return xu
                        if xu in ['0','1','2','3','4']:
                            try:
                                n = int(xu)
                                if n in (0,1,2,3):
                                    return ['A','B','C','D'][n]
                                if n in (1,2,3,4):
                                    return ['A','B','C','D'][n-1]
                            except Exception:
                                return None
                        return None
                    if isinstance(x, int):
                        if x in (0,1,2,3):
                            return ['A','B','C','D'][x]
                        if x in (1,2,3,4):
                            return ['A','B','C','D'][x-1]
                    return None
                gold_letters = [_norm_gold(x) for x in correct_answers]
                correct = sum(1 for pred, gt in zip(predictions, gold_letters) if pred is not None and gt is not None and pred == gt)
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

def _append_jsonl(output_file, record):
    """Append a JSON record as a single line to output_file, creating parent dirs if needed."""
    try:
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    try:
        with open(output_file, 'a') as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Warning: failed to append progress to {output_file}: {e}")

def run_mmlu_single_task_streaming(model, tokenizer, task, max_samples=100, batch_size=8, output_file=None, adapter_label=None):
    """Run a single MMLU task with batch generation and stream progress to a JSONL file.

    Writes records like:
    {"ts": iso, "event": "category_start", "adapter": adapter_label, "category": task, "total": total}
    {"ts": iso, "event": "category_progress", "adapter": adapter_label, "category": task, "completed": n, "total": total}
    {"ts": iso, "event": "category_complete", "adapter": adapter_label, "category": task, "accuracy": acc, "correct": c, "total": t}
    """
    try:
        dataset = load_dataset("cais/mmlu", task, split="test")
    except Exception as e:
        if output_file:
            _append_jsonl(output_file, {
                "ts": datetime.utcnow().isoformat() + "Z",
                "event": "category_error",
                "adapter": adapter_label,
                "category": task,
                "error": str(e),
            })
        print(f"Error loading MMLU task {task}: {e}")
        return {"accuracy": 0, "correct": 0, "total": 0}

    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    questions = []
    correct_answers = []
    for item in dataset:
        question = f"{item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}"
        questions.append(question)
        correct_answers.append(item['answer'])

    total = len(questions)
    predictions = []

    if output_file:
        _append_jsonl(output_file, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "category_start",
            "adapter": adapter_label,
            "category": task,
            "total": total,
        })

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_prompts = []
        for q in batch_questions:
            prompt = (
                "<|im_start|>user\n"
                "Choose the correct answer to the multiple-choice question. "
                "Respond with only the letter A, B, C, or D.\n\n"
                f"{q}\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            batch_prompts.append(prompt)

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
            for j, output in enumerate(outputs):
                padded_len = inputs['input_ids'][j].shape[0]
                decoded = tokenizer.decode(output[padded_len:], skip_special_tokens=True).strip()
                response = decoded if decoded else tokenizer.decode(output, skip_special_tokens=True).strip()
                text_up = response.upper()
                answer_choice = None
                import re as _re
                m = _re.search(r"\b([ABCD])\b", text_up)
                if m:
                    answer_choice = m.group(1)
                else:
                    m2 = _re.search(r"(ANSWER\s*[:\-]?\s*|OPTION\s+)([ABCD])\b", text_up)
                    if m2:
                        answer_choice = m2.group(2)
                    else:
                        m3 = _re.search(r"\b([1-4])\b", text_up)
                        if m3:
                            num = int(m3.group(1))
                            answer_choice = ['A','B','C','D'][num-1]
                predictions.append(answer_choice)

        if output_file:
            _append_jsonl(output_file, {
                "ts": datetime.utcnow().isoformat() + "Z",
                "event": "category_progress",
                "adapter": adapter_label,
                "category": task,
                "completed": len(predictions),
                "total": total,
            })

    # Normalize gold answers to letters
    def _norm_gold(x):
        if isinstance(x, str):
            xu = x.strip().upper()
            if xu in ['A','B','C','D']:
                return xu
            if xu in ['0','1','2','3','4']:
                try:
                    n = int(xu)
                    if n in (0,1,2,3):
                        return ['A','B','C','D'][n]
                    if n in (1,2,3,4):
                        return ['A','B','C','D'][n-1]
                except Exception:
                    return None
            return None
        if isinstance(x, int):
            if x in (0,1,2,3):
                return ['A','B','C','D'][x]
            if x in (1,2,3,4):
                return ['A','B','C','D'][x-1]
        return None
    gold_letters = [_norm_gold(x) for x in correct_answers]
    correct = sum(1 for pred, gt in zip(predictions, gold_letters) if pred is not None and gt is not None and pred == gt)
    accuracy = (correct / total) if total > 0 else 0

    if output_file:
        _append_jsonl(output_file, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "category_complete",
            "adapter": adapter_label,
            "category": task,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        })

    return {"accuracy": accuracy, "correct": correct, "total": total}

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
    parser.add_argument("--perplexity_split", default="validation",
                       choices=["train", "validation", "test"],
                       help="Penn Treebank split to use for perplexity when no text file is provided")
    parser.add_argument("--mmlu_subset", default="all",
                       help="MMLU subset to evaluate (or 'all' for all tasks)")
    parser.add_argument("--glue_task", default="all",
                       choices=["sst2", "mrpc", "qnli", "all"],
                       help="GLUE task to evaluate: sst2, mrpc, qnli, or 'all' for all three tasks")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for GPU processing (default: 8)")
    # Multi-adapter MMLU extensions
    parser.add_argument("--adapter_1", default=None, help="Path to LoRA adapter 1")
    parser.add_argument("--adapter_2", default=None, help="Path to LoRA adapter 2")
    parser.add_argument("--adapter_3", default=None, help="Path to LoRA adapter 3")
    parser.add_argument("--mmlu_categories", default=None,
                       help="Comma-separated list of 3 MMLU categories/subjects to run (e.g., 'anatomy,astronomy,abstract_algebra')")
    parser.add_argument("--progress_file", default=None,
                       help="Optional path to a JSONL file to stream per-category progress events")
    
    args = parser.parse_args()
    
    print("Starting benchmark...")
    # print args
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Generate data file path from needle size and type
    data_file = get_data_file_path(args.needle_size, args.needle_type)
    print(f"Using data file: {data_file}")
    
    # Load model
    # Decide whether we're in multi-adapter MMLU streaming mode
    multi_adapter = args.benchmark in ["mmlu"] and any([args.adapter_1, args.adapter_2, args.adapter_3]) and args.mmlu_categories is not None

    # Normalize progress output file
    progress_file = args.progress_file
    if progress_file is None and multi_adapter:
        progress_file = os.path.splitext(args.output_file)[0] + ".mmlu_progress.jsonl"

    if multi_adapter:
        # Prepare adapters and categories (cross product run)
        adapters = [a for a in [args.adapter_1, args.adapter_2, args.adapter_3] if a]
        categories = [c.strip() for c in args.mmlu_categories.split(',') if c.strip()]
        if len(adapters) != 3 or len(categories) != 3:
            raise ValueError("Provide exactly 3 adapters via --adapter_1 --adapter_2 --adapter_3 and 3 categories via --mmlu_categories")
    
    print("Loading model...")
    if not multi_adapter:
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
            # Use Penn Treebank sentences by default
            texts = load_penn_treebank_texts(split=args.perplexity_split, max_samples=args.max_samples)
        
        print(f"Calculating perplexity on {len(texts)} texts...")
        perplexity_results = calculate_perplexity(model, tokenizer, texts)
        all_results["perplexity"] = perplexity_results
        
        print(f"Perplexity: {perplexity_results['perplexity']:.3f}")
        print(f"Average Loss: {perplexity_results['average_loss']:.3f}")
        print(f"Total Tokens: {perplexity_results['total_tokens']}")
    
    # Run MMLU benchmark
    if args.benchmark in ["mmlu"]: # , "all"
        if multi_adapter:
            print("\n=== Running MMLU Benchmark (multi-adapter streaming, all categories per adapter) ===")
            # Stream run start
            if progress_file:
                _append_jsonl(progress_file, {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "event": "run_start",
                    "benchmark": "mmlu",
                    "adapters": adapters,
                    "categories": categories,
                })

            mmlu_summaries = {}
            for idx, adapter_path_i in enumerate(adapters, start=1):
                print(f"\n--- Adapter {idx}: {adapter_path_i} ---")
                # Load model for this adapter
                model_i, tokenizer_i = load_model_and_tokenizer(args.model_name, adapter_path_i)

                # Stream adapter start (list categories)
                if progress_file:
                    _append_jsonl(progress_file, {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event": "adapter_start",
                        "adapter": adapter_path_i,
                        "index": idx,
                        "categories": categories,
                    })

                adapter_key = f"adapter_{idx}"
                mmlu_summaries[adapter_key] = {"adapter_path": adapter_path_i, "results": {}}

                for category_i in categories:
                    print(f"  > Category: {category_i}")
                    result_i = run_mmlu_single_task_streaming(
                        model_i, tokenizer_i, category_i, max_samples=args.max_samples,
                        batch_size=args.batch_size, output_file=progress_file, adapter_label=adapter_path_i,
                    )
                    mmlu_summaries[adapter_key]["results"][category_i] = result_i

                # Stream adapter complete (optional summary accuracy across categories)
                if progress_file:
                    # compute macro average accuracy across categories
                    accs = [r.get("accuracy", 0) for r in mmlu_summaries[adapter_key]["results"].values()]
                    avg_acc = float(sum(accs) / len(accs)) if accs else 0.0
                    _append_jsonl(progress_file, {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event": "adapter_complete",
                        "adapter": adapter_path_i,
                        "index": idx,
                        "avg_accuracy": avg_acc,
                    })

                # Free model to reduce VRAM before next adapter
                try:
                    del model_i
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            all_results["mmlu_multi_adapter"] = mmlu_summaries

            # Stream run complete
            if progress_file:
                _append_jsonl(progress_file, {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "event": "run_complete",
                    "benchmark": "mmlu",
                })

        else:
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
    
    # For multi-adapter mode, keep progress JSONL separate and still emit a summary JSON
    with open(args.output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()

