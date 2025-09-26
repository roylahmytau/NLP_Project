#!/usr/bin/env python3
"""
Generate 5 Q&A pairs per document from the 'qa_1_2048_fixed.jsonl' dataset.

For each record in /workspace/NLP_Project/needles/2048/qa_1_2048_fixed.jsonl,
send the document text to a chat/instruction model to synthesize 5 question-answer
pairs. Save the resulting text to individual files under outputs/qa_1_2048_fixed/.

This script uses Hugging Face transformers text-generation pipeline by default.
You can customize the model with --model and --device.
"""

import os
import json
import argparse
from typing import Iterator, Dict, Any, Tuple, Optional

from tqdm import tqdm

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None


DEFAULT_DATASET_PATH = \
    "/workspace/NLP_Project/needles/2048/qa_1_2048_fixed.jsonl"
DEFAULT_OUTPUT_DIR = \
    "/workspace/NLP_Project/outputs/qa_1_2048_fixed"


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue


def read_text_documents(path: str) -> Iterator[Tuple[int, str]]:
    """
    Yield (line_number, document_text) for each non-empty line that does not
    look like a Q/A line (e.g., starts with "Q1:" or "A1:").
    """
    def is_qa_line(s: str) -> bool:
        s2 = s.strip()
        if not s2:
            return False
        # Basic detection for lines like: Q1:, Q2:, A1:, A2:
        if len(s2) >= 3 and s2[0] in {"Q", "A"} and s2[1].isdigit() and s2[2] == ":":
            return True
        return False

    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if is_qa_line(line):
                # Skip existing Q/A lines present in the source file
                continue
            yield idx, line


def sniff_dataset_mode(path: str) -> str:
    """Return 'jsonl' if the first non-empty line starts with '{', else 'text'."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return "jsonl" if s.startswith("{") else "text"
    return "text"


def build_prompt(document_text: str, num_qas: int) -> str:
    return (
        "You are a helpful assistant. Read the document below and generate "
        f"{num_qas} diverse, high-quality question and answer pairs that can be "
        "answered solely from the document. Questions should vary in style "
        "and difficulty (factual, inference, paraphrase). Answers must be "
        "grounded strictly in the document and be concise.\n\n"
        "Return them in this exact format:\n"
        "Q1: ...\nA1: ...\n"
        "Q2: ...\nA2: ...\n"
        "Q3: ...\nA3: ...\n"
        "Q4: ...\nA4: ...\n"
        "Q5: ...\nA5: ...\n\n"
        "Document:\n" + document_text
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate 5 Q&A pairs per document from qa_1_2048_fixed.jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Absolute path to qa_1_2048_fixed.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated Q&A text files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id for text-generation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "jsonl", "text"],
        default="auto",
        help="Parse mode: auto-detect, force jsonl, or force text (one doc per non-empty non-Q/A line)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for transformers pipeline: 'cpu', 'cuda', or 'auto'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process",
    )
    parser.add_argument(
        "--num_qas",
        type=int,
        default=5,
        help="Number of Q&A pairs to request per document",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=800,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    if pipeline is None:
        raise RuntimeError(
            "transformers is not available. Please install it per requirements.txt"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Create generator pipeline
    generator = pipeline(
        "text-generation",
        model=args.model,
        device=args.device if args.device != "auto" else None,
    )

    mode = args.mode
    if mode == "auto":
        mode = sniff_dataset_mode(args.dataset)

    count = 0
    if mode == "jsonl":
        iterator: Iterator[Tuple[int, Optional[str]]] = (
            (i, r.get("doc") or r.get("document") or r.get("context"))
            for i, r in enumerate(read_jsonl(args.dataset))
        )
    else:
        iterator = ((i, text) for i, text in read_text_documents(args.dataset))

    for seq_index, doc_text in tqdm(iterator, desc="Processing records"):
        if args.limit is not None and count >= args.limit:
            break

        if not doc_text:
            # Skip if no content
            continue

        prompt = build_prompt(doc_text, args.num_qas)

        try:
            outputs = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"] if outputs and outputs[0] else ""
        except Exception as e:
            text = f"ERROR: generation failed: {e}"

        # Name by sequential processed order so record_0 matches the first parsed document
        out_path = os.path.join(args.output_dir, f"record_{count}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

        count += 1

    print(f"Done. Wrote {count} files to {args.output_dir}")


if __name__ == "__main__":
    main()


