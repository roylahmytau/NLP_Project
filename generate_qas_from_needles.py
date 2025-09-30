#!/usr/bin/env python3
"""
Generate Q&A pairs per document by iterating over 3-sentence chunks.

For each record in /workspace/NLP_Project/needles/2048/qa_1_2048_fixed.jsonl,
split the document into paragraphs of 3 sentences and, for each paragraph,
ask the model to synthesize exactly 1 question-answer pair. Aggregate all
chunk-level Q&A pairs and save them into a single file per record under
outputs/qa_1_2048_fixed/.

This script uses Hugging Face transformers text-generation pipeline by default.
You can customize the model with --model and --device.
"""

import os
import json
import argparse
from typing import Iterator, Dict, Any, Tuple, Optional
import re

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
def extract_document(record: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the source document text from a record.

    Tries multiple common keys and simple fallbacks.
    Returns a stripped string or None if nothing usable is found.
    """
    candidates = [
        record.get("doc"),
        record.get("document"),
        record.get("context"),
        record.get("docs"),
        record.get("text"),
        record.get("content"),
    ]
    for val in candidates:
        if isinstance(val, str) and val.strip():
            return val.strip()
        # If list of strings, join
        if isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            joined = "\n\n".join(x.strip() for x in val if x and x.strip())
            if joined.strip():
                return joined
    # Some datasets have {"instruction": ..., "doc": ...}
    instr = record.get("instruction")
    doc = record.get("doc")
    if isinstance(doc, str) and doc.strip():
        return doc.strip()
    # Nothing usable
    return None



def sniff_dataset_mode(path: str) -> str:
    """Sniff dataset structure.

    Returns one of:
    - 'jsonl': each non-empty line is a JSON object
    - 'json_array': file is a JSON array of objects
    - 'text': plain text file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                return "jsonl"
            if s.startswith("["):
                return "json_array"
            return "text"
    return "text"


def build_prompt(document_text: str, num_qas: int) -> str:
    if num_qas == 1:
        return (
            "You are a helpful assistant. Read the text below and generate exactly one "
            "high-quality question and its answer that can be answered solely from the text. "
            "The answer must be concise and contain only the answer (no extra words).\n\n"
            "Return your response in exactly these two lines and nothing else:\n"
            "Q: ...\nA: ...\n\n"
            "Text:\n" + document_text
        )
    # Fallback for multi-QA requests
    return (
        "You are a helpful assistant. Read the document below and generate "
        f"{num_qas} diverse, high-quality question and answer pairs that can be "
        "answered solely from the document. Questions should vary in style "
        "and difficulty (factual, inference, paraphrase). Answers must be "
        "grounded strictly in the document and be concise, having just the answer without any other text.\n\n"
        "Return them in this exact format:\n"
        "Q1: ...\nA1: ...\n"
        "Q2: ...\nA2: ...\n"
        "Q3: ...\nA3: ...\n"
        f"{num_qas} such question and answer pairs.\n\n"
        "Document:\n" + document_text
    )


def extract_first_qa(generated_text: str) -> Optional[str]:
    """Extract the first Q/A pair from model output and normalize to two lines.

    Recognizes lines starting with Q:, Q1:, A:, A1:. If multiple pairs are present,
    only the first encountered pair is returned.
    """
    if not generated_text:
        return None
    lines = [ln.strip() for ln in generated_text.splitlines() if ln.strip()]
    q_pattern = re.compile(r"^Q(?:\d+)?:\s*(.+)$", flags=re.IGNORECASE)
    a_pattern = re.compile(r"^A(?:\d+)?:\s*(.+)$", flags=re.IGNORECASE)

    question: Optional[str] = None
    answer: Optional[str] = None

    # Find first Q then the first A after it
    for i, ln in enumerate(lines):
        if question is None:
            m = q_pattern.match(ln)
            if m:
                question = m.group(1).strip()
                # look ahead for answer
                for ln2 in lines[i + 1 :]:
                    m2 = a_pattern.match(ln2)
                    if m2:
                        answer = m2.group(1).strip()
                        break
                break

    if question and answer:
        return f"Q: {question}\nA: {answer}"

    # Fallback: try to coerce first two non-empty lines into Q/A
    if len(lines) >= 2:
        return f"Q: {lines[0]}\nA: {lines[1]}"

    return None


def split_into_sentences(text: str) -> Iterator[str]:
    """A simple sentence splitter based on punctuation boundaries.

    Splits on '.', '!' or '?' followed by whitespace. Keeps the punctuation
    attached to the sentence. This is a heuristic and may not perfectly
    handle all edge cases, but suffices for general prose.
    """
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return iter(())

    # Split on sentence-ending punctuation followed by a space
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    # Yield non-empty sentences
    return (s for s in parts if s)


def chunk_sentences(sentences: Iterator[str], sentences_per_chunk: int = 3) -> Iterator[str]:
    """Group sentences into chunks of a fixed size and yield paragraph strings."""
    buf: list[str] = []
    for s in sentences:
        if not s:
            continue
        buf.append(s)
        if len(buf) >= sentences_per_chunk:
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs per document by iterating over 3-sentence chunks"
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
        default=1,
        help="Number of Q&A pairs to request per chunk (use 1)",
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
    parser.add_argument(
        "--sentences_per_chunk",
        type=int,
        default=3,
        help="Number of sentences to include in each chunk",
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
            (i, extract_document(r))
            for i, r in enumerate(read_jsonl(args.dataset))
        )
    elif mode == "json_array":
        # Load entire JSON array, then iterate items
        with open(args.dataset, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except Exception as e:
                print(f"Failed to parse JSON array: {e}")
                records = []
        iterator = (
            (i, extract_document(rec))
            for i, rec in enumerate(records)
        )
    else:
        iterator = ((i, text) for i, text in read_text_documents(args.dataset))

    for seq_index, doc_text in tqdm(iterator, desc="Processing records"):
        if args.limit is not None and count >= args.limit:
            break

        if not doc_text or not doc_text.strip():
            # Skip if no content
            continue

        # Prepare output path and open file once per record
        out_path = os.path.join(args.output_dir, f"record_{count}.txt")
        # Truncate/create the file at the start of processing this record
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")

        # Split into chunks of N sentences and generate 1 QA per chunk
        sentences_iter = split_into_sentences(doc_text)
        chunks_iter = chunk_sentences(sentences_iter, sentences_per_chunk=args.sentences_per_chunk)

        # Append each chunk's output immediately so progress is visible on disk
        for chunk_text in chunks_iter:
            prompt = build_prompt(chunk_text, 1)
            try:
                outputs = generator(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    return_full_text=False,
                )
                gen = outputs[0]["generated_text"] if outputs and outputs[0] else ""
            except Exception as e:
                gen = f"ERROR: generation failed: {e}"
            if gen and gen.strip():
                first_pair = extract_first_qa(gen)
                if first_pair:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(first_pair.strip() + "\n")
                        f.flush()

        count += 1

    print(f"Done. Wrote {count} files to {args.output_dir}")


if __name__ == "__main__":
    main()


