
#!/usr/bin/env python3
"""
benchmark.py — Thin orchestrator for running NVIDIA's RULER benchmark end‑to‑end
on a given model/provider.

It wraps the three official scripts from the RULER repo:
  1) scripts/data/prepare.py   → generate synthetic tasks as JSONL
  2) scripts/pred/call_api.py  → call your model and produce predictions
  3) scripts/eval/evaluate.py  → compute RULER scores

Repo (must be present or cloneable): https://github.com/NVIDIA/RULER

Quickstart examples
-------------------
# Evaluate a Hugging Face model locally with transformers (bf16 GPU recommended)
python benchmark.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --provider hf \
  --lengths 4096,8192 \
  --num-samples 50

# Evaluate an OpenAI model (requires OPENAI_API_KEY in env)
python benchmark.py \
  --model gpt-4o-mini \
  --provider openai \
  --lengths 4096,8192 \
  --num-samples 50

# Evaluate a vLLM server (point --model to the HF id served by vLLM)
python benchmark.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --provider vllm --server-host 127.0.0.1 --server-port 8000 \
  --lengths 4096 \
  --num-samples 50

Outputs
-------
For each context length L, results live under:
  <output_dir>/synthetic/<L>/
    data/  → generated datasets per task (JSONL)
    pred/  → predictions per task (JSONL) + summary-*.csv + submission.csv

Notes
-----
• Dependencies: see RULER/docker/requirements.txt. At minimum you’ll need
  torch, transformers, pyyaml, tiktoken, tqdm, nltk, and NVIDIA NeMo (nemo_toolkit).
  Using the official Docker image is easiest:
    docker pull cphsieh/ruler:0.2.0
• Tokenization: RULER’s prepare.py needs a tokenizer *type* and *path*.
  This script infers sensible defaults based on --provider, but you can override
  with --tokenizer-type/--tokenizer-path if needed.
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional

DEFAULT_TASKS = [
    # From scripts/synthetic.yaml in the RULER repo
    "niah_single_1","niah_single_2","niah_single_3",
    "niah_multikey_1","niah_multikey_2","niah_multikey_3",
    "niah_multivalue","niah_multiquery",
    "vt","cwe","fwe","qa_1","qa_2",
]

def sh(cmd: List[str], cwd: Optional[Path]=None, env: Optional[Dict[str,str]]=None):
    """Run a shell command, stream output, and raise on nonzero exit."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"Command failed with exit code {ret}: {' '.join(cmd)}")

def ensure_ruler(ruler_dir: Path, auto_clone: bool=True):
    """Make sure the RULER repo exists locally; clone if missing (optional)."""
    if ruler_dir.exists():
        return
    if not auto_clone:
        raise SystemExit(f"RULER repo not found at {ruler_dir}. Clone it or pass --ruler-dir.")
    print(f"[info] Cloning RULER into {ruler_dir} ...")
    sh(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/RULER", str(ruler_dir)])

def load_tasks(ruler_dir: Path, custom_tasks: Optional[str]) -> List[str]:
    """Return list of task names. If custom passed, use it; else read synthetic.yaml or fallback."""
    if custom_tasks and custom_tasks.lower() != "auto":
        return [t.strip() for t in custom_tasks.split(",") if t.strip()]
    # try to read from YAML
    syn_yaml = ruler_dir / "scripts" / "synthetic.yaml"
    if syn_yaml.exists():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(syn_yaml.read_text())
            # keys at top-level are task names
            tasks = list(data.keys())
            if tasks:
                return tasks
        except Exception as e:
            print(f"[warn] Failed to parse {syn_yaml}: {e}. Falling back to default tasks.")
    return DEFAULT_TASKS

def infer_tokenizer(provider: str, model: str, tok_type: Optional[str], tok_path: Optional[str]):
    """Infer tokenizer_type and tokenizer_path for RULER's prepare.py."""
    if tok_type and tok_path:
        return tok_type, tok_path
    if tok_type and not tok_path:
        # If type is provided but path missing, use model as a default path/name
        return tok_type, model
    if not tok_type and tok_path:
        # If path provided but type missing, guess from provider
        guessed = "hf" if provider in {"hf","vllm","trtllm","sglang","mamba"} else "openai"
        return guessed, tok_path
    # neither provided: choose sensibly
    if provider in {"hf","vllm","trtllm","sglang","mamba"}:
        return "hf", model
    elif provider in {"openai","gemini"}:
        # RULER's 'openai' tokenizer uses tiktoken encodings for known model names.
        return "openai", model
    else:
        # last resort
        return "hf", model

def main():
    parser = argparse.ArgumentParser(description="Run NVIDIA RULER on a model.")
    parser.add_argument("--ruler-dir", type=Path, default=Path(""),
                        help="Path to local clone of https://github.com/NVIDIA/RULER.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to store datasets & predictions. Default: <ruler-dir>/results/<model_slug>")
    parser.add_argument("--provider", choices=["hf","openai","gemini","vllm","trtllm","sglang","mamba"],
                        default="hf", help="Which client to use inside RULER.")
    parser.add_argument("--model", required=True,
                        help="Model name or path (HF id, local path, or API model name).")
    parser.add_argument("--lengths", default="4096",
                        help="Comma-separated max_seq_length values, e.g. 4096,8192,16384")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Samples per task subset (RULER default is 500).")
    parser.add_argument("--subset", choices=["validation","test"], default="validation")
    parser.add_argument("--tasks", default="auto",
                        help="Comma-separated RULER task names, or 'auto' to use synthetic.yaml.")
    parser.add_argument("--threads", type=int, default=4, help="Parallel threads for API calls.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop-words", default="", help="Comma-separated stop words.")
    parser.add_argument("--remove-newline-tab", action="store_true",
                        help="Strip newlines/tabs from generated data.")
    parser.add_argument("--chunk-amount", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--server-host", default="127.0.0.1",
                        help="For server providers (vllm/sglang/trtllm).")
    parser.add_argument("--server-port", default="5000",
                        help="For server providers (vllm/sglang/trtllm).")
    parser.add_argument("--tokenizer-type", choices=["hf","nemo","openai"], default=None,
                        help="Override RULER tokenizer type. Defaults to provider-derived.")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Override RULER tokenizer path/name. Defaults to --model.")
    parser.add_argument("--no-auto-clone", action="store_true",
                        help="Do not attempt to git clone RULER if missing.")
    args = parser.parse_args()

    # Resolve paths and sanity checks
    ensure_ruler(args.ruler_dir, auto_clone=not args.no_auto_clone)
    model_slug = args.model.replace("/", "__").replace(":", "_")
    out_root = args.output_dir or (args.ruler_dir / "results" / model_slug)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.ruler_dir, args.tasks)
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    tok_type, tok_path = infer_tokenizer(args.provider, args.model, args.tokenizer_type, args.tokenizer_path)

    # Check API keys for hosted providers
    if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        print("[warn] OPENAI_API_KEY env var not set; OpenAI calls will fail.", file=sys.stderr)
    if args.provider == "gemini" and "GOOGLE_API_KEY" not in os.environ:
        print("[warn] GOOGLE_API_KEY env var not set; Gemini calls will fail.", file=sys.stderr)

    # RULER working dirs (scripts expect these relative cwd's)
    data_scripts = args.ruler_dir / "scripts" / "data"
    pred_scripts = args.ruler_dir / "scripts" / "pred"
    eval_scripts = args.ruler_dir / "scripts" / "eval"
    for p in [data_scripts, pred_scripts, eval_scripts]:
        if not p.exists():
            raise SystemExit(f"Expected RULER path not found: {p}")

    # Run
    for L in lengths:
        run_root = out_root / "synthetic" / str(L)
        data_dir = run_root / "data"
        pred_dir = run_root / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Context length: {L} tokens ===")
        print(f"[info] Output root: {run_root}")

        # 1) Prepare datasets per task
        for task in tasks:
            print(f"\n--- Prepare task: {task} ---")
            cmd = [
                sys.executable, "prepare.py",
                "--save_dir", str(data_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--subset", args.subset,
                "--tokenizer_path", tok_path,
                "--tokenizer_type", tok_type,
                "--max_seq_length", str(L),
                "--model_template_type", "base",
                "--num_samples", str(args.num_samples),
                "--chunk_idx", str(args.chunk_idx),
                "--chunk_amount", str(args.chunk_amount),
            ]
            if args.remove_newline_tab:
                cmd.append("--remove_newline_tab")
            sh(cmd, cwd=data_scripts)

            # 2) Call the model / API to produce predictions
            print(f"--- Predict task: {task} ---")
            cmd = [
                sys.executable, "call_api.py",
                "--data_dir", str(data_dir),
                "--save_dir", str(pred_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--subset", args.subset,
                "--chunk_idx", str(args.chunk_idx),
                "--chunk_amount", str(args.chunk_amount),
                "--server_type", args.provider,
                "--model_name_or_path", args.model,
                "--threads", str(args.threads),
                "--temperature", str(args.temperature),
                "--top_k", str(args.top_k),
                "--top_p", str(args.top_p),
                "--stop_words", args.stop_words,
            ]
            # server info when applicable
            if args.provider in {"vllm","trtllm","sglang"}:
                cmd += ["--server_host", args.server_host, "--server_port", str(args.server_port)]
            sh(cmd, cwd=pred_scripts)

        # 3) Evaluate (aggregates per-task predictions and writes CSVs)
        print("\n=== Evaluate ===")
        cmd = [
            sys.executable, "evaluate.py",
            "--data_dir", str(pred_dir),
            "--benchmark", "synthetic",
        ]
        sh(cmd, cwd=eval_scripts)

    print("\nAll done! Look for summary-*.csv files under:", out_root)

if __name__ == "__main__":
    main()
