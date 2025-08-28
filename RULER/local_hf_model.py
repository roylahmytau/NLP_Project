
#!/usr/bin/env python3
"""
local_hf_model.py — tiny local Hugging Face text-generation client

Purpose
-------
Give you a simple, *ungated* local model you can actually run and, if you like,
use alongside RULER (either via RULER's built-in `--provider hf` or by
batching over a JSONL yourself and then running RULER's evaluate.py).

Defaults
--------
- Model: Qwen/Qwen2.5-0.5B-Instruct (public)
- Works on CPU or GPU
- Exposes __call__ and process_batch (matching what RULER wrappers expect)

Examples
--------
# Single prompt
python local_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hi in one sentence."

# Batch over a RULER dataset JSONL and write predictions
python local_hf_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --in-jsonl /workspace/runs/gpt-4o-mini/synthetic/4096/data/niah_single_1/validation.jsonl \
  --out-jsonl /workspace/runs/local-qwen/pred/niah_single_1.jsonl \
  --max-new-tokens 128

# Then evaluate that folder with RULER
python scripts/eval/evaluate.py --data_dir /workspace/runs/local-qwen/pred --benchmark synthetic

Requirements
------------
- transformers, torch, accelerate (present in the RULER Docker image)
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.getenv("OPEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None

class LocalHFModel:
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL):
        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dtype policy
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def _single_generate(self, prompt: str, cfg: GenerationConfig) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        do_sample = cfg.temperature > 0.0
        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        gen_ids = out_ids[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if cfg.stop:
            for s in cfg.stop:
                if s:
                    text = text.split(s)[0]
        return text.strip()

    # RULER-like call signature
    def __call__(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0,
                 top_p: float = 1.0, stop: Optional[List[str]] = None, **kwargs) -> dict:
        cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stop=stop)
        text = self._single_generate(prompt, cfg)
        return {"text": [text]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        return [self.__call__(p, **kwargs) for p in prompts]

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, objs: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_prompt(obj: dict) -> str:
    # Try common fields used in synthetic tasks
    if "prompt" in obj and isinstance(obj["prompt"], str):
        return obj["prompt"]
    if "input" in obj and isinstance(obj["input"], str):
        return obj["input"]
    # fallback: join context + question/query if present
    ctx = obj.get("context") or obj.get("passage") or ""
    q = obj.get("question") or obj.get("query") or ""
    if ctx or q:
        return (str(ctx) + "\n" + str(q)).strip()
    # last resort: stringify the whole object (not ideal but unblocks)
    return json.dumps(obj, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser(description="Run a small local HF model for prompts or JSONL batches.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path (default: %(default)s)")
    ap.add_argument("--prompt", default=None, help="Single prompt to run")
    ap.add_argument("--in-jsonl", dest="in_jsonl", default=None, help="Read prompts from JSONL (tries 'prompt', 'input', or 'context'+'question')")
    ap.add_argument("--out-jsonl", dest="out_jsonl", default=None, help="Write predictions to this JSONL (requires --in-jsonl)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--stop-words", default="", help="Comma-separated stop words to truncate generations")
    args = ap.parse_args()

    stop = [s for s in args.stop_words.split(",") if s] if args.stop_words else None

    llm = LocalHFModel(model_name_or_path=args.model)

    if args.prompt is not None:
        out = llm(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, stop=stop)
        print(out["text"][0])
        return

    if args.in_jsonl:
        if not args.out_jsonl:
            raise SystemExit("--out-jsonl is required when using --in-jsonl")
        preds = []
        for obj in read_jsonl(args.in_jsonl):
            prompt = extract_prompt(obj)
            out = llm(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, stop=stop)
            new_obj = dict(obj)
            new_obj["prediction"] = out["text"][0]
            preds.append(new_obj)
        write_jsonl(args.out_jsonl, preds)
        print(f"Wrote {len(preds)} predictions to {args.out_jsonl}")
        return

    ap.print_help()

if __name__ == "__main__":
    main()
