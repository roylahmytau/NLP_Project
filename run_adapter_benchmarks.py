#!/usr/bin/env python3
"""
Batch runner to evaluate all adapters in the adapters folder.

For each detected adapter directory, runs:
- Perplexity on Penn Treebank (via benchmark_model.py default)
- GLUE (SST-2, MRPC, QNLI) using existing logic in benchmark_model.py

Aggregates results into one JSON file mapping adapter name -> results.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional
import threading



REPO_ROOT = "/workspace/NLP_Project"
ADAPTERS_ROOT_DEFAULT = os.path.join(REPO_ROOT, "adapters")
OUTPUTS_DIR_DEFAULT = os.path.join(REPO_ROOT, "outputs")
BENCHMARK_SCRIPT = os.path.join(REPO_ROOT, "benchmark_model.py")


def is_adapter_dir(path: str) -> bool:
    """Heuristically decide whether a directory is a PEFT adapter directory."""
    if not os.path.isdir(path):
        return False
    try:
        entries = set(os.listdir(path))
    except Exception:
        return False
    # Common PEFT adapter files
    indicators = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "config.json",
    }
    return any(ind in entries for ind in indicators)


def find_adapter_dirs(root: str) -> List[str]:
    """Find adapter directories under root (searches recursively)."""
    adapter_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if is_adapter_dir(dirpath):
            adapter_dirs.append(dirpath)
    # Sort for stable ordering
    adapter_dirs.sort()
    return adapter_dirs


def run_benchmark(adapter_dir: str,
                  benchmark: str,
                  model_name: str,
                  batch_size: int,
                  max_samples: int,
                  perplexity_split: str,
                  glue_task: str,
                  tmp_output: str,
                  live_log_file: Optional[str] = None) -> int:
    """Invoke benchmark_model.py for a single benchmark and adapter.

    Streams stdout/stderr to console and optionally to a live log file.
    """
    cmd = [
        sys.executable, BENCHMARK_SCRIPT,
        "--adapter_path", adapter_dir,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--max_samples", str(max_samples),
        "--benchmark", benchmark,
        "--output_file", tmp_output,
    ]
    if benchmark == "perplexity":
        cmd += ["--perplexity_split", perplexity_split]
    if benchmark == "glue":
        cmd += ["--glue_task", glue_task]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump_output():
        if process.stdout is None:
            return
        # Open once if live log requested
        log_fh = open(live_log_file, "a", encoding="utf-8") if live_log_file else None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_fh is not None:
                    log_fh.write(line)
                    log_fh.flush()
        finally:
            if log_fh is not None:
                log_fh.flush()
                log_fh.close()

    t = threading.Thread(target=pump_output, daemon=True)
    t.start()
    ret = process.wait()
    t.join(timeout=5)
    if ret != 0:
        print(f"Benchmark command failed ({benchmark}) for {adapter_dir}")
    return ret


def load_results(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Run perplexity and GLUE for all adapters and aggregate results")
    parser.add_argument("--adapters_root", type=str, default=ADAPTERS_ROOT_DEFAULT, help="Root folder containing adapters")
    parser.add_argument("--outputs_dir", type=str, default=OUTPUTS_DIR_DEFAULT, help="Directory to store temp and final results")
    parser.add_argument("--output_file", type=str, default=None, help="Output file (live-updated NDJSON log of events/results)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--needle_size", type=str, default="2048", choices=["2048", "32768", "131072"], help="Needle size (unused for these benchmarks but required by script)")
    parser.add_argument("--needle_type", type=str, default="qa_1",
                        choices=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"],
                        help="Needle type (unused for these benchmarks but required by script)")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size for GLUE prompts")
    parser.add_argument("--max_samples", type=int, default=100, help="Max validation samples per task / PTB lines")
    parser.add_argument("--perplexity_split", type=str, default="validation", choices=["train", "validation", "test"], help="Penn Treebank split")
    parser.add_argument("--glue_task", type=str, default="all", choices=["sst2", "mrpc", "qnli", "all"], help="GLUE tasks to evaluate")
    args = parser.parse_args()

    adapters_root = os.path.abspath(args.adapters_root)
    outputs_dir = os.path.abspath(args.outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    if args.output_file:
        final_output = os.path.abspath(args.output_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(outputs_dir, f"adapter_benchmarks_{ts}.json")

    adapter_dirs = find_adapter_dirs(adapters_root)
    if not adapter_dirs:
        print(f"No adapters found under {adapters_root}")
        with open(final_output, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        print(f"Wrote empty results to {final_output}")
        return

    print(f"Found {len(adapter_dirs)} adapter directories. Running benchmarks...")

    # Open stream file for live updates
    stream_path = final_output
    # Start with a header line for readability
    with open(stream_path, "w", encoding="utf-8") as stream:
        stream.write(f"# Adapter Benchmarks Stream - {datetime.now().isoformat()}\n")
        stream.flush()

    aggregated: Dict[str, Dict] = {}

    for adapter_dir in adapter_dirs:
        adapter_name = os.path.basename(adapter_dir.rstrip(os.sep))
        print(f"\n=== Adapter: {adapter_name} ===")
        with open(stream_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "adapter_start", "adapter": adapter_name, "path": adapter_dir}) + "\n")
            stream.flush()

        # Temp outputs
        tmp_perp = os.path.join(outputs_dir, f"{adapter_name}_perplexity.json")
        tmp_glue = os.path.join(outputs_dir, f"{adapter_name}_glue.json")

        # Run Perplexity
        with open(stream_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "benchmark_start", "adapter": adapter_name, "benchmark": "perplexity"}) + "\n")
            stream.flush()
        code1 = run_benchmark(
            adapter_dir=adapter_dir,
            benchmark="perplexity",
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            perplexity_split=args.perplexity_split,
            glue_task=args.glue_task,
            tmp_output=tmp_perp,
            live_log_file=stream_path,
        )

        # Run GLUE
        with open(stream_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "benchmark_start", "adapter": adapter_name, "benchmark": "glue"}) + "\n")
            stream.flush()
        code2 = run_benchmark(
            adapter_dir=adapter_dir,
            benchmark="glue",
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            perplexity_split=args.perplexity_split,
            glue_task=args.glue_task,
            tmp_output=tmp_glue,
            live_log_file=stream_path,
        )

        res_perp = load_results(tmp_perp) if code1 == 0 else None
        res_glue = load_results(tmp_glue) if code2 == 0 else None

        entry: Dict[str, Dict] = {
            "adapter_name": adapter_name,
            "adapter_path": adapter_dir,
        }
        if res_perp and isinstance(res_perp.get("results"), dict):
            entry.update({"perplexity": res_perp["results"].get("perplexity")})
            with open(stream_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "event": "result",
                    "adapter": adapter_name,
                    "benchmark": "perplexity",
                    "result": entry["perplexity"],
                }) + "\n")
                stream.flush()
        else:
            entry.update({"perplexity": None})

        if res_glue and isinstance(res_glue.get("results"), dict):
            entry.update({"glue": res_glue["results"].get("glue")})
            with open(stream_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "event": "result",
                    "adapter": adapter_name,
                    "benchmark": "glue",
                    "result": entry["glue"],
                }) + "\n")
                stream.flush()
        else:
            entry.update({"glue": None})

        aggregated[adapter_name] = entry

    # Append a compact summary as a final event line
    with open(stream_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "summary", "results": aggregated}) + "\n")
        stream.write(json.dumps({"event": "done"}) + "\n")
        stream.flush()

    print(f"\nAll done. Live log (with results) at {final_output}")


if __name__ == "__main__":
    main()


