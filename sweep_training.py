#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime
import argparse


DEFAULTS = {
    "epochs": 20,
    "batch_size": 1,
    "learning_rate": 0.001,
    "adapter_r": 5,
    "adapter_layers": 5,
}


# SWEEP_VALUES = {
#     "epochs": [0, 1, 5, 10, 20],
#     "batch_size": [1, 2, 4],
#     "learning_rate": [0.01, 0.001, 0.0001],
#     "adapter_r": [5, 8, 11],
#     "adapter_layers": [5, 10, 15],
# }

SWEEP_VALUES = {
    "epochs": [20],
    "batch_size": [1],
    "learning_rate": [ 0.001],
    "adapter_r": [5],
    "adapter_layers": [5],
}


def format_lr(value: float) -> str:
    # Represent learning rate compactly for filenames
    # 0.001 -> 1e-3, 0.0001 -> 1e-4, 0.01 -> 1e-2
    return f"{value:.0e}".replace("+", "").replace("-0", "-")


def build_run_name(overrides: dict, needle_size: str, needle_type: str) -> str:
    epochs = overrides.get("epochs", DEFAULTS["epochs"])
    batch = overrides.get("batch_size", DEFAULTS["batch_size"])
    lr = overrides.get("learning_rate", DEFAULTS["learning_rate"])
    r = overrides.get("adapter_r", DEFAULTS["adapter_r"])
    layers = overrides.get("adapter_layers", DEFAULTS["adapter_layers"])
    genqa = overrides.get("use_generated_qa", False)
    genqa_tag = "genQA" if genqa else "noGenQA"
    return f"{needle_type}_{needle_size}_ep{epochs}_bs{batch}_lr{format_lr(lr)}_r{r}_layers{layers}_{genqa_tag}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_single(overrides: dict, logs_root: str, adapters_root: str, needle_size: str, needle_type: str) -> int:
    run_name = build_run_name(overrides, needle_size, needle_type)
    adapter_dir = os.path.join(adapters_root, run_name)
    log_dir = os.path.join(logs_root, run_name)
    ensure_dir(adapter_dir)
    ensure_dir(log_dir)

    # Compose command
    cmd = [
        "python", "run_training_and_benchmarks.py",
        "--needle_size", str(needle_size),
        "--needle_type", str(needle_type),
        "--epochs", str(overrides.get("epochs", DEFAULTS["epochs"])),
        "--batch_size", str(overrides.get("batch_size", DEFAULTS["batch_size"])),
        "--learning_rate", str(overrides.get("learning_rate", DEFAULTS["learning_rate"])),
        "--adapter_r", str(overrides.get("adapter_r", DEFAULTS["adapter_r"])),
        "--adapter_layers", str(overrides.get("adapter_layers", DEFAULTS["adapter_layers"])),
        "--output_dir", adapter_dir,
        "--keep_adapter",
    ]

    if overrides.get("use_generated_qa", False):
        # Let the underlying training script auto-resolve generated QA dir
        cmd += ["--use_generated_qa"]

    # Log files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = os.path.join(log_dir, f"stdout_{ts}.log")
    stderr_path = os.path.join(log_dir, f"stderr_{ts}.log")

    print(f"\n=== Running: {run_name} ===")
    print("Command:", " ".join(cmd))
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        process = subprocess.run(cmd, stdout=out, stderr=err)
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument("needle_size", choices=["2048", "32768", "131072"], help="Needle size to train on")
    parser.add_argument("--needle_type", default="qa_1", choices=[
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_single_1", "niah_single_2", "niah_single_3", "qa_1"
    ], help="Needle data type")
    parser.add_argument("--use_generated_qa", action="store_true", help="Enable generated QA augmentation in training runs")
    args = parser.parse_args()

    # Organize outputs by size/type for clarity
    logs_root = os.path.abspath(os.path.join("logs", f"{args.needle_type}_{args.needle_size}"))
    adapters_root = os.path.abspath(os.path.join("adapters", f"{args.needle_type}_{args.needle_size}"))
    ensure_dir(logs_root)
    ensure_dir(adapters_root)

    failures = []

    # Single sweep according to the provided flag
    use_genqa = args.use_generated_qa

    # Run the pure defaults ONCE
    default_overrides = DEFAULTS.copy()
    default_overrides["use_generated_qa"] = use_genqa
    code = run_single(default_overrides, logs_root, adapters_root, args.needle_size, args.needle_type)
    if code != 0:
        failures.append(("default", f"genQA={use_genqa}", code))

    # For each hyperparameter, run only the non-default values (one-at-a-time deviations)
    for key, values in SWEEP_VALUES.items():
        default_val = DEFAULTS[key]
        for value in values:
            if value == default_val:
                continue  # skip duplicate default configuration
            overrides = DEFAULTS.copy()
            overrides[key] = value
            overrides["use_generated_qa"] = use_genqa
            code = run_single(overrides, logs_root, adapters_root, args.needle_size, args.needle_type)
            if code != 0:
                failures.append((f"{key}={value} (genQA={use_genqa})", value, code))

    if failures:
        print("\nSome runs failed:")
        for key, value, code in failures:
            print(f"- {key}={value} -> exit {code}")
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()


