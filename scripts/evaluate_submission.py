#!/usr/bin/env python3
"""
Evaluate a single student submission and write metrics to a JSON file.

Usage:
    python scripts/evaluate_submission.py \
        --submission submissions/alice \
        --data-dir data \
        --output-json results/alice.json

Hidden test set:
    Set environment variable HIDDEN_TEST_NPZ to a local path to override
    the public data/test.npz used for evaluation.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_checkpoint(submission_dir: Path) -> Path:
    for name in ("weights.pth", "best.pth", "checkpoint.pth"):
        p = submission_dir / name
        if p.exists():
            return p
    pths = list(submission_dir.glob("*.pth"))
    if len(pths) == 1:
        return pths[0]
    raise FileNotFoundError(
        f"No .pth checkpoint found in {submission_dir}. "
        "Expected weights.pth, best.pth, or checkpoint.pth."
    )


def load_submission_meta(submission_dir: Path) -> dict:
    yaml_path = submission_dir / "submission.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing submission.yaml in {submission_dir}")
    # Minimal YAML parsing (no external dependency needed for simple key: value pairs)
    meta = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def run_evaluate(checkpoint: Path, data_dir: Path, output_json: Path) -> dict:
    """
    Call training/evaluate.py as a subprocess.

        python evaluate.py --checkpoint <path> --data-dir <dir> --json <out.json>
    """
    evaluate_script = REPO_ROOT / "training" / "evaluate.py"
    if not evaluate_script.exists():
        raise FileNotFoundError(f"evaluate.py not found at {evaluate_script}")

    # evaluate.py writes JSON to the file given by --json.
    # Use a sibling temp file so we don't collide with the final output_json.
    raw_json = output_json.with_suffix(".raw.json")

    cmd = [
        sys.executable,
        str(evaluate_script),
        "--checkpoint", str(checkpoint),
        "--data-dir", str(data_dir),
        "--json", str(raw_json),
    ]

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Always surface stdout/stderr so CI logs are readable.
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py exited with code {result.returncode}")

    if not raw_json.exists():
        raise RuntimeError(
            f"evaluate.py succeeded but did not write {raw_json}. "
            "Check that --json is a supported flag."
        )

    with open(raw_json) as f:
        metrics = json.load(f)

    raw_json.unlink()   # clean up temp file
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a NanoPitch student submission.")
    parser.add_argument("--submission", required=True,
                        help="Path to student submission directory (e.g. submissions/alice)")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing training data and the default test.npz")
    parser.add_argument("--output-json", required=True,
                        help="Where to write the JSON results file")
    args = parser.parse_args()

    submission_dir = Path(args.submission).resolve()
    output_json = Path(args.output_json).resolve()

    # --- 1. Load submission metadata ---
    meta = load_submission_meta(submission_dir)
    student_name = meta.get("name", submission_dir.name)
    note = meta.get("note", "")
    print(f"Evaluating submission from: {student_name}")
    if note:
        print(f"Note: {note}")

    # --- 2. Find checkpoint ---
    checkpoint = find_checkpoint(submission_dir)
    print(f"Checkpoint: {checkpoint}")

    # --- 3. Resolve data directory ---
    data_dir = Path(args.data_dir).resolve()

    hidden_test = os.environ.get("HIDDEN_TEST_NPZ", "").strip()
    if hidden_test:
        # Use a temporary data directory so we don't overwrite the original.
        tmp = tempfile.mkdtemp(prefix="nanopitch_eval_")
        tmp_data = Path(tmp)
        # Copy everything from the real data dir (clean.npz, noise.npz, etc.)
        for f in data_dir.iterdir():
            shutil.copy2(f, tmp_data / f.name)
        # Override test.npz with the hidden set.
        hidden_path = Path(hidden_test)
        if not hidden_path.exists():
            raise FileNotFoundError(f"HIDDEN_TEST_NPZ points to non-existent file: {hidden_path}")
        shutil.copy2(hidden_path, tmp_data / "test.npz")
        data_dir = tmp_data
        print(f"Using hidden test set: {hidden_path}")
    else:
        print(f"Using public test set from: {data_dir}")

    # --- 4. Run evaluation ---
    output_json.parent.mkdir(parents=True, exist_ok=True)
    metrics = run_evaluate(checkpoint, data_dir, output_json)

    # --- 5. Attach submission metadata to results ---
    metrics["student_name"] = student_name
    metrics["note"] = note
    metrics["submission_dir"] = submission_dir.name
    metrics["checkpoint"] = str(checkpoint.name)

    # --- 6. Write results ---
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults written to: {output_json}")

    # Print a short summary to stdout for CI logs.
    print("\n=== Summary ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
