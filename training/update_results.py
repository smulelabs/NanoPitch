"""
update_results.py
=================

Reads a completed run's checkpoint + runs evaluate.py, then upserts a row
into RESULTS.MD.

Usage:
    python update_results.py --run-dir ./runs/my_first_model \
                             --data-dir ../data \
                             [--checkpoint best_macro_rpa.pth] \
                             [--name baseline] \
                             [--note "short description"] \
                             [--results-md ../RESULTS.MD]

The checkpoint is expected to contain the training args (saved automatically
since the 'args' key was added to ckpt in train.py). Older checkpoints
without 'args' will still work — key-args column is left blank.
"""
import argparse
import os
import re
import subprocess
import sys

import torch


# Canonical argparse defaults (must match train.py). If you add or rename a
# flag in train.py, update this dict so "Key args" diffs stay correct.
TRAIN_DEFAULTS = {
    "cond_size": 64, "gru_size": 96,
    "epochs": 50, "batch_size": 32, "lr": 1e-3, "seq_len": 200,
    "num_workers": 0, "amp": False,
    "snr_range": [-5.0, 20.0], "p_clean": 0.0, "snr_bias": 1.0,
    "w_vad": 0.1, "w_pitch": 1.0,
    "pitch_sigma": 1.2, "pitch_mask": "vad",
    "scheduler": "constant", "vad_target": "f0", "eval_vdr": "pitch",
    "augment": "none",
    "lr_t0": 10, "patience": 0,
    "curriculum": False,
    "curriculum_vad_epochs": 10, "curriculum_pitch_epochs": 20,
}

# Which args are interesting to show in the "Key args" column. Paths/device
# are excluded — they don't affect what the model learned.
INTERESTING_ARGS = [
    "cond_size", "gru_size", "seq_len", "epochs", "batch_size", "lr",
    "w_vad", "w_pitch", "amp",
    "scheduler", "lr_t0",
    "vad_target", "augment", "p_clean", "snr_bias", "snr_range",
    "pitch_sigma", "pitch_mask",
    "curriculum", "curriculum_vad_epochs", "curriculum_pitch_epochs",
    "eval_vdr",
]


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return " ".join(fmt_val(x) for x in v)
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def diff_args(args):
    """Return list of 'flag=value' strings for non-default interesting args."""
    out = []
    for k in INTERESTING_ARGS:
        if k not in args:
            continue
        v = args[k]
        dv = TRAIN_DEFAULTS.get(k)
        if v == dv:
            continue
        out.append(f"`--{k.replace('_', '-')}`={fmt_val(v)}")
    return out


def run_evaluate(checkpoint_path, data_dir):
    """Run evaluate.py and return its stdout."""
    script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    cmd = [sys.executable, "-u", script,
           "--checkpoint", checkpoint_path, "--data-dir", data_dir]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit(f"evaluate.py failed (exit {res.returncode})")
    return res.stdout


# Per-condition / overall rows look like:
#   -5 dB          89.6%      9.9%     94.9%     96.0%      5.1%      70.0
#   +0 dB          88.6%     10.6%     97.0%     97.0%      3.0%      12.4
#   clean          95.1%     12.0%     95.2%     95.2%      4.8%      13.1
#   overall        91.8%     10.9%     96.4%     96.6%      3.6%      28.7
ROW_RE = re.compile(
    r"^\s*(?P<cond>overall|clean|[-+]?\d+\s*dB)\s+"
    r"(\d+\.\d+)%\s+"   # VAD Acc
    r"(\d+\.\d+)%\s+"   # VDR
    r"(\d+\.\d+)%\s+"   # RPA
    r"(\d+\.\d+)%\s+"   # RCA
    r"(\d+\.\d+)%\s+"   # Gross
    r"(\d+\.\d+)\s*$"   # Med cents
)

# Per-condition order for the per-condition table.
CONDITIONS = ["-5 dB", "+0 dB", "+5 dB", "+10 dB", "+20 dB", "clean"]


def _normalize_cond(s):
    s = s.strip()
    if s == "clean" or s == "overall":
        return s
    # Normalise "-5 dB", "5 dB", "+5 dB" → "+5 dB" / "-5 dB".
    m = re.match(r"([-+]?)(\d+)\s*dB", s)
    if not m:
        return s
    sign = m.group(1) or "+"
    return f"{sign}{int(m.group(2))} dB"


def parse_eval(stdout):
    """Parse both offline and realtime tables into {condition: (vad, vdr, rpa,
    rca, gross, med)} dicts.

    evaluate.py prints Offline first, then Realtime. We split on the second
    'overall' occurrence."""
    rows = []  # list of (cond, tuple_of_floats)
    for line in stdout.splitlines():
        m = ROW_RE.match(line)
        if m:
            vals = tuple(float(m.group(i)) for i in range(2, 8))
            rows.append((_normalize_cond(m.group("cond")), vals))
    # Split into offline / realtime at the boundary between two 'overall's.
    overall_idxs = [i for i, (c, _) in enumerate(rows) if c == "overall"]
    if len(overall_idxs) < 2:
        sys.exit(f"Could not find two 'overall' rows in evaluate output "
                 f"(found {len(overall_idxs)}).")
    off_rows = dict(rows[: overall_idxs[0] + 1])
    rt_rows = dict(rows[overall_idxs[0] + 1: overall_idxs[1] + 1])
    return {"offline": off_rows, "realtime": rt_rows}


def summary_metrics(parsed):
    """Collapse parsed per-condition data into the summary-row scalars."""
    off = parsed["offline"]["overall"]
    rt = parsed["realtime"]["overall"]
    # VAD Acc is decoder-independent → same value in both tables.
    return {"vad": off[0],
            "off_rpa": off[2], "off_vdr": off[1], "off_med": off[5],
            "rt_rpa": rt[2], "rt_vdr": rt[1], "rt_med": rt[5]}


def per_condition_rpa(parsed):
    """Realtime RPA per SNR condition (percent). Missing conditions → None."""
    rt = parsed["realtime"]
    out = {}
    for cond in CONDITIONS:
        vals = rt.get(cond)
        out[cond] = vals[2] if vals is not None else None
    out["macro"] = rt["overall"][2]
    return out


def make_summary_row(number, name, args_diff, note, metrics):
    ka = ", ".join(args_diff) if args_diff else "defaults"
    note_cell = note or "_tbd_"
    return (f"| {number} | `{name}` | {ka} | {note_cell} | "
            f"{metrics['vad']:.1f} | {metrics['off_rpa']:.1f} | "
            f"{metrics['off_med']:.1f} | {metrics['rt_rpa']:.1f} | "
            f"{metrics['rt_vdr']:.1f} | {metrics['rt_med']:.1f} |")


def make_per_cond_row(number, name, per_cond):
    def cell(v):
        return f"{v:.1f}" if v is not None else "—"
    return (f"| {number} | `{name}` | "
            f"{cell(per_cond['-5 dB'])} | {cell(per_cond['+0 dB'])} | "
            f"{cell(per_cond['+5 dB'])} | {cell(per_cond['+10 dB'])} | "
            f"{cell(per_cond['+20 dB'])} | {cell(per_cond['clean'])} | "
            f"{cell(per_cond['macro'])} |")


def _section_bounds(lines, section_title):
    """Return [start, end) indices for lines inside `## <section_title>`
    (exclusive of the header itself, up to but not including the next `## `)."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"## {section_title}":
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return start, end


DATA_ROW_RE = re.compile(r"^\|\s*[^|]*\|\s*`([^`]+)`\s*\|")


def _renumber(data_rows):
    """Rewrite the leading '| N |' of each row to be 1..len sequentially."""
    out = []
    for i, row in enumerate(data_rows, start=1):
        out.append(re.sub(r"^\|\s*[^|]*\|", f"| {i} |", row, count=1))
    return out


def upsert_row(results_md_path, section_title, run_name, new_row,
               sort_by_last_col_desc=False):
    """Replace any existing row matching `run_name` inside
    `## <section_title>`, else append. Always renumbers rows 1..N. If
    `sort_by_last_col_desc` is true, sort data rows by the last numeric
    column (descending) before renumbering."""
    with open(results_md_path, "r") as f:
        lines = f.readlines()

    bounds = _section_bounds(lines, section_title)
    if bounds is None:
        sys.exit(f"Section '## {section_title}' not found in {results_md_path}")
    start, end = bounds

    # Split the section into: header/separator, data rows, trailing blanks.
    header_end = start
    while header_end < end and not DATA_ROW_RE.match(lines[header_end]):
        header_end += 1
    data_end = header_end
    while data_end < end and DATA_ROW_RE.match(lines[data_end]):
        data_end += 1

    data_rows = [l.rstrip("\n") for l in lines[header_end:data_end]]

    # Replace or append.
    replaced = False
    for i, row in enumerate(data_rows):
        m = DATA_ROW_RE.match(row)
        if m and m.group(1) == run_name:
            data_rows[i] = new_row
            replaced = True
            break
    if not replaced:
        data_rows.append(new_row)

    if sort_by_last_col_desc:
        def last_val(row):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            try:
                return float(cells[-1])
            except ValueError:
                return float("-inf")
        data_rows.sort(key=last_val, reverse=True)

    data_rows = _renumber(data_rows)

    new_lines = (lines[:header_end]
                 + [r + "\n" for r in data_rows]
                 + lines[data_end:])
    with open(results_md_path, "w") as f:
        f.writelines(new_lines)
    action = "Updated" if replaced else "Appended"
    print(f"{action} '{run_name}' in section '{section_title}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="path to run directory containing checkpoints/")
    p.add_argument("--data-dir", required=True,
                   help="folder containing clean.npz, noise.npz, test.npz")
    p.add_argument("--checkpoint", default="best.pth",
                   help="checkpoint filename inside <run-dir>/checkpoints/")
    p.add_argument("--name", default=None,
                   help="row label in RESULTS.MD (default: run-dir basename)")
    p.add_argument("--note", default=None,
                   help="'Expectation / note' cell text")
    p.add_argument("--results-md", default=None,
                   help="path to RESULTS.MD (default: ../RESULTS.MD)")
    p.add_argument("--print-only", action="store_true",
                   help="print the markdown row to stdout, don't touch the file")
    cli = p.parse_args()

    ckpt_path = os.path.join(cli.run_dir, "checkpoints", cli.checkpoint)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) or {}
    if not args:
        print("Warning: checkpoint has no 'args' dict — 'Key args' column will be empty.")

    stdout = run_evaluate(ckpt_path, cli.data_dir)
    parsed = parse_eval(stdout)
    metrics = summary_metrics(parsed)
    per_cond = per_condition_rpa(parsed)

    name = cli.name or os.path.basename(os.path.normpath(cli.run_dir))
    summary_row = make_summary_row("_", name, diff_args(args), cli.note, metrics)
    per_cond_row = make_per_cond_row("_", name, per_cond)
    print(summary_row)
    print(per_cond_row)

    if cli.print_only:
        return

    results_md = cli.results_md or os.path.join(
        os.path.dirname(__file__), "..", "RESULTS.MD")
    results_md = os.path.abspath(results_md)
    upsert_row(results_md, "Runs", name, summary_row)
    upsert_row(results_md, "Per-condition rtRPA", name, per_cond_row,
               sort_by_last_col_desc=True)


if __name__ == "__main__":
    main()
