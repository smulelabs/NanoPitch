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
    "num_workers": 0,
    "snr_range": [-5.0, 20.0], "p_clean": 0.0, "snr_bias": 1.0,
    "w_vad": 0.1, "w_pitch": 1.0,
    "vad_loss": "bce", "vad_pos_weight": 2.3, "vad_focal_gamma": 2.0,
    "balanced_sampling": False,
    "pitch_sigma": 1.2, "pitch_mask": "vad",
    "scheduler": "constant", "vad_target": "f0", "eval_vdr": "pitch",
    "augment": "none",
    "freq_mask_param": 4, "n_freq_masks": 2,
    "time_mask_param": 10, "n_time_masks": 2,
    "lr_t0": 10, "patience": 0,
    "curriculum": False,
    "curriculum_vad_epochs": 10, "curriculum_pitch_epochs": 20,
    "resume": None,
}

# Which args are interesting to show in the "Key args" column. Paths/device
# are excluded — they don't affect what the model learned.
INTERESTING_ARGS = [
    "cond_size", "gru_size", "seq_len", "epochs", "batch_size", "lr",
    "w_vad", "w_pitch",
    "vad_loss", "vad_pos_weight", "vad_focal_gamma",
    "balanced_sampling",
    "scheduler", "lr_t0",
    "vad_target", "augment", "p_clean", "snr_bias", "snr_range",
    "freq_mask_param", "n_freq_masks", "time_mask_param", "n_time_masks",
    "pitch_sigma", "pitch_mask",
    "curriculum", "curriculum_vad_epochs", "curriculum_pitch_epochs",
    "eval_vdr", "resume",
]


def _resume_base_run(resume_path):
    """Given a --resume path like './runs/foo/checkpoints/best.pth', return
    the base run name ('foo'). Fall back to the file basename."""
    p = os.path.normpath(resume_path)
    parts = p.split(os.sep)
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i > 0:
            return parts[i - 1]
    return os.path.basename(p)


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return " ".join(fmt_val(x) for x in v)
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def _parse_metric_val(cell_str):
    """Extract the numeric value from a cell like '95.0 (+3.9)' or '95.0'."""
    m = re.match(r"\s*([\d.]+)", cell_str.strip())
    return float(m.group(1)) if m else None


def _fmt_with_delta(val, base_val):
    """Format a metric value with its signed delta from baseline."""
    if base_val is None or val is None:
        return f"{val:.1f}" if val is not None else "—"
    d = val - base_val
    sign = "+" if d >= 0 else ""
    return f"{val:.1f} ({sign}{d:.1f})"


def _extract_baseline_from_runs(results_md_path):
    """Return {vad, rt_rpa, rt_vdr, rt_med} from the 'baseline' Runs row, or None."""
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return None
    _, _, rows = sl
    for row in rows:
        m = DATA_ROW_RE.match(row)
        if m and m.group(1) == "baseline":
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) >= 8:
                return {
                    "vad":    _parse_metric_val(cells[4]),
                    "rt_rpa": _parse_metric_val(cells[5]),
                    "rt_vdr": _parse_metric_val(cells[6]),
                    "rt_med": _parse_metric_val(cells[7]),
                }
    return None


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
        if k == "resume" and v:
            out.append(f"`--resume`={_resume_base_run(v)}")
        else:
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
    """Collapse parsed per-condition data into the summary-row scalars.
    Tracked metrics are realtime-only; VAD Acc is decoder-independent."""
    rt = parsed["realtime"]["overall"]
    return {"vad": rt[0],
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


def make_summary_row(number, name, args_diff, note, metrics, baseline_metrics=None):
    ka = ", ".join(args_diff) if args_diff else "defaults"
    note_cell = note or "_tbd_"
    if baseline_metrics and name != "baseline":
        vad_s = _fmt_with_delta(metrics["vad"],    baseline_metrics["vad"])
        rpa_s = _fmt_with_delta(metrics["rt_rpa"], baseline_metrics["rt_rpa"])
        vdr_s = _fmt_with_delta(metrics["rt_vdr"], baseline_metrics["rt_vdr"])
        med_s = _fmt_with_delta(metrics["rt_med"], baseline_metrics["rt_med"])
    else:
        vad_s = f"{metrics['vad']:.1f}"
        rpa_s = f"{metrics['rt_rpa']:.1f}"
        vdr_s = f"{metrics['rt_vdr']:.1f}"
        med_s = f"{metrics['rt_med']:.1f}"
    return (f"| {number} | `{name}` | {ka} | {note_cell} | "
            f"{vad_s} | {rpa_s} | {vdr_s} | {med_s} |")


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


def _section_slice(lines, section_title):
    """Return (header_end, data_end, data_rows_list) for the given section,
    or None if the section doesn't exist."""
    bounds = _section_bounds(lines, section_title)
    if bounds is None:
        return None
    start, end = bounds
    header_end = start
    while header_end < end and not DATA_ROW_RE.match(lines[header_end]):
        header_end += 1
    data_end = header_end
    while data_end < end and DATA_ROW_RE.match(lines[data_end]):
        data_end += 1
    data_rows = [l.rstrip("\n") for l in lines[header_end:data_end]]
    return header_end, data_end, data_rows


def _write_section(results_md_path, section_title, new_data_rows):
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, section_title)
    if sl is None:
        sys.exit(f"Section '## {section_title}' not found in {results_md_path}")
    header_end, data_end, _ = sl
    new_lines = (lines[:header_end]
                 + [r + "\n" for r in new_data_rows]
                 + lines[data_end:])
    with open(results_md_path, "w") as f:
        f.writelines(new_lines)


def _runs_name_to_number(results_md_path):
    """Map run-name → integer row # from the Runs section (post-renumber)."""
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return {}
    _, _, rows = sl
    out = {}
    num_re = re.compile(r"^\|\s*(\d+)\s*\|")
    for row in rows:
        name_m = DATA_ROW_RE.match(row)
        num_m = num_re.match(row)
        if name_m and num_m:
            out[name_m.group(1)] = int(num_m.group(1))
    return out


def _renumber_from_map(data_rows, name_to_num):
    """Rewrite the leading '| N |' of each row using name_to_num. Unknown
    names fall back to '—'."""
    out = []
    for row in data_rows:
        m = DATA_ROW_RE.match(row)
        name = m.group(1) if m else None
        num = name_to_num.get(name)
        label = str(num) if num is not None else "—"
        out.append(re.sub(r"^\|\s*[^|]*\|", f"| {label} |", row, count=1))
    return out


def reformat_deltas(results_md_path):
    """Rewrite every Runs row to include (±delta) annotations from baseline.
    Baseline row itself shows plain numbers. Idempotent — safe to re-run."""
    baseline = _extract_baseline_from_runs(results_md_path)
    if baseline is None:
        sys.exit("No 'baseline' row found in Runs table — cannot compute deltas.")
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        sys.exit("'## Runs' section not found.")
    _, _, rows = sl
    new_rows = []
    for row in rows:
        m = DATA_ROW_RE.match(row)
        if not m:
            new_rows.append(row)
            continue
        name = m.group(1)
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 8:
            new_rows.append(row)
            continue
        vad_v = _parse_metric_val(cells[4])
        rpa_v = _parse_metric_val(cells[5])
        vdr_v = _parse_metric_val(cells[6])
        med_v = _parse_metric_val(cells[7])
        if name == "baseline":
            vad_s, rpa_s, vdr_s, med_s = (f"{vad_v:.1f}", f"{rpa_v:.1f}",
                                           f"{vdr_v:.1f}", f"{med_v:.1f}")
        else:
            vad_s = _fmt_with_delta(vad_v, baseline["vad"])
            rpa_s = _fmt_with_delta(rpa_v, baseline["rt_rpa"])
            vdr_s = _fmt_with_delta(vdr_v, baseline["rt_vdr"])
            med_s = _fmt_with_delta(med_v, baseline["rt_med"])
        new_rows.append(f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                        f"{vad_s} | {rpa_s} | {vdr_s} | {med_s} |")
    _write_section(results_md_path, "Runs", new_rows)
    print(f"Reformatted {len(new_rows)} rows with delta annotations.")


def upsert_row(results_md_path, section_title, run_name, new_row,
               sort_by_last_col_desc=False, renumber_from=None):
    """Replace any existing row matching `run_name` inside
    `## <section_title>`, else append. If `sort_by_last_col_desc` is true,
    sort data rows by the last numeric column (descending) before renumbering.
    If `renumber_from` is a dict(name→int), use it to renumber; otherwise
    renumber sequentially 1..N."""
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, section_title)
    if sl is None:
        sys.exit(f"Section '## {section_title}' not found in {results_md_path}")
    _, _, data_rows = sl

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
                return _parse_metric_val(cells[-1]) or float("-inf")
            except (ValueError, IndexError):
                return float("-inf")
        data_rows.sort(key=last_val, reverse=True)

    # Pin baseline to the top of the Runs section (not per-condition).
    if section_title == "Runs":
        base = [r for r in data_rows
                if DATA_ROW_RE.match(r) and DATA_ROW_RE.match(r).group(1) == "baseline"]
        rest = [r for r in data_rows
                if not (DATA_ROW_RE.match(r) and DATA_ROW_RE.match(r).group(1) == "baseline")]
        data_rows = base + rest

    if renumber_from is not None:
        data_rows = _renumber_from_map(data_rows, renumber_from)
    else:
        data_rows = _renumber(data_rows)

    _write_section(results_md_path, section_title, data_rows)
    action = "Updated" if replaced else "Appended"
    print(f"{action} '{run_name}' in section '{section_title}'")


def delete_row(results_md_path, run_name):
    """Remove `run_name` from Runs + Per-condition rtRPA, then renumber:
    Runs sequentially 1..N, per-condition rows using the new Runs map."""
    removed_any = False
    for section in ("Runs", "Per-condition rtRPA"):
        with open(results_md_path, "r") as f:
            lines = f.readlines()
        sl = _section_slice(lines, section)
        if sl is None:
            continue
        _, _, data_rows = sl
        new_rows = []
        removed_here = False
        for row in data_rows:
            m = DATA_ROW_RE.match(row)
            if m and m.group(1) == run_name:
                removed_here = True
                continue
            new_rows.append(row)
        if not removed_here:
            continue
        removed_any = True
        if section == "Runs":
            new_rows = _renumber(new_rows)
        _write_section(results_md_path, section, new_rows)
        print(f"Removed '{run_name}' from section '{section}'")

    # Re-sync per-condition numbering against the updated Runs table.
    name_to_num = _runs_name_to_number(results_md_path)
    with open(results_md_path, "r") as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Per-condition rtRPA")
    if sl is not None:
        _, _, data_rows = sl
        _write_section(results_md_path, "Per-condition rtRPA",
                       _renumber_from_map(data_rows, name_to_num))

    if not removed_any:
        print(f"No row matching '{run_name}' found.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="path to run directory containing checkpoints/")
    p.add_argument("--data-dir", default=None,
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
    p.add_argument("--delete", default=None, metavar="NAME",
                   help="remove the named run from both tables and exit")
    p.add_argument("--reformat-deltas", action="store_true",
                   help="rewrite the Runs table so every row shows (±delta) from "
                        "the 'baseline' row; then exit")
    cli = p.parse_args()

    results_md = cli.results_md or os.path.join(
        os.path.dirname(__file__), "..", "RESULTS.MD")
    results_md = os.path.abspath(results_md)

    if cli.delete:
        delete_row(results_md, cli.delete)
        return

    if cli.reformat_deltas:
        reformat_deltas(results_md)
        return

    if not cli.run_dir or not cli.data_dir:
        sys.exit("--run-dir and --data-dir are required (unless using --delete).")

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
    baseline_metrics = _extract_baseline_from_runs(results_md)
    summary_row = make_summary_row("_", name, diff_args(args), cli.note, metrics,
                                   baseline_metrics)
    per_cond_row = make_per_cond_row("_", name, per_cond)
    print(summary_row)
    print(per_cond_row)

    if cli.print_only:
        return

    # Runs section: sequential renumber. Then reload to get the authoritative
    # name→number map and use it to number the per-condition row.
    upsert_row(results_md, "Runs", name, summary_row)
    name_to_num = _runs_name_to_number(results_md)
    upsert_row(results_md, "Per-condition rtRPA", name, per_cond_row,
               sort_by_last_col_desc=True, renumber_from=name_to_num)


if __name__ == "__main__":
    main()
