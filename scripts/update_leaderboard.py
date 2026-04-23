#!/usr/bin/env python3
"""
Rebuild LEADERBOARD.md from all JSON files in the results/ directory.

Usage:
    python scripts/update_leaderboard.py
    python scripts/update_leaderboard.py --results-dir results --output LEADERBOARD.md
"""

import argparse
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SNR_CONDITIONS = ["clean", "-5 dB", "+0 dB", "+5 dB", "+10 dB", "+20 dB"]


def get_condition(metrics: dict, condition: str, key: str):
    """Pull a value from the nested JSON structure."""
    return metrics.get(condition, {}).get(key)


def macro_avg(metrics: dict, key: str):
    """Average a metric across all SNR conditions. Returns None if no data."""
    vals = []
    for cond in SNR_CONDITIONS:
        v = get_condition(metrics, cond, key)
        if v is not None:
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def format_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def format_cents(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}¢"
    except (TypeError, ValueError):
        return str(v)


def build_table(entries: list[dict], sort_key: str, lower_is_better: bool) -> str:
    """
    Build a ranked markdown table for a single metric.
    sort_key: one of "rpa_clean", "rpa_macro", "ger_clean", "ger_macro"
    """
    arrow = "↓" if lower_is_better else "↑"

    label_map = {
        "rpa_clean":  f"RPA Clean {arrow}",
        "rpa_macro":  f"RPA Macro Avg {arrow}",
        "ger_clean":  f"Gross Err Clean {arrow}",
        "ger_macro":  f"Gross Err Macro Avg {arrow}",
    }
    primary_label = label_map[sort_key]

    # All tables share: Rank | Student | <primary> | Note
    # Plus supporting columns per metric type
    if sort_key == "rpa_clean":
        columns = [
            ("Rank",           None),
            ("Student",        lambda m: m.get("student_name") or "—"),
            (primary_label,    lambda m: format_pct(get_condition(m, "clean", "realtime_rpa"))),
            ("RPA +0 dB",      lambda m: format_pct(get_condition(m, "+0 dB", "realtime_rpa"))),
            ("RPA -5 dB",      lambda m: format_pct(get_condition(m, "-5 dB", "realtime_rpa"))),
            ("VAD Acc",        lambda m: format_pct(get_condition(m, "clean", "vad_acc"))),
            ("Median Err",     lambda m: format_cents(get_condition(m, "clean", "realtime_median_cents"))),
            ("Note",           lambda m: m.get("note") or "—"),
        ]
    elif sort_key == "rpa_macro":
        columns = [
            ("Rank",           None),
            ("Student",        lambda m: m.get("student_name") or "—"),
            (primary_label,    lambda m: format_pct(macro_avg(m, "realtime_rpa"))),
            ("RPA Clean",      lambda m: format_pct(get_condition(m, "clean", "realtime_rpa"))),
            ("RPA +0 dB",      lambda m: format_pct(get_condition(m, "+0 dB", "realtime_rpa"))),
            ("RPA -5 dB",      lambda m: format_pct(get_condition(m, "-5 dB", "realtime_rpa"))),
            ("Note",           lambda m: m.get("note") or "—"),
        ]
    elif sort_key == "ger_clean":
        columns = [
            ("Rank",           None),
            ("Student",        lambda m: m.get("student_name") or "—"),
            (primary_label,    lambda m: format_pct(get_condition(m, "clean", "realtime_gross_err"))),
            ("GER +0 dB",      lambda m: format_pct(get_condition(m, "+0 dB", "realtime_gross_err"))),
            ("GER -5 dB",      lambda m: format_pct(get_condition(m, "-5 dB", "realtime_gross_err"))),
            ("Note",           lambda m: m.get("note") or "—"),
        ]
    else:  # ger_macro
        columns = [
            ("Rank",           None),
            ("Student",        lambda m: m.get("student_name") or "—"),
            (primary_label,    lambda m: format_pct(macro_avg(m, "realtime_gross_err"))),
            ("GER Clean",      lambda m: format_pct(get_condition(m, "clean", "realtime_gross_err"))),
            ("GER +0 dB",      lambda m: format_pct(get_condition(m, "+0 dB", "realtime_gross_err"))),
            ("GER -5 dB",      lambda m: format_pct(get_condition(m, "-5 dB", "realtime_gross_err"))),
            ("Note",           lambda m: m.get("note") or "—"),
        ]

    # Sort entries for this table
    def sort_val(m):
        v = m.get(f"_val_{sort_key}")
        if v is None:
            return float("inf") if lower_is_better else float("-inf")
        return v

    sorted_entries = sorted(entries, key=sort_val, reverse=(not lower_is_better))

    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep    = "| " + " | ".join("---" for _ in columns) + " |"
    rows   = [header, sep]

    for rank, m in enumerate(sorted_entries, start=1):
        cells = []
        for i, (label, fn) in enumerate(columns):
            if fn is None:
                cells.append(str(rank))
            else:
                cells.append(fn(m))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing per-student JSON result files")
    parser.add_argument("--output", default="LEADERBOARD.md",
                        help="Output Markdown file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    # --- Collect all entries and pre-compute sort values ---
    entries = []
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load {json_path}: {e}")
            continue
        m["_file"] = json_path.name
        m["_val_rpa_clean"]  = get_condition(m, "clean", "realtime_rpa")
        m["_val_rpa_macro"]  = macro_avg(m, "realtime_rpa")
        m["_val_ger_clean"]  = get_condition(m, "clean", "realtime_gross_err")
        m["_val_ger_macro"]  = macro_avg(m, "realtime_gross_err")
        if m["_val_rpa_clean"] is not None:
            m["_val_rpa_clean"] = float(m["_val_rpa_clean"])
        entries.append(m)

    today = date.today().isoformat()

    def make_section(sort_key, lower_is_better, heading, description):
        if not entries:
            return f"### {heading}\n\n_No submissions yet._"
        table = build_table(entries, sort_key, lower_is_better)
        return f"### {heading}\n\n{description}\n\n{table}"

    rpa_clean_section = make_section(
        "rpa_clean", False,
        "RPA — Clean Audio ↑",
        "Raw Pitch Accuracy on clean (no-noise) test clips. Higher is better.",
    )
    rpa_macro_section = make_section(
        "rpa_macro", False,
        "RPA — Macro Average (all SNR conditions) ↑",
        "Mean RPA across all 6 SNR conditions (clean, −5 dB, 0 dB, +5 dB, +10 dB, +20 dB). Higher is better.",
    )
    ger_clean_section = make_section(
        "ger_clean", True,
        "Gross Error Rate — Clean Audio ↓",
        "Fraction of voiced frames with pitch error > 50 cents on clean audio. Lower is better.",
    )
    ger_macro_section = make_section(
        "ger_macro", True,
        "Gross Error Rate — Macro Average (all SNR conditions) ↓",
        "Mean gross error rate across all 6 SNR conditions. Lower is better.",
    )

    content = f"""# NanoPitch Student Leaderboard

*Last updated: {today}*

All metrics use the **realtime Viterbi decoder** (no lookahead), matching the browser deployment.

---

## 1. RPA Leaderboards

{rpa_clean_section}

{rpa_macro_section}

---

## 2. Gross Error Rate Leaderboards

{ger_clean_section}

{ger_macro_section}

---

## Metrics glossary

| Metric | Description |
|--------|-------------|
| RPA | Raw Pitch Accuracy — % of voiced frames within 50 cents of ground truth (higher = better) |
| Gross Error Rate (GER) | % of voiced frames with pitch error > 50 cents (lower = better) |
| VAD Acc | Voice Activity Detection accuracy — % of frames correctly classified as voiced/unvoiced |
| Median Err | Median pitch error in cents across voiced frames (100 cents = 1 semitone) |
| Macro Avg | Mean of the metric across all 6 SNR conditions: clean, −5 dB, 0 dB, +5 dB, +10 dB, +20 dB |
"""

    output_path.write_text(content)
    print(f"Leaderboard written to {output_path} ({len(entries)} entries).")


if __name__ == "__main__":
    main()
