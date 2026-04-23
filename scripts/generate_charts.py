#!/usr/bin/env python3
"""
Generate leaderboard_charts.html — interactive Chart.js visualizations
from all JSON files in the results/ directory.

Usage:
    python scripts/generate_charts.py
    python scripts/generate_charts.py --results-dir results --output leaderboard_charts.html
"""

import argparse
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SNR_CONDITIONS = ["clean", "-5 dB", "+0 dB", "+5 dB", "+10 dB", "+20 dB"]
SNR_LABELS     = ["Clean", "−5 dB", "0 dB", "+5 dB", "+10 dB", "+20 dB"]

# Distinct, accessible color palette
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def get_condition(m, cond, key):
    return m.get(cond, {}).get(key)


def macro_avg(m, key):
    vals = [float(get_condition(m, c, key)) for c in SNR_CONDITIONS
            if get_condition(m, c, key) is not None]
    return sum(vals) / len(vals) if vals else None


def build_chart_data(entries):
    """Build all data structures needed by Chart.js."""
    labels = [m.get("student_name", m.get("_file", "?")) for m in entries]
    colors = PALETTE[:len(entries)]

    # Per-condition series: for each entry, list of values across SNR_CONDITIONS
    rpa_by_snr = [
        [round((get_condition(m, c, "realtime_rpa") or 0) * 100, 2)
         for c in SNR_CONDITIONS]
        for m in entries
    ]
    ger_by_snr = [
        [round((get_condition(m, c, "realtime_gross_err") or 0) * 100, 2)
         for c in SNR_CONDITIONS]
        for m in entries
    ]

    # Summary scalars for chart bars
    rpa_clean  = [round((get_condition(m, "clean", "realtime_rpa")       or 0) * 100, 2) for m in entries]
    rpa_macro  = [round((macro_avg(m, "realtime_rpa")                    or 0) * 100, 2) for m in entries]
    ger_clean  = [round((get_condition(m, "clean", "realtime_gross_err") or 0) * 100, 2) for m in entries]
    ger_macro  = [round((macro_avg(m, "realtime_gross_err")              or 0) * 100, 2) for m in entries]

    # Flat row objects for the sortable table (all metrics in one place)
    def pct(v): return round(float(v) * 100, 2) if v is not None else None
    table_rows = [
        {
            "name":      m.get("student_name", m.get("_file", "?")),
            "note":      m.get("note", ""),
            "rpa_clean": pct(get_condition(m, "clean", "realtime_rpa")),
            "rpa_macro": pct(macro_avg(m, "realtime_rpa")),
            "ger_clean": pct(get_condition(m, "clean", "realtime_gross_err")),
            "ger_macro": pct(macro_avg(m, "realtime_gross_err")),
            "vdr_clean": pct(get_condition(m, "clean", "realtime_vdr")),
            "vdr_macro": pct(macro_avg(m, "realtime_vdr")),
            "vad_clean": pct(get_condition(m, "clean", "vad_acc")),
            "vad_macro": pct(macro_avg(m, "vad_acc")),
        }
        for m in entries
    ]

    return {
        "labels":      labels,
        "colors":      colors,
        "snr_labels":  SNR_LABELS,
        "rpa_by_snr":  rpa_by_snr,
        "ger_by_snr":  ger_by_snr,
        "rpa_clean":   rpa_clean,
        "rpa_macro":   rpa_macro,
        "ger_clean":   ger_clean,
        "ger_macro":   ger_macro,
        "table_rows":  table_rows,
    }


def render_html(data: dict, today: str, n_entries: int) -> str:
    data_json = json.dumps(data, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NanoPitch Leaderboard Charts</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f7fa;
      color: #1a1a2e;
      margin: 0;
      padding: 1.5rem;
    }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
    .subtitle {{ color: #555; font-size: 0.9rem; margin-bottom: 2rem; }}
    h2 {{ font-size: 1.1rem; margin: 2rem 0 0.8rem; color: #333; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    .card {{
      background: #fff;
      border-radius: 10px;
      padding: 1.2rem 1.5rem;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }}
    .card h3 {{ margin: 0 0 0.8rem; font-size: 0.95rem; color: #444; }}
    canvas {{ max-height: 280px; }}
    .full-width {{ grid-column: 1 / -1; }}
    .full-width canvas {{ max-height: 320px; }}
    @media (max-width: 700px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

    /* ── sortable table ── */
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    thead th {{
      background: #f0f2f5;
      padding: 0.55rem 0.8rem;
      text-align: left;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
      border-bottom: 2px solid #d0d4da;
    }}
    thead th:hover {{ background: #e4e8ee; }}
    thead th.active {{ background: #dce6f5; color: #1a4a8a; }}
    tbody tr {{ border-bottom: 1px solid #eee; }}
    tbody tr:hover {{ background: #f9fbff; }}
    tbody td {{ padding: 0.5rem 0.8rem; vertical-align: top; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.rank {{ text-align: center; color: #888; font-weight: 600; }}
    td.note-cell {{
      max-width: 220px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: #666;
    }}
    .swatch {{ display: inline-block; width: 10px; height: 10px;
               border-radius: 2px; margin-right: 5px; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>NanoPitch Student Leaderboard</h1>
  <p class="subtitle">
    {n_entries} submission{"s" if n_entries != 1 else ""} &nbsp;·&nbsp; Last updated: {today} &nbsp;·&nbsp;
    Realtime Viterbi decoder &nbsp;·&nbsp;
    <a href="LEADERBOARD.md">Static MD table</a>
  </p>

  <h2>Rankings</h2>
  <div class="card">
    <p style="font-size:0.82rem;color:#666;margin:0 0 0.8rem">
      Click any column header to sort. Click again to reverse. Default: RPA Clean, best first.
    </p>
    <div class="table-wrap">
      <table id="leaderTable">
        <thead><tr></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <h2>Summary — 4 Key Metrics</h2>
  <div class="grid-2">
    <div class="card">
      <h3>RPA — Clean Audio ↑ (higher is better)</h3>
      <canvas id="rpaCleanchart"></canvas>
    </div>
    <div class="card">
      <h3>RPA — Macro Average across all SNR conditions ↑</h3>
      <canvas id="rpaMacrochart"></canvas>
    </div>
    <div class="card">
      <h3>Gross Error Rate — Clean Audio ↓ (lower is better)</h3>
      <canvas id="gerCleanchart"></canvas>
    </div>
    <div class="card">
      <h3>Gross Error Rate — Macro Average across all SNR conditions ↓</h3>
      <canvas id="gerMacrochart"></canvas>
    </div>
  </div>

  <h2>Performance Across Noise Conditions</h2>
  <div class="grid-2">
    <div class="card full-width">
      <h3>RPA by SNR Condition ↑</h3>
      <canvas id="rpaSnrchart"></canvas>
    </div>
    <div class="card full-width">
      <h3>Gross Error Rate by SNR Condition ↓</h3>
      <canvas id="gerSnrchart"></canvas>
    </div>
  </div>

<script>
const D = {data_json};

// ── sortable table ────────────────────────────────────────────────────────────

const COLS = [
  {{ key: "name",      label: "Student",       fmt: v => v,                  lower: null  }},
  {{ key: "rpa_clean", label: "RPA Clean",     fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "rpa_macro", label: "RPA Macro Avg", fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "ger_clean", label: "GER Clean",     fmt: v => v.toFixed(1) + "%", lower: true  }},
  {{ key: "ger_macro", label: "GER Macro Avg", fmt: v => v.toFixed(1) + "%", lower: true  }},
  {{ key: "vdr_clean", label: "VDR Clean",     fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "vdr_macro", label: "VDR Macro Avg", fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "vad_clean", label: "VAD Acc Clean", fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "vad_macro", label: "VAD Acc Macro", fmt: v => v.toFixed(1) + "%", lower: false }},
  {{ key: "note",      label: "Note",          fmt: v => v,                  lower: null  }},
];

// suffix shown in the header when column is not active
const HINT = {{ false: " ↑", true: " ↓", null: "" }};

let sortKey = "rpa_clean";
let sortAsc  = false;   // false = descending = best-first for "higher is better" cols

function renderTable() {{
  // ── header ──
  const headRow = document.querySelector("#leaderTable thead tr");
  headRow.innerHTML = COLS.map(c => {{
    const active = c.key === sortKey;
    const arrow  = active ? (sortAsc ? " ▲" : " ▼") : HINT[c.lower];
    return `<th data-key="${{c.key}}"${{active ? ' class="active"' : ''}}>${{c.label}}${{arrow}}</th>`;
  }}).join("");

  // ── body ──
  const sorted = [...D.table_rows].sort((a, b) => {{
    const av = a[sortKey], bv = b[sortKey];
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  }});

  document.querySelector("#leaderTable tbody").innerHTML = sorted.map((r, i) => {{
    const swatch = D.colors[D.table_rows.indexOf(r)] || "#aaa";
    return "<tr>" +
      `<td class="rank">${{i + 1}}</td>` +
      COLS.map(c => {{
        const val = r[c.key];
        const text = val != null ? c.fmt(val) : "—";
        if (c.key === "name") return `<td><span class="swatch" style="background:${{swatch}}"></span>${{text}}</td>`;
        if (c.key === "note") return `<td class="note-cell" title="${{text}}">${{text}}</td>`;
        return `<td class="num">${{text}}</td>`;
      }}).join("") +
      "</tr>";
  }}).join("");
}}

document.querySelector("#leaderTable thead tr").addEventListener("click", e => {{
  const th = e.target.closest("th[data-key]");
  if (!th) return;
  const key = th.dataset.key;
  if (key === sortKey) {{
    sortAsc = !sortAsc;
  }} else {{
    sortKey = key;
    const col = COLS.find(c => c.key === key);
    // default direction: lower-is-better cols sort asc, others desc
    sortAsc = col.lower === true;
  }}
  renderTable();
}});

renderTable();

// ── helpers ──────────────────────────────────────────────────────────────────

function barDatasets(values, alpha) {{
  return D.labels.map((label, i) => ({{
    label,
    data: [values[i]],
    backgroundColor: D.colors[i] + (alpha || "cc"),
    borderColor:     D.colors[i],
    borderWidth: 1.5,
    borderRadius: 5,
  }}));
}}

function lineDatasets() {{
  return D.labels.map((label, i) => ({{
    label,
    data: D.rpa_by_snr[i],
    borderColor: D.colors[i],
    backgroundColor: D.colors[i] + "30",
    borderWidth: 2.5,
    pointRadius: 5,
    tension: 0.3,
    fill: false,
  }}));
}}

function gerLineDatasets() {{
  return D.labels.map((label, i) => ({{
    label,
    data: D.ger_by_snr[i],
    borderColor: D.colors[i],
    backgroundColor: D.colors[i] + "30",
    borderWidth: 2.5,
    pointRadius: 5,
    tension: 0.3,
    fill: false,
  }}));
}}

const pctOptions = (suffix, legendPos) => ({{
  responsive: true,
  plugins: {{
    legend: {{ position: legendPos || "top", labels: {{ boxWidth: 14, font: {{ size: 11 }} }} }},
    tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}${{suffix}}` }} }},
  }},
  scales: {{
    y: {{ ticks: {{ callback: v => v + suffix }}, beginAtZero: false }},
    x: {{ grid: {{ display: false }} }},
  }},
}});

// ── summary bar charts ────────────────────────────────────────────────────────

new Chart(document.getElementById("rpaCleanchart"), {{
  type: "bar",
  data: {{ labels: ["RPA Clean"], datasets: barDatasets(D.rpa_clean) }},
  options: pctOptions("%"),
}});

new Chart(document.getElementById("rpaMacrochart"), {{
  type: "bar",
  data: {{ labels: ["RPA Macro Avg"], datasets: barDatasets(D.rpa_macro) }},
  options: pctOptions("%"),
}});

new Chart(document.getElementById("gerCleanchart"), {{
  type: "bar",
  data: {{ labels: ["GER Clean"], datasets: barDatasets(D.ger_clean) }},
  options: pctOptions("%"),
}});

new Chart(document.getElementById("gerMacrochart"), {{
  type: "bar",
  data: {{ labels: ["GER Macro Avg"], datasets: barDatasets(D.ger_macro) }},
  options: pctOptions("%"),
}});

// ── per-SNR line charts ───────────────────────────────────────────────────────

new Chart(document.getElementById("rpaSnrchart"), {{
  type: "line",
  data: {{ labels: D.snr_labels, datasets: lineDatasets() }},
  options: {{
    ...pctOptions("%", "top"),
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v + "%" }} }},
      x: {{ grid: {{ color: "#eee" }} }},
    }},
  }},
}});

new Chart(document.getElementById("gerSnrchart"), {{
  type: "line",
  data: {{ labels: D.snr_labels, datasets: gerLineDatasets() }},
  options: {{
    ...pctOptions("%", "top"),
    scales: {{
      y: {{ min: 0, ticks: {{ callback: v => v + "%" }} }},
      x: {{ grid: {{ color: "#eee" }} }},
    }},
  }},
}});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="leaderboard_charts.html")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    entries = []
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load {json_path}: {e}")
            continue
        m["_file"] = json_path.stem
        entries.append(m)

    # Sort by RPA clean descending (same order as primary leaderboard)
    entries.sort(
        key=lambda m: float(get_condition(m, "clean", "realtime_rpa") or 0),
        reverse=True,
    )

    data = build_chart_data(entries)
    today = date.today().isoformat()
    html = render_html(data, today, len(entries))
    output_path.write_text(html)
    print(f"Charts written to {output_path} ({len(entries)} entries).")


if __name__ == "__main__":
    main()
