"""Aggregate GTBench results for the Qwen3 size-scaling study into a
markdown report + summary.json.

We report, per decode mode (nothink / think), a table of score vs model
size, plus an avg-score scaling row and the think-minus-nothink delta.
Score = (win + 0.5*draw) / completed (Normal) matches.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent           # eval/
RES = ROOT / "results"
MODELS = ["q0_6b", "q1_7b", "q4b", "q8b", "q14b", "q32b"]
MODES = ["nothink", "think"]
SIZE = {"q0_6b": "0.6B", "q1_7b": "1.7B", "q4b": "4B",
        "q8b": "8B", "q14b": "14B", "q32b": "32B"}
MIN_N = 8  # min completed games for a cell to count toward the average


def analyze_gtbench():
    """-> out[model][mode][game] = {score, completion_rate, ...}"""
    out = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        for mode in MODES:
            tag = f"{model}-{mode}"
            for path in glob.glob(str(RES / "gtbench" / tag / "*" / "*.jsonl")):
                game = Path(path).parent.name
                wins = draws = losses = normal = total = 0
                for line in open(path):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for mt in rec.get("matches", []):
                        total += 1
                        if mt.get("status") != "Normal":
                            continue
                        normal += 1
                        w = mt.get("winner", "") or ""
                        if not w:
                            draws += 1
                        elif tag in w:
                            wins += 1
                        else:
                            losses += 1
                if total == 0:
                    continue
                out[model][mode][game] = {
                    "matches_total": total, "matches_normal": normal,
                    "wins": wins, "draws": draws, "losses": losses,
                    "score": (wins + 0.5 * draws) / normal if normal else None,
                    "completion_rate": normal / total,
                }
    return {m: dict(v) for m, v in out.items()}


def fmt(v):
    return "—" if v is None else f"{v:.0%}"


def _avg_by_size(data, mode, score_key="score", n_key=None, min_n=0):
    """avg score over games for each model at a given mode."""
    avg = {}
    for m in MODELS:
        cells = data.get(m, {}).get(mode, {})
        ss = []
        for g, s in cells.items():
            if s.get(score_key) is None:
                continue
            if n_key is not None and s.get(n_key, 0) < min_n:
                continue
            ss.append(s[score_key])
        avg[m] = sum(ss) / len(ss) if ss else None
    return avg


def _score_table(data, mode, n_key):
    games = sorted({g for m in MODELS for g in data.get(m, {}).get(mode, {})})
    L = ["| Game | " + " | ".join(SIZE[m] for m in MODELS) + " |",
         "|:--|" + "--:|" * len(MODELS)]
    for g in games:
        row = [g]
        for m in MODELS:
            s = data.get(m, {}).get(mode, {}).get(g)
            if s and s.get("score") is not None:
                row.append(f"{s['score']:.0%} (n={s[n_key]})")
            else:
                row.append("—")
        L.append("| " + " | ".join(row) + " |")
    avg = _avg_by_size(data, mode, n_key=n_key, min_n=MIN_N)
    L.append("| **avg score** | " + " | ".join(fmt(avg[m]) for m in MODELS) + " |")
    return "\n".join(L), avg


def build(gt):
    L = ["# Qwen3 size-scaling on GTBench", "",
         "6 Qwen3 chat models (0.6B → 32B), each evaluated with thinking **off** and **on**.",
         f"Score = (win + 0.5·draw) / completed games. Cells show score (n completed); "
         f"the **avg score** row averages over games with n≥{MIN_N}.", ""]

    scaling = {"gtbench": {}}

    L += ["## GTBench — candidate vs built-in random_agent", ""]
    for mode in MODES:
        tbl, avg = _score_table(gt, mode, "matches_normal")
        scaling["gtbench"][mode] = avg
        L += [f"### thinking = {mode}", "", tbl, "",
              "| **avg completion** | " + " | ".join(
                  fmt(_avg_by_size(gt, mode, score_key="completion_rate")[m]) for m in MODELS
              ) + " |", ""]

    # scaling summary + thinking delta
    L += ["## Scaling summary (avg score by size)", "",
          "| mode | " + " | ".join(SIZE[m] for m in MODELS) + " |",
          "|:--|" + "--:|" * len(MODELS)]
    d = scaling["gtbench"]
    for mode in MODES:
        L.append(f"| {mode} | " + " | ".join(fmt(d[mode][m]) for m in MODELS) + " |")
    # think - nothink delta
    delta = []
    for m in MODELS:
        a, b = d["think"].get(m), d["nothink"].get(m)
        delta.append("—" if a is None or b is None else f"{(a - b) * 100:+.0f}pp")
    L.append("| **think−nothink** | " + " | ".join(delta) + " |")
    L += ["",
          "> Caveat: 20 games/cell ⇒ 95% CI ≈ ±22pp; read the cross-game **avg** and the "
          "size trend, not single cells.", ""]
    return "\n".join(L), scaling


if __name__ == "__main__":
    gt = analyze_gtbench()
    report, scaling = build(gt)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump({"gtbench": gt, "scaling": scaling},
              open(RES / "summary.json", "w"), indent=1)
    open(RES / "REPORT.md", "w").write(report)
    print(report)
