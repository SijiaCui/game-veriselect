"""Aggregate Qwen2.5 GTBench results (nothink only) into a markdown scaling
table + summary_qwen25.json. Score = (win + 0.5*draw) / completed matches,
candidate vs the built-in random_agent. Mirrors eval/analyze_eval.py."""
import glob, json, os
from collections import defaultdict
from pathlib import Path

# results/gtbench of THIS checkout (eval/gtbench/ -> <repo>/eval/results/gtbench),
# overridable with GTBENCH_RESULTS for results kept elsewhere.
RES = Path(os.environ.get("GTBENCH_RESULTS")
           or Path(__file__).resolve().parents[1] / "results" / "gtbench")
MODELS = ["q25_0_5b", "q25_1_5b", "q25_3b", "q25_7b", "q25_14b", "q25_32b", "q25_72b"]
SIZE = {"q25_0_5b": "0.5B", "q25_1_5b": "1.5B", "q25_3b": "3B", "q25_7b": "7B",
        "q25_14b": "14B", "q25_32b": "32B", "q25_72b": "72B"}
MODE = "nothink"
MIN_N = 8


def analyze():
    out = defaultdict(dict)
    for model in MODELS:
        tag = f"{model}-{MODE}"
        for path in glob.glob(str(RES / tag / "*" / "*.jsonl")):
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
            out[model][game] = {"matches_total": total, "matches_normal": normal,
                                "wins": wins, "draws": draws, "losses": losses,
                                "score": (wins + 0.5 * draws) / normal if normal else None,
                                "completion_rate": normal / total}
    return {m: dict(v) for m, v in out.items()}


def fmt(v): return "—" if v is None else f"{v:.0%}"


def avg_by_size(data, key="score", n_key=None, min_n=0):
    a = {}
    for m in MODELS:
        ss = []
        for g, s in data.get(m, {}).items():
            if s.get(key) is None:
                continue
            if n_key and s.get(n_key, 0) < min_n:
                continue
            ss.append(s[key])
        a[m] = sum(ss) / len(ss) if ss else None
    return a


def build(data):
    L = ["# Qwen2.5-Instruct size-scaling on GTBench (default / non-thinking)", "",
         "7 Qwen2.5 chat models (0.5B → 72B) vs the built-in random_agent. "
         "Qwen2.5 has no thinking mode, so nothink only.",
         f"Score = (win + 0.5·draw) / completed. Cells: score (n completed); "
         f"**avg** averages games with n≥{MIN_N}.", "",
         "| Game | " + " | ".join(SIZE[m] for m in MODELS) + " |",
         "|:--|" + "--:|" * len(MODELS)]
    games = sorted({g for m in MODELS for g in data.get(m, {})})
    for g in games:
        row = [g]
        for m in MODELS:
            s = data.get(m, {}).get(g)
            row.append(f"{s['score']:.0%} (n={s['matches_normal']})" if s and s.get("score") is not None else "—")
        L.append("| " + " | ".join(row) + " |")
    avg = avg_by_size(data, n_key="matches_normal", min_n=MIN_N)
    comp = avg_by_size(data, key="completion_rate")
    L.append("| **avg score** | " + " | ".join(fmt(avg[m]) for m in MODELS) + " |")
    L.append("| **avg completion** | " + " | ".join(fmt(comp[m]) for m in MODELS) + " |")
    L += ["", "> Caveat: 20 games/cell ⇒ 95% CI ≈ ±22pp; read the cross-game avg and size trend.", ""]
    return "\n".join(L), {"avg_score": avg, "avg_completion": comp}


if __name__ == "__main__":
    data = analyze()
    report, scaling = build(data)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump({"gtbench": data, "scaling": scaling},
              open(RES / "summary_qwen25.json", "w"), indent=1)
    open(RES / "REPORT_qwen25.md", "w").write(report)
    print(report)
