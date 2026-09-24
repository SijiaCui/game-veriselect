"""Aggregate the 7 Qwen2.5-*B-Instruct GameSolve-Hard per-model JSONs into
summary_qwen25.json (same shape as the Qwen3 summary.json) + a markdown scaling
table. Non-thinking / default mode only (Qwen2.5 has no thinking mode)."""
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("GAMESOLVE_RESULTS") or Path(__file__).resolve().parents[1] / "results" / "gamesolve")
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
FILES = [(s, OUT / f"Qwen2.5-{s}-Instruct.json") for s in SIZES]

models, rows = {}, []
cfg, np_ = {}, {}  # run config/n_prompts, taken from the first surviving file (all identical)
for s, f in FILES:
    if not f.exists():
        print(f"MISSING {f.name}"); continue
    d = json.load(open(f))
    models[d["model"]] = {"gen_seconds": d["gen_seconds"], "summary": d["summary"]}
    if not cfg:
        cfg, np_ = d["config"], d["n_prompts"]
    bs = d["summary"]["by_split"]
    row = {"size": s}
    for split in ("ID", "OOD"):
        ov = bs.get(split, {}).get("overall")
        if ov:
            for k in ("pass1", "pass1_full", "majority", "oracle", "pass1_graded"):
                row[f"{split}_{k}"] = ov[k]
    rows.append(row)

summary = {
    "benchmark": "GameSolve-Hard (revised: eased hardest tiers, [-15,15] OOD)",
    "reward": "revised reward.py — exact=structural (NE pure+class+mixed SUPPORT; BR full); exact_full=+exact prob vectors",
    "metrics": "pass1=mean exact; pass1_full=mean exact_full; majority@8=modal-canon exact; oracle@8=any-of-8 exact",
    "mode": "default / non-thinking (Qwen2.5 has no thinking mode)",
    "config": cfg, "n_prompts": np_, "models": models,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "summary_qwen25.json").write_text(json.dumps(summary, indent=2))

# markdown scaling table
def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "—"
lines = ["# Qwen2.5-Instruct on GameSolve-Hard (default / non-thinking)", "",
         f"Config: n={cfg.get('n')}, per_group={cfg.get('per_group')}, seed={cfg.get('seed')}, "
         f"temp={cfg.get('temperature')}, top_p={cfg.get('top_p')}; "
         f"prompts ID={np_.get('ID')} OOD={np_.get('OOD')} total={np_.get('total')}.",
         "Score defs: pass@1=mean exact; maj=majority@8; oracle=any-of-8; graded=partial reward.", ""]
for split in ("ID", "OOD"):
    lines += [f"## {split}", "",
              "| size | pass@1 | full | maj@8 | oracle@8 | graded |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {size} | {p1} | {pf} | {mj} | {orc} | {gr} |".format(
            size=r["size"], p1=fmt(r.get(f"{split}_pass1")), pf=fmt(r.get(f"{split}_pass1_full")),
            mj=fmt(r.get(f"{split}_majority")), orc=fmt(r.get(f"{split}_oracle")),
            gr=fmt(r.get(f"{split}_pass1_graded"))))
    lines.append("")
(OUT / "REPORT_qwen25.md").write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nwrote {OUT/'summary_qwen25.json'} and {OUT/'REPORT_qwen25.md'}")
