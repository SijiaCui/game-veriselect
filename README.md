# VeriSelect

A test-time best-of-N selector. Training-free.

VeriSelect is a formally-sound game-theory *process verifier* used at inference time to
rank N i.i.d. traces drawn from a policy, returning the argmax. Training-free, and
ground-truth-free in its gold-free variant.

The pivot is **selection vs. shaping** of one identical signal: gradient-ascending a
policy against the verifier (shaping) lets it manufacture verifier-satisfying-but-wrong
traces and *hurts*; ranking N samples from a natural policy (selection) can't be gamed
and *helps*.

## Structure

```
game-veriselect/
├── veriselect/   the core selection method + process verifiers
├── bench/        benchmark data and generators
│   ├── gamesolve/    in-house GameSolve-Hard matrix-game data/generators
│   └── GTBench/      interactive game-theory benchmark (vendored source)
└── eval/         evaluation harness (GameSolve-Hard + GTBench)
```

See the README in each directory for detail: `veriselect/README.md`,
`bench/README.md`, `eval/README.md`.

## Setup

```bash
pip install -r requirements.txt
```

The evaluation runners also need the model weights, which live **outside** this repo.
Point them at that directory with `MODEL_ROOT`:

```bash
export MODEL_ROOT=/path/to/models        # e.g. holds Qwen3-8B/, Qwen2.5-7B-Instruct/
```

## Quick check

```bash
python3 veriselect/veriselect.py         # -> "veriselect selfcheck OK"
```

## License

Apache-2.0.
