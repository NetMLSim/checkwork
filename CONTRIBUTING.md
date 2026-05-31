# Contributing To CheckWork

Thank you for your interest in CheckWork. This repository is maintained primarily as an academic artifact for the APNet '26 paper, so contributions should keep the codebase small, reproducible, and easy to audit.

## Scope

Appropriate contributions include:

- fixes to checkpoint trace generation in `CheckWork/Wrapper/CheckpointWrapper.py`;
- small improvements to the included robustness and dense BERT experiments;
- documentation updates that improve reproducibility;
- compatibility fixes for supported Chakra or ASTRA-sim releases.

Please avoid committing generated traces, simulator logs, virtual environments, external repositories, or unrelated experiments. In particular, `astra-sim/`, `chakra_env/`, `param/`, `HolisticTraceAnalysis/`, `traces/`, and `results/timeline_logs/` should stay local.

## Development Workflow

Set up the environment with:

```bash
./scripts/bootstrap.sh
source chakra_env/bin/activate
```

Before sending a change, run the smallest relevant check:

```bash
cd CheckWork
python3 test_checkfreq_like.py
python3 test_wrapper_new.py
```

For experiment changes, also run a quick sweep:

```bash
cd experiment/checkfreq_robustness
python3 run_robustness.py --quick --local
```

## Style

Follow the style of the surrounding Python code: 4-space indentation, clear names, and comments only where they explain a non-obvious invariant. Keep experiment scripts self-contained and make new generated outputs reproducible from a documented command.

## Pull Requests

Use one focused topic per pull request. In the description, include the motivation, the commands you ran, and whether the change affects trace semantics, simulator invocation, or only documentation.
