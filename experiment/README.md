# Experiments

This artifact includes the two experiment suites needed for the CheckWork robustness evaluation.

- `checkfreq_robustness/` sweeps checkpoint cadence, checkpoint size, storage tier, network topology, and model family.
- `checkfreq_bert_dense/` runs a higher-resolution BERT-only sweep over checkpoint cadence and checkpoint size.

Each experiment can be run with `python3 run_robustness.py --quick` for a smoke test or `python3 run_robustness.py` for the full study. Generated traces and raw simulator logs are ignored by Git.
