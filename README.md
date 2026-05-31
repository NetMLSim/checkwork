# CheckWork

**CheckWork: Enabling Trace-Driven Analysis of Checkpointing Overhead in Distributed ML Training**

CheckWork is the artifact for the APNet '26 paper of the same name. It generates checkpoint-aware [Chakra execution traces](https://github.com/mlcommons/chakra) for distributed ML training workloads and evaluates their overhead with Chakra-compatible simulators such as [ASTRA-sim](https://astra-sim.github.io/).

This repository is intentionally minimal. It contains the CheckWork trace-generation extension, two paper-supporting experiment suites, and the documentation needed to reproduce them. It does **not** vendor ASTRA-sim, Chakra environments, `param`, HolisticTraceAnalysis, generated traces, or simulator logs.

## Relationship To MLSynth And Chakra

CheckWork is built on [MLSynth](https://github.com/sefianeadel/MLSynth), which provides synthetic distributed-training workload generation. The `CheckWork/` directory is an MLSynth-derived trace generator extended with `CheckpointWrapper`, which injects checkpoint operations into the training DAG before serializing it as Chakra ET protobuf streams.

CheckWork also depends on the [Chakra](https://github.com/mlcommons/chakra) execution-trace schema and Python utilities. Chakra is installed from upstream during setup; it is not vendored in this repository.

## Repository Layout

```text
CheckWork/                         MLSynth-derived trace generator with CheckWork extensions
CheckWork/Wrapper/CheckpointWrapper.py
                                   Core checkpoint-node injection logic
experiment/checkfreq_robustness/   Robustness sweep over model, cadence, size, storage, and topology
experiment/checkfreq_bert_dense/   Dense BERT-only cadence/size sweep
scripts/bootstrap.sh               Python environment bootstrap for Chakra + analysis dependencies
```

## Setup

```bash
git clone https://github.com/eldarhasanov079/checkwork.git
cd checkwork
./scripts/bootstrap.sh
source chakra_env/bin/activate
```

Build ASTRA-sim separately and tag the Docker image as `astra-sim:latest`:

```bash
git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git
cd astra-sim
# Follow the upstream Docker build instructions:
# https://astra-sim.github.io/astra-sim-docs/getting_started/build.html#using-docker
docker build -t astra-sim:latest -f docker/Dockerfile .
```

Optional external repositories, not needed for the two included experiments, can be cloned separately if you want schema regeneration or trace-visualization tooling:

```bash
git clone https://github.com/astra-sim/param.git
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
```

## Running The Experiments

```bash
cd experiment/checkfreq_robustness
python3 run_robustness.py --quick
python3 plot_results.py

cd ../checkfreq_bert_dense
python3 run_robustness.py --quick
python3 plot_results.py
```

For full runs, omit `--quick`. Use `--local` to exercise the built-in topological simulator when Docker or ASTRA-sim is unavailable; the local simulator is useful for sanity checks but is lower fidelity than ASTRA-sim.

Generated traces are written under each experiment's `traces/` directory. Raw simulator logs are written under `results/timeline_logs/`. Both are intentionally ignored by Git.

## Citation

If you use CheckWork in academic work, please cite:

```bibtex
@inproceedings{hasanov2026checkwork,
  title     = {{CheckWork}: Enabling Trace-Driven Analysis of Checkpointing
               Overhead in Distributed {ML} Training},
  author    = {Hasanov, Eldar and Sefiane, Adel and Farshin, Alireza and
               Kogias, Marios},
  booktitle = {Proceedings of the 10th Asia-Pacific Workshop on Networking
               (APNet '26)},
  year      = {2026},
  location  = {Singapore},
  publisher = {ACM}
}
```

A machine-readable citation file is available in `CITATION.cff`.

## License And Acknowledgements

CheckWork is released under the Apache License 2.0. It builds directly on MLSynth and Chakra, and the experiments use ASTRA-sim as the reference replay backend. See `NOTICE` for attribution details.
