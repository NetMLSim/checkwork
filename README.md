# CheckWork

**CheckWork: Enabling Trace-Driven Analysis of Checkpointing Overhead in Distributed ML Training**

Artifact accompanying our paper published at **APNet '26 — the 10th Asia-Pacific Workshop on Networking**.

**[Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3820441.3820476)** · **[DOI](https://doi.org/10.1145/3820441.3820476)** · **[APNet '26](https://conferences.sigcomm.org/events/apnet2026/)**

## Overview

Checkpointing is essential for fault tolerance in large-scale distributed ML training, but checkpoint traffic can interfere with computation and training communication, introducing significant overhead.

**CheckWork** enables trace-driven analysis of this overhead by generating checkpoint-aware [Chakra](https://github.com/mlcommons/chakra) execution traces for distributed ML workloads. CheckWork augments training DAGs with checkpoint operations, allowing different checkpointing strategies to be evaluated using Chakra-compatible simulators such as [ASTRA-sim](https://astra-sim.github.io/).

The framework builds on [MLSynth](https://github.com/sefianeadel/MLSynth) and extends its synthetic workload generation with checkpoint operations.

This repository contains the CheckWork trace generator, the experiment suites used in the paper, and the scripts required to reproduce the evaluation.


## Setup

CheckWork uses [ASTRA-sim](https://github.com/astra-sim/astra-sim) as the reference simulation backend. Build ASTRA-sim separately and tag its Docker image as `astra-sim:latest`:

```bash
git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git
cd astra-sim

docker build -t astra-sim:latest -f docker/Dockerfile .
```

See the [ASTRA-sim documentation](https://astra-sim.github.io/astra-sim-docs/getting_started/build.html#using-docker) for detailed build instructions.


## Paper

**CheckWork: Enabling Trace-Driven Analysis of Checkpointing Overhead in Distributed ML Training**
Eldar Hasanov, Adel Sefiane, Alireza Farshin, and Marios Kogias
*Proceedings of the 10th Asia-Pacific Workshop on Networking (APNet '26)*, 2026, pp. 239–245.

**[Read the paper on ACM Digital Library](https://dl.acm.org/doi/10.1145/3820441.3820476)**

## Citation

If you use CheckWork in your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3820441.3820476,
  author    = {Hasanov, Eldar and Sefiane, Adel and Farshin, Alireza and Kogias, Marios},
  title     = {CheckWork: Enabling Trace-Driven Analysis of Checkpointing Overhead in Distributed ML Training},
  year      = {2026},
  isbn      = {9798400726644},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3820441.3820476},
  doi       = {10.1145/3820441.3820476},
  booktitle = {Proceedings of the 10th Asia-Pacific Workshop on Networking},
  pages     = {239--245},
  numpages  = {7},
  keywords  = {Checkpointing, Synthetic Chakra Execution Traces, Simulation},
  series    = {APNet '26}
}
```

## License

CheckWork is released under the **Apache License 2.0**.

CheckWork builds on [MLSynth](https://github.com/sefianeadel/MLSynth) and [Chakra](https://github.com/mlcommons/chakra), and uses [ASTRA-sim](https://github.com/astra-sim/astra-sim) as the reference simulation backend. See `NOTICE` for attribution details.
