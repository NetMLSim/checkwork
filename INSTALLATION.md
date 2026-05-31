# Installation

This repository keeps external simulator and trace-analysis dependencies out of the tree. A working setup has three pieces: Python dependencies, Chakra Python utilities, and an ASTRA-sim build.

## Python And Chakra

```bash
./scripts/bootstrap.sh
source chakra_env/bin/activate
```

The bootstrap script creates `chakra_env/`, installs `pyyaml`, `numpy`, `pandas`, `matplotlib`, installs Chakra from GitHub, and pins a compatible protobuf runtime.

## ASTRA-sim

The experiment scripts expect a Docker image named `astra-sim:latest` with the analytical congestion-aware binary available at the standard upstream path.

```bash
git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git
cd astra-sim
# Follow upstream Docker build instructions, then tag the result:
docker build -t astra-sim:latest -f docker/Dockerfile .
```

See the upstream ASTRA-sim documentation for platform-specific build notes: https://astra-sim.github.io/astra-sim-docs/getting_started/build.html

## Optional Repositories

These are not required for the included experiments, but may be useful for schema regeneration or trace visualization:

```bash
git clone https://github.com/astra-sim/param.git
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
```

## Smoke Test

```bash
source chakra_env/bin/activate
cd experiment/checkfreq_robustness
python3 run_robustness.py --quick --local
```

Use the same command without `--local` once `astra-sim:latest` is available.
