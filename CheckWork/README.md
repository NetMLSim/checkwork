# CheckWork Trace Generator

This directory contains the MLSynth-derived trace generator used by CheckWork. The main CheckWork contribution is `Wrapper/CheckpointWrapper.py`, which injects checkpoint operations into synthetic distributed-training DAGs before they are emitted as Chakra execution traces.

Generate a trace from a YAML workload description with:

```bash
python3 synthesise_workload.py -c input_checkfreq_like.yaml
```

Outputs are written under `output/<workload>/` and are ignored by Git.

The base workload-generation structure follows MLSynth: `Layer/` models individual layers, `Model/` composes full networks, `Orchestrator/` schedules distributed execution, and `Wrapper/` modifies the generated DAG.
