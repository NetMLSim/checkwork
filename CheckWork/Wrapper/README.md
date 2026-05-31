# CheckpointWrapper

Injects checkpoint-related Chakra nodes into MLSynth-derived execution
traces (ETs) at end-of-iteration boundaries.

This is the core of the CheckWork tool: by emitting checkpoint operations as
real DAG nodes (with dependencies, cost, and duration) instead of post-hoc
adjustments, you can drop the resulting traces straight into ASTRA-sim (or
any Chakra-compatible simulator) and measure overhead the same way you would
measure any other workload feature.

## Quick start

```yaml
# input.yaml
model:
  name: transformer
  num_layers: 24
  sequence_len: 2048
  vocab_size: 51200
  hidden_size: 20480
  batch_size: 32
  num_microbatches: 8
  bytes_per_val: 2
  scale: 1

parallelism: { dp_size: 8, pp_size: 1, tp_size: 1 }
num_iterations: 200

wrapper:
  type: checkpoint
  mode: checkfreq_like
  checkpoint_every_n: 25
  state_multiplier: 3.0          # model + optimizer state size factor
  stage_costs:
    snapshot_micros: 200
    persist_micros: 50000
```

```bash
python3 synthesise_workload.py -c input.yaml
# -> output/<workload_name>/et/*.et + comm_groups.json
```

## Modes

| Mode             | Behaviour                                                                                                                                                |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sync`           | Single blocking node per checkpoint event. Cost = snapshot + persist (each scaled by `overhead_multiplier`).                                            |
| `async`          | Local kickoff + background write (`WRITE_BG`). Background phase concurrency controlled by `max_inflight`.                                                |
| `remote_sync`    | Kickoff + remote transfer; next iteration fences on **all** emitted nodes. Destination set by `remote_target`.                                          |
| `remote_async`   | Kickoff + remote transfer in background. Destination set by `remote_target`.                                                                             |
| `checkfreq_like` | Two-phase snapshot/persist (CheckFreq, 2021). Snapshot fences the next weight update; persist runs in background. `max_inflight` defaults to 1.         |
| `pipelined`      | Same DAG shape as `checkfreq_like`. Use `max_inflight: N` to approximate PCcheck-style multiple-in-flight persists.                                      |
| `tiered`         | User-defined list of stages, each with its own cadence, kind, and destination. Approximates Gemini-style multi-tier checkpointing.                       |

**Legacy mode aliases** (kept working; emit `DeprecationWarning`):
`remote_rank0` → `remote_sync` + `remote_target: rank0`;
`remote_rank0_async` → `remote_async` + `remote_target: rank0`.

### `remote_target`

For `remote_sync` and `remote_async`:

| Value      | Behaviour                                                                                                                |
|------------|--------------------------------------------------------------------------------------------------------------------------|
| `ring`     | Each rank sends to the next, receives from the previous (GPU↔GPU proxy).                                                 |
| `rank0`    | Gather to rank 0: every non-zero rank sends to rank 0; rank 0 has receives from each.                                    |
| `storage`  | Send to a virtual storage rank (default id = `num_npus`; configurable via `storage_rank_id`).                            |

Legacy: `use_storage_sink: true` with `remote_async` is equivalent to
`remote_target: storage`.

## Configurability

Every knob below is optional. When omitted, the default preserves the
historical behaviour for the supported checkpoint modes (verified by
`CheckWork/test_checkfreq_like.py` and `CheckWork/test_wrapper_new.py`).

### Top-level (`wrapper:`)

| Option                            | Default     | Description                                                                                                                                                                                                  |
|-----------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`                            | —           | Must be `"checkpoint"`.                                                                                                                                                                                       |
| `mode`                            | `sync`      | One of: `sync`, `async`, `remote_sync`, `remote_async`, `checkfreq_like`, `pipelined`, `tiered` (plus legacy aliases above).                                                                                  |
| `remote_target`                   | `ring`      | For remote modes: `ring`, `rank0`, `storage`.                                                                                                                                                                |
| `checkpoint_every_n`              | `1`         | Iterations between checkpoint events (validated > 0).                                                                                                                                                         |
| `state_multiplier`                | `3.0`       | Factor applied to model parameters to get bytes-per-rank (model + optimizer + overhead).                                                                                                                      |
| `overhead_multiplier`             | `1.0`       | Scales **every per-byte cost and duration** uniformly across all modes. Applied symmetrically (sync, async, remote_*, checkfreq_like, pipelined, tiered).                                                    |
| `persist_in_background_no_graph`  | `false`     | `checkfreq_like`/`pipelined`: skip the persist node entirely. Used for calibration where the persist runs out-of-graph.                                                                                       |
| `drain_when_backpressure`         | `false`     | Add an explicit `DRAIN` boundary node when there is a previous background write still in flight. Equivalent to `max_inflight: 1` plus an explicit drain.                                                     |
| `max_inflight`                    | `1`         | Max background writes/uploads that may overlap. `0` = unlimited (no chain), `1` = legacy chain, `N` = sliding window of size N.                                                                              |
| `align_with`                      | `iteration` | `iteration` (default) or `epoch`. With `epoch`, checkpoint cadence is `checkpoint_every_n × iterations_per_epoch` iterations.                                                                                |
| `iterations_per_epoch`            | `1`         | Used when `align_with: epoch`.                                                                                                                                                                                |
| `comm_tag_base`                   | `8000`      | Per-iter comm tag = `comm_tag_base + iteration`. Configure to avoid collisions with other wrappers in multi-job runs.                                                                                         |
| `storage_rank_id`                 | `num_npus`  | NPU id used as the destination for `remote_target: storage` and `tiered.destination: storage`. Registered as a singleton comm group by the orchestrator.                                                     |
| `checkpoint_bytes_respect_scale`  | `true`      | If `false`, the model `scale` field does **not** shrink checkpoint bytes. Useful for calibration runs that shrink compute/comm via `scale` without shrinking the checkpoint workload.                        |
| `seed`                            | `null`      | RNG seed for cost jitter (per-rank).                                                                                                                                                                          |
| `rank_overrides`                  | `{}`        | `{npu_id: {state_multiplier?, <any stage_costs key>?}}`. Lets you model rank-0 aggregation hotspots, asymmetric per-rank bandwidths, etc.                                                                    |
| `stages`                          | `[]`        | Required for `mode: tiered`. List of `{kind, every_n, destination, on_critical_path?}` dicts.                                                                                                                |

### Per-stage cost model (`stage_costs:`)

All optional. Missing keys fall back to legacy top-level keys (which now emit
deprecation warnings).

| Key                              | Default               | Description                                                                                                              |
|----------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------|
| `snapshot_multiplier`            | `1.0`                 | Snapshot flops = `bytes × this × overhead_multiplier`.                                                                  |
| `persist_multiplier`             | `5.0`                 | Persist flops = `bytes × this × overhead_multiplier`.                                                                   |
| `write_bg_multiplier`            | `1.0`                 | Async local `WRITE_BG` cost = `bytes × this × overhead_multiplier`.                                                     |
| `snapshot_micros`                | `null`                | Explicit override of snapshot duration; takes precedence over BW and multiplier-derived values.                          |
| `persist_micros`                 | `null`                | Explicit override of persist duration.                                                                                   |
| `sync_total_micros`              | `null`                | Override the **total** sync-node duration (replaces snap+persist sum).                                                    |
| `snapshot_bandwidth_GBps`        | `null`                | If set (and `snapshot_micros` is not), snapshot duration = `bytes / (GB/s × 1e9) × 1e6` µs. Models PCIe GPU→host etc.    |
| `persist_bandwidth_GBps`         | `null`                | Same, for the persist stage (typically NVMe / network bandwidth).                                                        |
| `kickoff_micros`                 | `100`                 | Duration of the kickoff node. When `0`, kickoff also has zero flops (legacy convention).                                  |
| `kickoff_flops_cap`              | mode default          | Cap on kickoff flops. Defaults: `checkfreq_like/pipelined` 1e5, local `async` 1e6, `remote_*` 5e5. Override to lift caps. |
| `write_bg_flops_cap`             | `null`                | Cap on local `WRITE_BG` flops. Default is uncapped (matches legacy async).                                                |
| `snapshot_flops_cap`             | `null`                | Cap on snapshot flops. Default uncapped.                                                                                  |
| `persist_flops_cap`              | `null`                | Cap on persist flops. Default uncapped.                                                                                   |
| `compression_ratio`              | `1.0`                 | `output_bytes = input_bytes / compression_ratio` (post-compression). Approximates Check-N-Run-style quantized checkpoints.|
| `compression_cost_per_byte_micros` | `0.0`               | Extra microseconds **added to snapshot duration** per (effective) byte. Models GPU-side compression cost.                 |
| `incremental_fraction`           | `1.0`                 | Fraction of state actually written per event. Approximates IncrCP / Check-N-Run incremental snapshots.                    |
| `cost_jitter_pct`                | `0.0`                 | Per-event lognormal multiplier with `sigma = pct/100`, mean = 1. Models real storage tail variance.                       |

### Tiered mode stages

Each stage in `stages:` is a dict:

| Key                | Default    | Allowed values                                  | Notes                                                                                              |
|--------------------|------------|------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `kind`             | —          | `snapshot`, `persist`, `write_bg`              | Which `stage_costs` formula to use.                                                                 |
| `every_n`          | `1`        | int > 0                                        | Stage fires when `iteration % every_n == 0`.                                                        |
| `destination`      | `local`    | `local`, `ring`, `rank0`, `storage`            | If non-`local`, emits the appropriate send/recv nodes after the compute.                            |
| `on_critical_path` | `false`    | bool                                           | If `true`, the next iteration's first compute fences on the stage's tail (like `checkfreq_like`).   |

#### Gemini-like example (snapshot to CPU mem every iter, persist to storage every 50)

```yaml
wrapper:
  type: checkpoint
  mode: tiered
  storage_rank_id: 32
  stages:
    - { kind: snapshot, every_n: 1,  destination: local,   on_critical_path: true }
    - { kind: persist,  every_n: 50, destination: storage, on_critical_path: false }
  stage_costs:
    snapshot_bandwidth_GBps: 12     # GPU -> host (PCIe Gen4)
    persist_bandwidth_GBps: 3       # host -> SSD / network
    cost_jitter_pct: 15
  seed: 1
```

#### PCcheck-like pipelined example

```yaml
wrapper:
  type: checkpoint
  mode: pipelined            # alias for checkfreq_like with concurrent persists
  checkpoint_every_n: 5
  state_multiplier: 3.0
  max_inflight: 4            # up to 4 persists may overlap before back-pressure
  stage_costs:
    snapshot_micros: 200
    persist_micros: 80000
```

## Node naming convention

Every emitted node follows the pattern:

```
<PREFIX>_iter<N>[_<extra>]_npu<NPU_ID>
```

Tests and post-processing scripts grep by name, so a single helper
(`Wrapper.CheckpointWrapper._name`) is the source of truth.

| Mode                | Prefixes (in emission order)                                                                                                  |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `sync`              | `COMP_NODE_CHECKPOINT_SAVE`                                                                                                   |
| `async`             | (`COMP_NODE_CHECKPOINT_DRAIN`?,) `COMP_NODE_CHECKPOINT_KICKOFF`, `COMP_NODE_CHECKPOINT_WRITE_BG`                              |
| `remote_sync`/ring  | `COMP_NODE_CHECKPOINT_REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_REMOTE`, `COMM_RECV_NODE_CHECKPOINT_REMOTE`                 |
| `remote_sync`/rank0 | `..._REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_RANK0` or `COMM_RECV_NODE_CHECKPOINT_RANK0_..._from{src}`                    |
| `remote_sync`/storage | `..._REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_REMOTE_SYNC`                                                              |
| `remote_async`/ring | (`..._REMOTE_DRAIN`?,) `..._REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_REMOTE_ASYNC`, `COMM_RECV_NODE_CHECKPOINT_REMOTE_ASYNC` |
| `remote_async`/rank0 | (`..._REMOTE_DRAIN`?,) `..._REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_RANK0_ASYNC` / `COMM_RECV_NODE_CHECKPOINT_RANK0_ASYNC_..._from{src}` |
| `remote_async`/storage | (`..._REMOTE_DRAIN`?,) `..._REMOTE_KICKOFF`, `COMM_SEND_NODE_CHECKPOINT_REMOTE_ASYNC`                                    |
| `checkfreq_like`, `pipelined` | (`DRAIN`?,) `CF_KICKOFF`, `CF_SNAPSHOT`, `CF_PERSIST` (skip persist if `persist_in_background_no_graph: true`) |
| `tiered`            | `TIERED_STAGE_COMP_..._s{idx}_{kind}`, `TIERED_STAGE_SEND/RECV_..._s{idx}_{destination}`                                     |

## Return value: `CheckpointInjection`

`get_checkpoint_nodes(...)` returns a dataclass:

```python
@dataclass
class CheckpointInjection:
    nodes: list[ChakraNode]                  # nodes to append for this iter
    bg_tail: Optional[ChakraNode]            # next ckpt fences here (max_inflight)
    snapshot_fence: Optional[ChakraNode]     # next weight update fences here
    boundary_nodes: Optional[list[ChakraNode]]  # nodes the next iter fences on
                                             # (default: [nodes[0]])
```

For backwards compatibility it is also unpackable as a 3-tuple:

```python
nodes, bg_tail, snapshot_fence = wrapper.get_checkpoint_nodes(...)
```

The orchestrator uses `boundary_nodes` when set (e.g. `remote_sync` ring,
`tiered` with `on_critical_path: true`), otherwise falls back to
`[nodes[0]]`.

## Storage rank

When the wrapper uses `remote_target: storage` (or a tiered stage with
`destination: storage`), the orchestrator (`MegatronLM.generate_comm_groups`)
adds a singleton comm group `checkpoint_storage` containing
`storage_rank_id`. ASTRA-sim ignores unreferenced ranks for P2P sends, but
other simulators may use the group to allocate the endpoint.

## Programmatic use / extending

Add a new mode in three steps:

1. Add the mode name to `_VALID_MODES` in
   `Wrapper/CheckpointWrapper.py`.
2. Implement `_emit_<mode>(self, npu_id, valid_parents, iteration,
   prev_write_bg) -> CheckpointInjection`.
3. Register it in the `_emitters` dispatch table inside `__init__`.

See `_emit_tiered` for a complete worked example that uses the full
`StageCostModel` machinery, the per-rank override path, and the storage
rank.

## Tests

From `CheckWork/`:

```bash
python3 test_checkfreq_like.py   # legacy regression test
python3 test_wrapper_new.py      # full configurability/validation suite (30 tests)
```
