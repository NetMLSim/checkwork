#!/usr/bin/env python3
"""
Validate checkfreq_like checkpoint mode: dependency structure and one-in-flight persist.
Run from CheckWork: python test_checkfreq_like.py
Or from repo root: python CheckWork/test_checkfreq_like.py
"""
import sys
from pathlib import Path

# Allow importing from CheckWork when run from repo root
MLSYNTH = Path(__file__).resolve().parent
if str(MLSYNTH) not in sys.path:
    sys.path.insert(0, str(MLSYNTH))

from Model.Transformer import Transformer
from Wrapper.CheckpointWrapper import CheckpointWrapper
from Orchestrator.MegatronLM import MegatronLM

# Chakra Node type for filtering (first message per .et is GlobalMetadata)
try:
    from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
except ImportError:
    ChakraNode = None


def _is_node(obj):
    return hasattr(obj, "data_deps") and hasattr(obj, "name") and hasattr(obj, "id")


def build_node_maps(nodes_list):
    """From list of messages (GlobalMetadata + ChakraNode), return id->node, name->node for nodes only."""
    id_to_node = {}
    name_to_node = {}
    for item in nodes_list:
        if _is_node(item):
            id_to_node[item.id] = item
            name_to_node[item.name] = item
    return id_to_node, name_to_node


def _get_config():
    """Minimal config for checkfreq_like validation (no yaml dependency)."""
    return {
        "model": {
            "name": "transformer",
            "num_layers": 24,
            "sequence_len": 2048,
            "vocab_size": 51200,
            "hidden_size": 20480,
            "batch_size": 32,
            "num_microbatches": 8,
            "bytes_per_val": 2,
            "scale": 1,
        },
        "parallelism": {"dp_size": 2, "pp_size": 2, "tp_size": 1},
        "num_iterations": 5,
        "wrapper": {
            "type": "checkpoint",
            "mode": "checkfreq_like",
            "checkpoint_every_n": 1,
            "state_multiplier": 3.0,
            "snapshot_overhead_multiplier": 1.0,
            "persist_overhead_multiplier": 5.0,
            "kickoff_cost_micros": 100,
        },
    }


def main():
    cfg = _get_config()

    model = Transformer(cfg)
    model = CheckpointWrapper(model, cfg)
    assert model.mode == "checkfreq_like"
    orchestrator = MegatronLM(model, cfg)
    orchestrator.generate_comm_groups()  # required before exec() (sets _pg_name_to_id)
    nodes = orchestrator.exec()

    num_iterations = cfg.get("num_iterations", 5)
    npu_id = 0
    node_list = nodes[npu_id]
    id_to_node, name_to_node = build_node_maps(node_list)

    errors = []

    # 1) Checkpoint nodes exist for each checkpoint iteration; snapshot -> kickoff, persist -> snapshot
    for i in range(num_iterations):
        if i % model.checkpoint_every_n != 0:
            continue
        kickoff = name_to_node.get(f"CF_KICKOFF_iter{i}_npu{npu_id}")
        snapshot = name_to_node.get(f"CF_SNAPSHOT_iter{i}_npu{npu_id}")
        persist = name_to_node.get(f"CF_PERSIST_iter{i}_npu{npu_id}")
        if not kickoff:
            errors.append(f"Missing CF_KICKOFF_iter{i}_npu{npu_id}")
        if not snapshot:
            errors.append(f"Missing CF_SNAPSHOT_iter{i}_npu{npu_id}")
        if not persist:
            errors.append(f"Missing CF_PERSIST_iter{i}_npu{npu_id}")
        if snapshot and kickoff and kickoff.id not in snapshot.data_deps:
            errors.append(f"CF_SNAPSHOT_iter{i} must depend on CF_KICKOFF_iter{i}")
        if persist and snapshot and snapshot.id not in persist.data_deps:
            errors.append(f"CF_PERSIST_iter{i} must depend on CF_SNAPSHOT_iter{i}")

    # 2) Iteration i+1 forward does NOT depend on snapshot(i) or persist(i)
    for i in range(num_iterations - 1):
        if i % model.checkpoint_every_n != 0:
            continue
        snapshot = name_to_node.get(f"CF_SNAPSHOT_iter{i}_npu{npu_id}")
        persist = name_to_node.get(f"CF_PERSIST_iter{i}_npu{npu_id}")
        if not snapshot or not persist:
            continue
        # First forward compute of iter i+1 (any microbatch, layer 0)
        fwd_name = f"COMP_NODE_FWD_iter{i+1}_b0_dp0pp0tp0"
        fwd_node = name_to_node.get(fwd_name)
        if fwd_node:
            if snapshot.id in fwd_node.data_deps:
                errors.append(f"FWD iter{i+1} must not depend on CF_SNAPSHOT_iter{i}")
            if persist.id in fwd_node.data_deps:
                errors.append(f"FWD iter{i+1} must not depend on CF_PERSIST_iter{i}")

    # 3) Weight update of iteration i+1 (DP all-reduce or last backward) depends on snapshot(i)
    for i in range(num_iterations - 1):
        if i % model.checkpoint_every_n != 0:
            continue
        snapshot = name_to_node.get(f"CF_SNAPSHOT_iter{i}_npu{npu_id}")
        if not snapshot:
            continue
        # Find any node that has snapshot(i) as dependency (the fence is attached to weight update)
        nodes_depending_on_snapshot = [n for n in id_to_node.values() if snapshot.id in n.data_deps]
        dp_name = f"COMM_COLL_NODE_DP_All-Reduce_iter{i+1}_dp0pp0tp0"
        def is_weight_update(n):
            return n.name == dp_name or (n.name.startswith(f"COMP_NODE_BCKWD_iter{i+1}_") and "dp0pp0tp0" in n.name)
        if not any(is_weight_update(n) for n in nodes_depending_on_snapshot):
            errors.append(f"Weight update (DP all-reduce or backward) for iter{i+1} must depend on CF_SNAPSHOT_iter{i}")

    # 4) One-in-flight: when drain exists, it depends on prev persist
    for i in range(1, num_iterations):
        if i % model.checkpoint_every_n != 0:
            continue
        drain = name_to_node.get(f"DRAIN_iter{i}_npu{npu_id}")
        prev_persist = name_to_node.get(f"CF_PERSIST_iter{i-1}_npu{npu_id}")
        if drain and prev_persist and prev_persist.id not in drain.data_deps:
            errors.append(f"DRAIN_iter{i} must depend on CF_PERSIST_iter{i-1}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    print("checkfreq_like validation passed: snapshot/persist deps, FWD overlap, weight-update fence, one-in-flight drain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
