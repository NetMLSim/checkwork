#!/usr/bin/env python3
"""
Tests for the refactored CheckpointWrapper: validates correctness fixes,
low-risk extensions, new modes, and structural invariants.

These are deliberately self-contained (no astra-sim) so they run fast and can
serve as the wrapper's regression suite.

Run from CheckWork:
  python3 test_wrapper_new.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

MLSYNTH = Path(__file__).resolve().parent
if str(MLSYNTH) not in sys.path:
    sys.path.insert(0, str(MLSYNTH))

# Chakra import (skip gracefully if not installed)
try:
    from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # noqa: F401
except ImportError as e:
    print(f"SKIP: missing dependency ({e}). Install chakra and run from CheckWork.")
    sys.exit(0)

from Model.Transformer import Transformer
from Orchestrator.MegatronLM import MegatronLM
from Wrapper.CheckpointWrapper import (
    CheckpointWrapper,
    CheckpointInjection,
    StageCostModel,
    WrapperConfig,
)
from utils import reset_id_counter


_failures: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print("FAIL:", msg)


def _ok(msg: str) -> None:
    print("PASS:", msg)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _base_cfg(**wrapper_overrides):
    cfg = {
        "model": {
            "name": "transformer",
            "num_layers": 8,
            "sequence_len": 512,
            "vocab_size": 30000,
            "hidden_size": 1024,
            "batch_size": 32,
            "num_microbatches": 4,
            "bytes_per_val": 2,
            "scale": 1,
        },
        "parallelism": {"dp_size": 2, "pp_size": 2, "tp_size": 1},
        "num_iterations": 3,
        "wrapper": {"type": "checkpoint"},
    }
    cfg["wrapper"].update(wrapper_overrides)
    return cfg


class _FakeParent:
    """Fake parent node providing only the .id attribute."""
    def __init__(self, node_id: int):
        self.id = node_id


def _new_wrapper(**wrapper_overrides) -> CheckpointWrapper:
    cfg = _base_cfg(**wrapper_overrides)
    model = Transformer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return CheckpointWrapper(model, cfg)


def _names(inj: CheckpointInjection) -> list[str]:
    return [n.name for n in inj.nodes]


# -----------------------------------------------------------------------------
# 1. Validation (correctness fixes)
# -----------------------------------------------------------------------------
def test_validation_unknown_mode():
    try:
        _new_wrapper(mode="snyc")  # typo
        _fail("validation_unknown_mode: expected ValueError for typo mode 'snyc'")
    except ValueError:
        _ok("validation: rejects unknown mode")


def test_validation_unknown_remote_target():
    try:
        _new_wrapper(mode="remote_sync", remote_target="nope")
        _fail("validation_unknown_remote_target: expected ValueError")
    except ValueError:
        _ok("validation: rejects unknown remote_target")


def test_validation_checkpoint_every_n():
    for bad in (0, -1):
        try:
            _new_wrapper(mode="sync", checkpoint_every_n=bad)
            _fail(f"validation_checkpoint_every_n: expected ValueError for {bad}")
            return
        except ValueError:
            pass
    _ok("validation: rejects checkpoint_every_n <= 0")


def test_validation_max_inflight():
    try:
        _new_wrapper(mode="async", max_inflight=-1)
        _fail("validation_max_inflight: expected ValueError for -1")
    except ValueError:
        _ok("validation: rejects max_inflight < 0")


def test_validation_align_with():
    try:
        _new_wrapper(mode="sync", align_with="weekly")
        _fail("validation_align_with: expected ValueError")
    except ValueError:
        _ok("validation: rejects unknown align_with")


# -----------------------------------------------------------------------------
# 2. Deprecation warnings on legacy keys
# -----------------------------------------------------------------------------
def test_deprecation_warnings():
    cfg = _base_cfg(
        mode="checkfreq_like",
        snapshot_overhead_multiplier=1.0,
        persist_overhead_multiplier=5.0,
        snapshot_duration_micros=100,
        persist_duration_micros=500,
        kickoff_cost_micros=0,
    )
    model = Transformer(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CheckpointWrapper(model, cfg)
    legacy = {str(w.message).split(" ")[0] for w in caught
              if issubclass(w.category, DeprecationWarning)}
    for k in ("'kickoff_cost_micros'", "'snapshot_overhead_multiplier'",
              "'persist_overhead_multiplier'", "'snapshot_duration_micros'",
              "'persist_duration_micros'"):
        if k not in legacy:
            _fail(f"deprecation: missing warning for {k} (got {legacy})")
            return
    _ok("deprecation: warns on all legacy stage_cost keys")


def test_legacy_remote_rank0_alias():
    w = _new_wrapper(mode="remote_rank0", state_multiplier=1.0)
    assert w.mode == "remote_sync"
    assert w.cfg.remote_target == "rank0"
    assert w.checkpoint_boundary_all_nodes is True
    _ok("legacy: remote_rank0 alias maps to remote_sync + target rank0")


def test_legacy_use_storage_sink():
    w = _new_wrapper(mode="remote_async", use_storage_sink=True, state_multiplier=1.0)
    assert w.cfg.remote_target == "storage"
    assert w.uses_storage_rank is True
    _ok("legacy: use_storage_sink maps to remote_target=storage")


# -----------------------------------------------------------------------------
# 3. Symmetric overhead_multiplier across modes
# -----------------------------------------------------------------------------
def test_symmetric_overhead_multiplier():
    # With identical snapshot/persist costs and overhead_multiplier=2.0,
    # the per-byte sum in sync should equal snap+pers in checkfreq_like.
    common = dict(
        state_multiplier=1.0,
        snapshot_overhead_multiplier=1.0,
        persist_overhead_multiplier=5.0,
        overhead_multiplier=2.0,
        snapshot_duration_micros=100,
        persist_duration_micros=500,
        kickoff_cost_micros=1,  # non-zero so kickoff_cost != 0
    )
    p = _FakeParent(1)
    sw = _new_wrapper(mode="sync", **common)
    cw = _new_wrapper(mode="checkfreq_like", **common)
    sync_inj = sw.get_checkpoint_nodes(0, [p], iteration=0)
    cf_inj = cw.get_checkpoint_nodes(0, [p], iteration=0)
    # Sync emits 1 node; sum of its tensor_size = snap + pers (each *2 overhead)
    sync_total = sum(int(a.uint64_val) for a in sync_inj.nodes[0].attr if a.name == "tensor_size")
    # checkfreq_like: kickoff + snapshot + persist
    snap_node = next(n for n in cf_inj.nodes if n.name.startswith("CF_SNAPSHOT"))
    pers_node = next(n for n in cf_inj.nodes if n.name.startswith("CF_PERSIST"))
    cf_total = sum(int(a.uint64_val) for n in (snap_node, pers_node)
                   for a in n.attr if a.name == "tensor_size")
    if sync_total != cf_total:
        _fail(f"symmetric_overhead: sync_total={sync_total} != cf_total={cf_total}")
    else:
        _ok("symmetric_overhead: sync and checkfreq_like cost match under overhead=2")


# -----------------------------------------------------------------------------
# 4. checkpoint_bytes_respect_scale
# -----------------------------------------------------------------------------
def test_checkpoint_bytes_respect_scale():
    # scale shrinks bytes when respect_scale=True (default)
    cfg = _base_cfg(mode="sync", state_multiplier=1.0)
    cfg["model"]["scale"] = 0.1
    p = _FakeParent(1)
    m = Transformer(cfg)
    w_default = CheckpointWrapper(m, cfg)
    inj_default = w_default.get_checkpoint_nodes(0, [p], iteration=0)
    cost_default = next(int(a.uint64_val) for a in inj_default.nodes[0].attr if a.name == "tensor_size")
    # opt-out
    cfg2 = _base_cfg(mode="sync", state_multiplier=1.0, checkpoint_bytes_respect_scale=False)
    cfg2["model"]["scale"] = 0.1
    w_noscale = CheckpointWrapper(Transformer(cfg2), cfg2)
    inj_noscale = w_noscale.get_checkpoint_nodes(0, [p], iteration=0)
    cost_noscale = next(int(a.uint64_val) for a in inj_noscale.nodes[0].attr if a.name == "tensor_size")
    if cost_noscale <= cost_default * 5:  # cost_noscale should be ~10x larger
        _fail(f"checkpoint_bytes_respect_scale: noscale={cost_noscale} not >> default={cost_default}")
    else:
        _ok("checkpoint_bytes_respect_scale: false decouples scale from ckpt bytes")


# -----------------------------------------------------------------------------
# 5. max_inflight
# -----------------------------------------------------------------------------
def test_max_inflight_zero_no_chain():
    """max_inflight=0 -> background writes are independent (no chain)."""
    w = _new_wrapper(mode="async", state_multiplier=1.0, max_inflight=0)
    p = _FakeParent(1)
    inj0 = w.get_checkpoint_nodes(0, [p], iteration=0)
    inj1 = w.get_checkpoint_nodes(0, [p], iteration=1, prev_write_bg=inj0.bg_tail)
    write_bg1 = next(n for n in inj1.nodes if "WRITE_BG" in n.name)
    if inj0.bg_tail.id in list(write_bg1.data_deps):
        _fail("max_inflight=0: write_bg1 should not depend on bg_tail of iter 0")
    else:
        _ok("max_inflight=0: background writes are not chained")


def test_max_inflight_one_chained():
    w = _new_wrapper(mode="async", state_multiplier=1.0, max_inflight=1)
    p = _FakeParent(1)
    inj0 = w.get_checkpoint_nodes(0, [p], iteration=0)
    inj1 = w.get_checkpoint_nodes(0, [p], iteration=1, prev_write_bg=inj0.bg_tail)
    write_bg1 = next(n for n in inj1.nodes if "WRITE_BG" in n.name)
    if inj0.bg_tail.id not in list(write_bg1.data_deps):
        _fail("max_inflight=1: write_bg1 should chain on previous bg_tail")
    else:
        _ok("max_inflight=1: background writes are chained (legacy default)")


def test_max_inflight_window():
    """max_inflight=2 -> sliding window; only the oldest in the window blocks."""
    w = _new_wrapper(mode="async", state_multiplier=1.0, max_inflight=2)
    p = _FakeParent(1)
    injs = []
    bg_tail = None
    for it in range(4):
        inj = w.get_checkpoint_nodes(0, [p], iteration=it, prev_write_bg=bg_tail)
        injs.append(inj)
        bg_tail = inj.bg_tail
    # Iter 2 has 2 in-flight (iter0, iter1); iter2 should depend on iter0's bg (oldest in window)
    write_bg2 = next(n for n in injs[2].nodes if "WRITE_BG" in n.name)
    iter0_bg = injs[0].bg_tail
    iter1_bg = injs[1].bg_tail
    deps2 = list(write_bg2.data_deps)
    if iter0_bg.id not in deps2:
        _fail(f"max_inflight=2: iter2 should depend on iter0_bg ({iter0_bg.id}), deps={deps2}")
    elif iter1_bg.id in deps2:
        _fail(f"max_inflight=2: iter2 should NOT also chain iter1_bg; deps={deps2}")
    else:
        _ok("max_inflight=2: sliding window depends only on oldest in window")


# -----------------------------------------------------------------------------
# 6. Bandwidth-derived stage costs
# -----------------------------------------------------------------------------
def test_bandwidth_derived_costs():
    # 1e9 bytes at 1 GB/s = 1e9 / 1e9 * 1e6 micros = 1e6 micros
    # We give a tiny model so we control raw bytes via state_multiplier.
    w = _new_wrapper(
        mode="sync",
        state_multiplier=1.0,
        stage_costs={
            "snapshot_bandwidth_GBps": 1.0,
            "persist_bandwidth_GBps": 1.0,
        },
    )
    p = _FakeParent(1)
    inj = w.get_checkpoint_nodes(0, [p], iteration=0)
    dur = int(inj.nodes[0].duration_micros)
    # Derived duration must be > 100µs default (and finite); just sanity check.
    if dur < 100:
        _fail(f"bandwidth_derived_costs: duration {dur} unexpectedly low")
    else:
        _ok(f"bandwidth_derived_costs: BW=1GB/s yielded duration={dur}µs")


# -----------------------------------------------------------------------------
# 7. Compression / incremental
# -----------------------------------------------------------------------------
def test_compression_ratio():
    p = _FakeParent(1)
    baseline = _new_wrapper(mode="sync", state_multiplier=1.0).get_checkpoint_nodes(0, [p], 0)
    compressed = _new_wrapper(
        mode="sync", state_multiplier=1.0,
        stage_costs={"compression_ratio": 4.0},
    ).get_checkpoint_nodes(0, [p], 0)
    c_base = next(int(a.uint64_val) for a in baseline.nodes[0].attr if a.name == "tensor_size")
    c_comp = next(int(a.uint64_val) for a in compressed.nodes[0].attr if a.name == "tensor_size")
    if c_comp >= c_base:
        _fail(f"compression_ratio=4: cost should shrink, got {c_comp} vs {c_base}")
    else:
        _ok(f"compression_ratio=4: cost {c_base} -> {c_comp}")


def test_incremental_fraction():
    p = _FakeParent(1)
    baseline = _new_wrapper(mode="sync", state_multiplier=1.0).get_checkpoint_nodes(0, [p], 0)
    incr = _new_wrapper(
        mode="sync", state_multiplier=1.0,
        stage_costs={"incremental_fraction": 0.1},
    ).get_checkpoint_nodes(0, [p], 0)
    c_base = next(int(a.uint64_val) for a in baseline.nodes[0].attr if a.name == "tensor_size")
    c_incr = next(int(a.uint64_val) for a in incr.nodes[0].attr if a.name == "tensor_size")
    if c_incr >= c_base:
        _fail(f"incremental_fraction=0.1: cost should shrink, got {c_incr} vs {c_base}")
    else:
        _ok(f"incremental_fraction=0.1: cost {c_base} -> {c_incr}")


# -----------------------------------------------------------------------------
# 8. Per-event jitter (deterministic with seed)
# -----------------------------------------------------------------------------
def test_jitter_deterministic():
    p = _FakeParent(1)
    w1 = _new_wrapper(
        mode="sync", state_multiplier=1.0, seed=42,
        stage_costs={"cost_jitter_pct": 30.0},
    )
    w2 = _new_wrapper(
        mode="sync", state_multiplier=1.0, seed=42,
        stage_costs={"cost_jitter_pct": 30.0},
    )
    c1 = next(int(a.uint64_val) for a in w1.get_checkpoint_nodes(0, [p], 0).nodes[0].attr if a.name == "tensor_size")
    c2 = next(int(a.uint64_val) for a in w2.get_checkpoint_nodes(0, [p], 0).nodes[0].attr if a.name == "tensor_size")
    if c1 != c2:
        _fail(f"jitter_deterministic: same seed yielded different costs {c1} vs {c2}")
    else:
        _ok(f"jitter_deterministic: same seed reproduces cost {c1}")


def test_jitter_varies_across_iters():
    # Same seed, different iterations should yield different values (jitter applied each call).
    p = _FakeParent(1)
    w = _new_wrapper(
        mode="sync", state_multiplier=1.0, seed=7,
        stage_costs={"cost_jitter_pct": 50.0},
    )
    costs = []
    for it in range(5):
        c = next(int(a.uint64_val) for a in w.get_checkpoint_nodes(0, [p], it).nodes[0].attr if a.name == "tensor_size")
        costs.append(c)
    if len(set(costs)) <= 1:
        _fail(f"jitter_varies: costs all identical across iters: {costs}")
    else:
        _ok(f"jitter_varies: cost varies across iters under jitter; got {costs}")


# -----------------------------------------------------------------------------
# 9. Per-rank overrides
# -----------------------------------------------------------------------------
def test_rank_overrides():
    w = _new_wrapper(
        mode="sync", state_multiplier=1.0,
        rank_overrides={0: {"state_multiplier": 3.0}},
    )
    p = _FakeParent(1)
    inj_rank0 = w.get_checkpoint_nodes(0, [p], 0)
    inj_rank1 = w.get_checkpoint_nodes(1, [p], 0)
    c0 = next(int(a.uint64_val) for a in inj_rank0.nodes[0].attr if a.name == "tensor_size")
    c1 = next(int(a.uint64_val) for a in inj_rank1.nodes[0].attr if a.name == "tensor_size")
    if c0 <= c1:
        _fail(f"rank_overrides: rank0 cost {c0} should be > rank1 cost {c1}")
    else:
        _ok(f"rank_overrides: rank0 {c0} > rank1 {c1} (state_multiplier override applied)")


# -----------------------------------------------------------------------------
# 10. align_with epoch
# -----------------------------------------------------------------------------
def test_align_with_epoch():
    # checkpoint_every_n=1 * iterations_per_epoch=3 -> trigger every 3 iters
    w = _new_wrapper(
        mode="sync", checkpoint_every_n=1,
        align_with="epoch", iterations_per_epoch=3, state_multiplier=1.0,
    )
    p = _FakeParent(1)
    triggered = []
    for it in range(6):
        inj = w.get_checkpoint_nodes(0, [p], iteration=it)
        triggered.append(bool(inj.nodes))
    expected = [True, False, False, True, False, False]
    if triggered != expected:
        _fail(f"align_with_epoch: got {triggered}, expected {expected}")
    else:
        _ok("align_with_epoch: cadence multiplied by iterations_per_epoch")


# -----------------------------------------------------------------------------
# 11. comm_tag_base
# -----------------------------------------------------------------------------
def test_comm_tag_base():
    w_default = _new_wrapper(mode="remote_sync", state_multiplier=1.0)
    w_custom = _new_wrapper(mode="remote_sync", state_multiplier=1.0, comm_tag_base=42_000)
    p = _FakeParent(1)
    inj_default = w_default.get_checkpoint_nodes(1, [p], 0)
    inj_custom = w_custom.get_checkpoint_nodes(1, [p], 0)
    snd_default = next(n for n in inj_default.nodes if "COMM_SEND" in n.name)
    snd_custom = next(n for n in inj_custom.nodes if "COMM_SEND" in n.name)
    tag_default = next(int(a.int32_val) for a in snd_default.attr if a.name == "comm_tag")
    tag_custom = next(int(a.int32_val) for a in snd_custom.attr if a.name == "comm_tag")
    if tag_default != 8000 or tag_custom != 42_000:
        _fail(f"comm_tag_base: default={tag_default} (want 8000) custom={tag_custom} (want 42000)")
    else:
        _ok("comm_tag_base: default 8000 and custom 42000 honoured")


# -----------------------------------------------------------------------------
# 12. storage_rank_id
# -----------------------------------------------------------------------------
def test_storage_rank_id_default_and_override():
    w_default = _new_wrapper(mode="remote_async", remote_target="storage", state_multiplier=1.0)
    assert w_default.storage_rank_id == w_default.num_npus, (
        f"storage_rank_id default should be num_npus, got {w_default.storage_rank_id}"
    )
    w_custom = _new_wrapper(mode="remote_async", remote_target="storage",
                            state_multiplier=1.0, storage_rank_id=999)
    assert w_custom.storage_rank_id == 999
    # uses_storage_rank flag picked up
    assert w_default.uses_storage_rank is True
    _ok("storage_rank_id: default = num_npus; explicit override works")


def test_storage_rank_added_to_comm_groups():
    cfg = _base_cfg(mode="remote_async", remote_target="storage", state_multiplier=1.0)
    model = CheckpointWrapper(Transformer(cfg), cfg)
    orch = MegatronLM(model, cfg)
    groups = orch.generate_comm_groups()
    # The wrapper's storage rank id == num_npus by default
    storage_id = model.storage_rank_id
    found = any(storage_id in v for v in groups.values())
    if not found:
        _fail(f"storage_rank: id {storage_id} not in any comm group {groups}")
    else:
        _ok("storage_rank: registered as a comm group by the orchestrator")


# -----------------------------------------------------------------------------
# 13. CheckpointInjection: dataclass + iterable + boundary_nodes
# -----------------------------------------------------------------------------
def test_checkpoint_injection_iterable():
    w = _new_wrapper(mode="sync", state_multiplier=1.0)
    p = _FakeParent(1)
    inj = w.get_checkpoint_nodes(0, [p], 0)
    # New-style dataclass attributes
    assert hasattr(inj, "nodes")
    assert hasattr(inj, "bg_tail")
    assert hasattr(inj, "snapshot_fence")
    assert hasattr(inj, "boundary_nodes")
    # Legacy 3-tuple unpack still works
    a, b, c = inj
    assert a is inj.nodes
    _ok("CheckpointInjection: dataclass + legacy 3-tuple unpack")


def test_boundary_nodes_remote_sync():
    w = _new_wrapper(mode="remote_sync", state_multiplier=1.0)
    p = _FakeParent(1)
    inj = w.get_checkpoint_nodes(1, [p], 0)  # rank 1: kickoff+send (ring)
    if not inj.boundary_nodes:
        _fail("boundary_nodes_remote_sync: expected boundary_nodes to be set")
        return
    if list(inj.boundary_nodes) != list(inj.nodes):
        _fail("boundary_nodes_remote_sync: should equal full nodes list for ring fence")
    else:
        _ok("boundary_nodes_remote_sync: contains full nodes list")


# -----------------------------------------------------------------------------
# 14. New modes: tiered, pipelined
# -----------------------------------------------------------------------------
def test_tiered_mode_two_stages():
    w = _new_wrapper(
        mode="tiered", state_multiplier=1.0,
        stages=[
            {"kind": "snapshot", "every_n": 1, "destination": "local", "on_critical_path": True},
            {"kind": "persist", "every_n": 3, "destination": "storage", "on_critical_path": False},
        ],
    )
    assert w.uses_storage_rank is True
    p = _FakeParent(1)
    inj0 = w.get_checkpoint_nodes(0, [p], 0)
    inj1 = w.get_checkpoint_nodes(0, [p], 1)
    inj3 = w.get_checkpoint_nodes(0, [p], 3)
    # iter 0: both stages trigger
    stage_kinds_0 = [n.name for n in inj0.nodes]
    if not any("s0_snapshot" in n for n in stage_kinds_0):
        _fail("tiered: snapshot stage missing at iter 0")
        return
    if not any("s1_persist" in n for n in stage_kinds_0):
        _fail("tiered: persist stage missing at iter 0")
        return
    # iter 1: only snapshot
    if any("s1_persist" in n.name for n in inj1.nodes):
        _fail("tiered: persist stage should NOT fire at iter 1")
        return
    if not any("s0_snapshot" in n.name for n in inj1.nodes):
        _fail("tiered: snapshot stage should fire at iter 1")
        return
    # iter 3: both
    if not any("s1_persist" in n.name for n in inj3.nodes):
        _fail("tiered: persist stage should fire at iter 3")
        return
    # snapshot stage marked on_critical_path -> snapshot_fence set, boundary_nodes set
    if inj0.snapshot_fence is None:
        _fail("tiered: snapshot_fence should be set when snapshot is on_critical_path")
        return
    _ok("tiered: stages fire at their own cadence; snapshot_fence + boundary set correctly")


def test_pipelined_mode_alias():
    w = _new_wrapper(mode="pipelined", state_multiplier=1.0,
                     snapshot_overhead_multiplier=1.0,
                     persist_overhead_multiplier=5.0,
                     snapshot_duration_micros=100, persist_duration_micros=500,
                     kickoff_cost_micros=0)
    p = _FakeParent(1)
    inj = w.get_checkpoint_nodes(0, [p], 0)
    expected = {"CF_KICKOFF_iter0_npu0", "CF_SNAPSHOT_iter0_npu0", "CF_PERSIST_iter0_npu0"}
    if not expected.issubset({n.name for n in inj.nodes}):
        _fail(f"pipelined: expected CF_* nodes; got {[n.name for n in inj.nodes]}")
    else:
        _ok("pipelined: emits same CF nodes as checkfreq_like")


# -----------------------------------------------------------------------------
# 15. Base generator without a wrapper still works
# -----------------------------------------------------------------------------
def test_mlsynth_without_wrapper():
    cfg = _base_cfg()
    cfg.pop("wrapper", None)
    model = Transformer(cfg)
    orch = MegatronLM(model, cfg)
    orch.generate_comm_groups()
    nodes = orch.exec()
    # We have num_npus = 4 ranks; each should have GlobalMetadata + at least
    # a handful of FWD/BWD nodes.
    if len(nodes) != 4:
        _fail(f"mlsynth_no_wrapper: expected 4 ranks, got {len(nodes)}")
        return
    for npu_id, node_list in nodes.items():
        if len(node_list) < 5:
            _fail(f"mlsynth_no_wrapper: rank {npu_id} only has {len(node_list)} nodes")
            return
    _ok("mlsynth_no_wrapper: orchestrator runs with bare Transformer (no wrapper)")


# -----------------------------------------------------------------------------
# 16. reset_id_counter isolates simulations
# -----------------------------------------------------------------------------
def test_reset_id_counter():
    from utils import next_id, reset_id_counter
    reset_id_counter(0)
    a = next_id()
    b = next_id()
    reset_id_counter(0)
    c = next_id()
    if (a, b, c) != (0, 1, 0):
        _fail(f"reset_id_counter: got ({a},{b},{c}); expected (0,1,0)")
    else:
        _ok("reset_id_counter: resets the module-global node id counter")


# -----------------------------------------------------------------------------
# 17. Node naming convention (centralised)
# -----------------------------------------------------------------------------
def test_node_naming():
    from Wrapper.CheckpointWrapper import _name
    assert _name("FOO", 3, 7) == "FOO_iter3_npu7"
    assert _name("RECV", 1, 0, "from2") == "RECV_iter1_from2_npu0"
    _ok("node naming: prefix_iter{i}[_extra]_npu{n}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_validation_unknown_mode,
        test_validation_unknown_remote_target,
        test_validation_checkpoint_every_n,
        test_validation_max_inflight,
        test_validation_align_with,
        test_deprecation_warnings,
        test_legacy_remote_rank0_alias,
        test_legacy_use_storage_sink,
        test_symmetric_overhead_multiplier,
        test_checkpoint_bytes_respect_scale,
        test_max_inflight_zero_no_chain,
        test_max_inflight_one_chained,
        test_max_inflight_window,
        test_bandwidth_derived_costs,
        test_compression_ratio,
        test_incremental_fraction,
        test_jitter_deterministic,
        test_jitter_varies_across_iters,
        test_rank_overrides,
        test_align_with_epoch,
        test_comm_tag_base,
        test_storage_rank_id_default_and_override,
        test_storage_rank_added_to_comm_groups,
        test_checkpoint_injection_iterable,
        test_boundary_nodes_remote_sync,
        test_tiered_mode_two_stages,
        test_pipelined_mode_alias,
        test_mlsynth_without_wrapper,
        test_reset_id_counter,
        test_node_naming,
    ]
    for t in tests:
        reset_id_counter(0)
        try:
            t()
        except AssertionError as e:
            _fail(f"{t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            _fail(f"{t.__name__}: unexpected {type(e).__name__}: {e}")
    print()
    print(f"PASSED: {len(tests) - len(_failures)}/{len(tests)}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
