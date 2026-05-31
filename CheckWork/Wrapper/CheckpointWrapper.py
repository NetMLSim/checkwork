"""
Checkpoint-aware Chakra ET injection (CheckWork core component).

Supported modes
---------------
  sync            Snapshot+persist as one blocking node on the critical path.
  async           Local kickoff + background write (WRITE_BG). max_inflight
                  controls how many WRITE_BG nodes may overlap.
  remote_sync     Kickoff + remote transfer; next iteration waits on every
                  returned node. remote_target: ring | rank0 | storage.
  remote_async    Kickoff + background remote transfer. remote_target as above.
  checkfreq_like  Two-phase snapshot/persist; snapshot fences the next weight
                  update, persist runs in background.
  pipelined       Same shape as checkfreq_like, defaults max_inflight=4 to
                  approximate PCcheck-style multiple concurrent persists.
  tiered          User-defined list of stages, each with its own cadence and
                  destination. Approximates Gemini-style multi-tier
                  checkpointing.

Configuration
-------------
The wrapper reads the ``wrapper:`` block of the workload YAML.

  Top-level keys (most users only need a handful):
    mode                              str, required
    remote_target                     "ring" | "rank0" | "storage"
                                      (only for remote_*)
    checkpoint_every_n                int > 0
    state_multiplier                  float >= 0
    overhead_multiplier               float, scales every per-byte cost
    max_inflight                      int >= 0 (0 = unlimited, 1 = serialise)
    drain_when_backpressure           bool (deprecated alias for
                                      max_inflight=1 with an explicit drain node)
    persist_in_background_no_graph    bool (checkfreq_like: omit the persist
                                      node entirely, only snapshot on graph)
    align_with                        "iteration" | "epoch"
    iterations_per_epoch              int >= 1 (used when align_with=epoch)
    comm_tag_base                     int (default 8000; per-iter tag = base + i)
    storage_rank_id                   int (default: num_npus)
    checkpoint_bytes_respect_scale    bool (default True; if False, the
                                      model's ``scale`` field does not shrink
                                      checkpoint bytes)
    seed                              int (RNG seed for cost jitter)
    rank_overrides                    {npu_id: {state_multiplier?, ...cost
                                      knobs...}}
    stages                            list[dict] (tiered mode)

  stage_costs (per-stage cost model; everything optional):
    snapshot_multiplier               float (default 1.0)
    persist_multiplier                float (default 5.0)
    write_bg_multiplier               float (default 1.0)
    snapshot_micros                   int  (override snapshot duration)
    persist_micros                    int  (override persist duration)
    sync_total_micros                 int  (override the sync-mode total)
    snapshot_bandwidth_GBps           float (derive snapshot duration from
                                      bytes / BW; ignored if snapshot_micros set)
    persist_bandwidth_GBps            float (same, for persist)
    kickoff_micros                    int (default 100)
    kickoff_flops_cap                 int (default 500_000)
    write_bg_flops_cap                int (default 1_000_000)
    snapshot_flops_cap                int | None (default None = no cap)
    persist_flops_cap                 int | None (default None = no cap)
    cost_jitter_pct                   float (lognormal stdev as % of mean)
    compression_ratio                 float (output_bytes = input_bytes/ratio)
    compression_cost_per_byte_micros  float (added to snapshot duration)
    incremental_fraction              float (fraction of state written)

Legacy keys (still accepted, emit DeprecationWarning):
    snapshot_overhead_multiplier, persist_overhead_multiplier,
    snapshot_duration_micros, persist_duration_micros, sync_duration_micros,
    kickoff_cost_micros, use_storage_sink, mode=remote_rank0,
    mode=remote_rank0_async

Return contract
---------------
``get_checkpoint_nodes`` returns a :class:`CheckpointInjection` dataclass.
For backward compatibility it is also unpackable as a 3-tuple
``(nodes, bg_tail, snapshot_fence)``.
"""

from __future__ import annotations

import dataclasses
import random
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from Wrapper.Wrapper import Wrapper
from utils import compute, send, receive
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode


# ----------------------------------------------------------------------------
# Node naming. Names are preserved bit-identically with the pre-refactor
# wrapper so that existing tests/experiments grepping by name keep working.
# ----------------------------------------------------------------------------

# COMP_NODE prefixes
_N_SYNC = "COMP_NODE_CHECKPOINT_SAVE"
_N_CF_KICKOFF = "CF_KICKOFF"
_N_CF_SNAPSHOT = "CF_SNAPSHOT"
_N_CF_PERSIST = "CF_PERSIST"
_N_CF_DRAIN = "DRAIN"
_N_ASYNC_KICKOFF = "COMP_NODE_CHECKPOINT_KICKOFF"
_N_ASYNC_WRITE_BG = "COMP_NODE_CHECKPOINT_WRITE_BG"
_N_ASYNC_DRAIN = "COMP_NODE_CHECKPOINT_DRAIN"
_N_REMOTE_KICKOFF = "COMP_NODE_CHECKPOINT_REMOTE_KICKOFF"
_N_REMOTE_DRAIN = "COMP_NODE_CHECKPOINT_REMOTE_DRAIN"
_N_TIERED_STAGE_COMP = "TIERED_STAGE_COMP"

# COMM prefixes (remote_sync)
_N_REMOTE_SYNC_RING_SEND = "COMM_SEND_NODE_CHECKPOINT_REMOTE"
_N_REMOTE_SYNC_RING_RECV = "COMM_RECV_NODE_CHECKPOINT_REMOTE"
_N_REMOTE_SYNC_RANK0_SEND = "COMM_SEND_NODE_CHECKPOINT_RANK0"
_N_REMOTE_SYNC_RANK0_RECV = "COMM_RECV_NODE_CHECKPOINT_RANK0"
_N_REMOTE_SYNC_STORAGE_SEND = "COMM_SEND_NODE_CHECKPOINT_REMOTE_SYNC"
# Pre-refactor we emitted a trivial 1-flop "sync_join" node so nodes[0]
# could double as the boundary. The new boundary_nodes mechanism makes it
# redundant but we keep emitting it (cost 1, dur 1) so any prior experiment
# trace remains bit-identical.
_N_REMOTE_SYNC_JOIN = "COMP_NODE_CHECKPOINT_REMOTE_SYNC_JOIN"

# COMM prefixes (remote_async)
_N_REMOTE_ASYNC_RING_SEND = "COMM_SEND_NODE_CHECKPOINT_REMOTE_ASYNC"
_N_REMOTE_ASYNC_RING_RECV = "COMM_RECV_NODE_CHECKPOINT_REMOTE_ASYNC"
_N_REMOTE_ASYNC_RANK0_SEND = "COMM_SEND_NODE_CHECKPOINT_RANK0_ASYNC"
_N_REMOTE_ASYNC_RANK0_RECV = "COMM_RECV_NODE_CHECKPOINT_RANK0_ASYNC"
_N_TIERED_STAGE_SEND = "TIERED_STAGE_SEND"
_N_TIERED_STAGE_RECV = "TIERED_STAGE_RECV"


def _name(prefix: str, iteration: int, npu_id: int, *extras: Any) -> str:
    """Build a Chakra node name following the project convention.

    ``prefix_iter{i}[_extra_...]_npu{n}``
    """
    extra_str = ("_" + "_".join(str(x) for x in extras)) if extras else ""
    return f"{prefix}_iter{iteration}{extra_str}_npu{npu_id}"


# ----------------------------------------------------------------------------
# Public return type
# ----------------------------------------------------------------------------
@dataclass
class CheckpointInjection:
    """Result of one ``get_checkpoint_nodes`` call.

    Attributes:
        nodes:            All Chakra nodes the orchestrator should append for
                          this iteration.
        bg_tail:          The "background" tail node the next checkpoint must
                          fence on (e.g. previous write_bg, previous persist,
                          previous upload send/recv). Implements max_inflight.
        snapshot_fence:   For checkfreq_like / tiered with a CPU-mem snapshot
                          stage: the snapshot node the next iteration's weight
                          update must wait for. None otherwise.
        boundary_nodes:   Nodes the *next iteration's first compute* must
                          depend on. If None, the orchestrator uses
                          ``[nodes[0]]`` (the historical default).

    The dataclass is also iterable so that legacy callers doing
    ``nodes, bg, fence = wrapper.get_checkpoint_nodes(...)`` keep working.
    """

    nodes: List[ChakraNode] = field(default_factory=list)
    bg_tail: Optional[ChakraNode] = None
    snapshot_fence: Optional[ChakraNode] = None
    boundary_nodes: Optional[List[ChakraNode]] = None

    def __iter__(self):
        yield self.nodes
        yield self.bg_tail
        yield self.snapshot_fence


# ----------------------------------------------------------------------------
# Stage cost model
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Mode-specific historical kickoff caps. Used when the user has not set
# ``stage_costs.kickoff_flops_cap``. Lifted here so anyone can override (per
# the audit request) without changing existing behaviour.
# ----------------------------------------------------------------------------
_CF_KICKOFF_CAP_DEFAULT = 100_000      # checkfreq_like / pipelined kickoff
_ASYNC_KICKOFF_CAP_DEFAULT = 1_000_000  # local async kickoff (== bg cap)
_REMOTE_KICKOFF_CAP_DEFAULT = 500_000   # remote_sync / remote_async kickoff


@dataclass
class StageCostModel:
    """Per-stage cost parameters. Decoupled from wrapper config so per-rank
    overrides can derive a modified instance via :func:`dataclasses.replace`.
    """

    snapshot_multiplier: float = 1.0
    persist_multiplier: float = 5.0
    write_bg_multiplier: float = 1.0
    snapshot_micros: Optional[int] = None
    persist_micros: Optional[int] = None
    sync_total_micros: Optional[int] = None
    snapshot_bandwidth_GBps: Optional[float] = None
    persist_bandwidth_GBps: Optional[float] = None
    kickoff_micros: int = 100
    # None means "use mode-specific historical default" (see constants above).
    kickoff_flops_cap: Optional[int] = None
    # None means "no cap on background write flops" (matches historical async).
    write_bg_flops_cap: Optional[int] = None
    snapshot_flops_cap: Optional[int] = None
    persist_flops_cap: Optional[int] = None
    cost_jitter_pct: float = 0.0
    compression_ratio: float = 1.0
    compression_cost_per_byte_micros: float = 0.0
    incremental_fraction: float = 1.0

    def __post_init__(self) -> None:
        for fname in ("snapshot_multiplier", "persist_multiplier", "write_bg_multiplier"):
            if getattr(self, fname) < 0:
                raise ValueError(f"{fname} must be >= 0 (got {getattr(self, fname)})")
        if self.compression_ratio <= 0:
            raise ValueError(f"compression_ratio must be > 0 (got {self.compression_ratio})")
        if self.incremental_fraction < 0:
            raise ValueError(f"incremental_fraction must be >= 0 (got {self.incremental_fraction})")
        if self.cost_jitter_pct < 0:
            raise ValueError(f"cost_jitter_pct must be >= 0 (got {self.cost_jitter_pct})")
        if self.kickoff_micros < 0:
            raise ValueError(f"kickoff_micros must be >= 0 (got {self.kickoff_micros})")

    def effective_bytes(self, raw_bytes: int) -> int:
        eff = raw_bytes * self.incremental_fraction / self.compression_ratio
        return max(int(eff), 1)

    def _apply_jitter(self, value: float, rng: Optional[random.Random]) -> float:
        if self.cost_jitter_pct <= 0 or rng is None:
            return value
        sigma = self.cost_jitter_pct / 100.0
        mu = -0.5 * sigma * sigma  # so E[lognorm] = 1
        m = rng.lognormvariate(mu, sigma)
        m = max(0.01, m)
        return value * m

    def snapshot_cost_dur(self, raw_bytes: int, overhead_mult: float,
                          rng: Optional[random.Random] = None) -> Tuple[int, int]:
        eff = self.effective_bytes(raw_bytes)
        cost = eff * self.snapshot_multiplier * overhead_mult
        if self.snapshot_micros is not None:
            dur = float(self.snapshot_micros) * overhead_mult
        elif self.snapshot_bandwidth_GBps is not None:
            dur = max(1.0, eff / (self.snapshot_bandwidth_GBps * 1e9) * 1e6) * overhead_mult
        else:
            dur = 100.0 * self.snapshot_multiplier * overhead_mult
        if self.compression_cost_per_byte_micros > 0:
            dur += self.compression_cost_per_byte_micros * eff * overhead_mult
        cost = self._apply_jitter(cost, rng)
        dur = self._apply_jitter(dur, rng)
        cost_i = max(1, int(cost))
        if self.snapshot_flops_cap is not None:
            cost_i = min(cost_i, int(self.snapshot_flops_cap))
        return cost_i, max(1, int(dur))

    def persist_cost_dur(self, raw_bytes: int, overhead_mult: float,
                         rng: Optional[random.Random] = None) -> Tuple[int, int]:
        eff = self.effective_bytes(raw_bytes)
        cost = eff * self.persist_multiplier * overhead_mult
        if self.persist_micros is not None:
            dur = float(self.persist_micros) * overhead_mult
        elif self.persist_bandwidth_GBps is not None:
            dur = max(1.0, eff / (self.persist_bandwidth_GBps * 1e9) * 1e6) * overhead_mult
        else:
            dur = 100.0 * self.persist_multiplier * overhead_mult
        cost = self._apply_jitter(cost, rng)
        dur = self._apply_jitter(dur, rng)
        cost_i = max(1, int(cost))
        if self.persist_flops_cap is not None:
            cost_i = min(cost_i, int(self.persist_flops_cap))
        return cost_i, max(1, int(dur))

    def write_bg_cost_dur(self, raw_bytes: int, overhead_mult: float,
                          rng: Optional[random.Random] = None) -> Tuple[int, int]:
        """Local async background write. Historically used overhead_mult as the
        sole cost factor (the dedicated write_bg_multiplier defaults to 1.0 to
        preserve that). Duration: 100µs * write_bg_multiplier * overhead_mult."""
        eff = self.effective_bytes(raw_bytes)
        cost = eff * self.write_bg_multiplier * overhead_mult
        dur = 100.0 * self.write_bg_multiplier * overhead_mult
        cost = self._apply_jitter(cost, rng)
        dur = self._apply_jitter(dur, rng)
        cost_i = max(1, int(cost))
        if self.write_bg_flops_cap is not None:
            cost_i = min(cost_i, int(self.write_bg_flops_cap))
        return cost_i, max(1, int(dur))

    def resolved_kickoff_cap(self, mode_default: int) -> int:
        return int(self.kickoff_flops_cap if self.kickoff_flops_cap is not None else mode_default)


# ----------------------------------------------------------------------------
# Wrapper configuration
# ----------------------------------------------------------------------------
_VALID_MODES = {
    "sync", "async", "remote_sync", "remote_async",
    "checkfreq_like", "tiered", "pipelined",
}
_VALID_REMOTE_TARGETS = {"ring", "rank0", "storage"}
_VALID_ALIGN = {"iteration", "epoch"}
_VALID_STAGE_KINDS = {"snapshot", "persist", "write_bg"}
_VALID_STAGE_DESTINATIONS = {"local", "ring", "rank0", "storage"}

_LEGACY_KEY_MIGRATIONS = {
    "kickoff_cost_micros": "stage_costs.kickoff_micros",
    "snapshot_overhead_multiplier": "stage_costs.snapshot_multiplier",
    "persist_overhead_multiplier": "stage_costs.persist_multiplier",
    "snapshot_duration_micros": "stage_costs.snapshot_micros",
    "persist_duration_micros": "stage_costs.persist_micros",
    "sync_duration_micros": "stage_costs.sync_total_micros",
}


def _coerce_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


@dataclass
class WrapperConfig:
    """Parsed and validated wrapper configuration."""

    mode: str = "sync"
    remote_target: str = "ring"
    checkpoint_every_n: int = 1
    state_multiplier: float = 3.0
    overhead_multiplier: float = 1.0
    persist_in_background_no_graph: bool = False
    max_inflight: int = 1
    align_with: str = "iteration"
    iterations_per_epoch: int = 1
    comm_tag_base: int = 8000
    storage_rank_id: Optional[int] = None
    checkpoint_bytes_respect_scale: bool = True
    seed: Optional[int] = None
    rank_overrides: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    base_costs: StageCostModel = field(default_factory=StageCostModel)

    @classmethod
    def from_dict(cls, wc: dict) -> "WrapperConfig":
        # -- mode + remote_target with legacy aliases ------------------------
        raw_mode = str(wc.get("mode", "sync")).lower()
        legacy_target: Optional[str] = None
        if raw_mode == "remote_rank0":
            warnings.warn(
                "mode 'remote_rank0' is deprecated; use mode: remote_sync, "
                "remote_target: rank0",
                DeprecationWarning, stacklevel=3,
            )
            raw_mode = "remote_sync"
            legacy_target = "rank0"
        elif raw_mode == "remote_rank0_async":
            warnings.warn(
                "mode 'remote_rank0_async' is deprecated; use mode: remote_async, "
                "remote_target: rank0",
                DeprecationWarning, stacklevel=3,
            )
            raw_mode = "remote_async"
            legacy_target = "rank0"

        if raw_mode not in _VALID_MODES:
            raise ValueError(
                f"unknown checkpoint mode '{raw_mode}'. "
                f"Valid: {sorted(_VALID_MODES)}"
            )

        rt = legacy_target if legacy_target else str(wc.get("remote_target", "ring")).lower()
        if wc.get("use_storage_sink", False):
            warnings.warn(
                "'use_storage_sink' is deprecated; set remote_target: storage",
                DeprecationWarning, stacklevel=3,
            )
            if raw_mode == "remote_async":
                rt = "storage"
        if raw_mode in ("remote_sync", "remote_async") and rt not in _VALID_REMOTE_TARGETS:
            raise ValueError(
                f"unknown remote_target '{rt}'. "
                f"Valid: {sorted(_VALID_REMOTE_TARGETS)}"
            )

        # -- cadence -----------------------------------------------------------
        every_n = int(wc.get("checkpoint_every_n", 1))
        if every_n <= 0:
            raise ValueError(f"checkpoint_every_n must be > 0 (got {every_n})")
        align = str(wc.get("align_with", "iteration")).lower()
        if align not in _VALID_ALIGN:
            raise ValueError(f"align_with must be one of {sorted(_VALID_ALIGN)} (got {align})")
        iters_per_epoch = int(wc.get("iterations_per_epoch", 1))
        if iters_per_epoch < 1:
            raise ValueError(f"iterations_per_epoch must be >= 1 (got {iters_per_epoch})")

        # -- stage_costs (with legacy fallbacks) ------------------------------
        sc = wc.get("stage_costs") or {}
        for legacy_key, new_key in _LEGACY_KEY_MIGRATIONS.items():
            if legacy_key in wc:
                warnings.warn(
                    f"'{legacy_key}' is deprecated; use '{new_key}' under stage_costs",
                    DeprecationWarning, stacklevel=3,
                )

        base = StageCostModel(
            snapshot_multiplier=float(sc.get(
                "snapshot_multiplier",
                wc.get("snapshot_overhead_multiplier", 1.0),
            )),
            persist_multiplier=float(sc.get(
                "persist_multiplier",
                wc.get("persist_overhead_multiplier", 5.0),
            )),
            write_bg_multiplier=float(sc.get(
                "write_bg_multiplier",
                wc.get("overhead_multiplier", 1.0),
            )),
            snapshot_micros=_coerce_int_or_none(sc.get(
                "snapshot_micros",
                wc.get("snapshot_duration_micros"),
            )),
            persist_micros=_coerce_int_or_none(sc.get(
                "persist_micros",
                wc.get("persist_duration_micros"),
            )),
            sync_total_micros=_coerce_int_or_none(sc.get(
                "sync_total_micros",
                wc.get("sync_duration_micros"),
            )),
            snapshot_bandwidth_GBps=_coerce_float_or_none(sc.get("snapshot_bandwidth_GBps")),
            persist_bandwidth_GBps=_coerce_float_or_none(sc.get("persist_bandwidth_GBps")),
            kickoff_micros=int(sc.get(
                "kickoff_micros",
                wc.get("kickoff_cost_micros", 100),
            )),
            kickoff_flops_cap=_coerce_int_or_none(sc.get("kickoff_flops_cap")),
            write_bg_flops_cap=_coerce_int_or_none(sc.get("write_bg_flops_cap")),
            snapshot_flops_cap=_coerce_int_or_none(sc.get("snapshot_flops_cap")),
            persist_flops_cap=_coerce_int_or_none(sc.get("persist_flops_cap")),
            cost_jitter_pct=float(sc.get("cost_jitter_pct", 0.0)),
            compression_ratio=float(sc.get("compression_ratio", 1.0)),
            compression_cost_per_byte_micros=float(sc.get("compression_cost_per_byte_micros", 0.0)),
            incremental_fraction=float(sc.get("incremental_fraction", 1.0)),
        )

        # -- max_inflight (back-compat with drain_when_backpressure) ---------
        drain = bool(wc.get("drain_when_backpressure", False))
        if "max_inflight" in wc:
            max_inflight = int(wc["max_inflight"])
        else:
            # Legacy default: chain every background write (max_inflight=1).
            # drain_when_backpressure only added the explicit DRAIN node; the
            # chain-by-prev_write_bg has been the historical behaviour.
            max_inflight = 1
        if max_inflight < 0:
            raise ValueError(f"max_inflight must be >= 0 (got {max_inflight})")

        # -- stages (tiered mode) --------------------------------------------
        stages = list(wc.get("stages", []) or [])
        if raw_mode == "tiered":
            if not stages:
                raise ValueError("tiered mode requires a non-empty 'stages' list")
            for i, s in enumerate(stages):
                kind = str(s.get("kind", "")).lower()
                if kind not in _VALID_STAGE_KINDS:
                    raise ValueError(
                        f"stages[{i}].kind must be one of {sorted(_VALID_STAGE_KINDS)} "
                        f"(got '{kind}')"
                    )
                dest = str(s.get("destination", "local")).lower()
                if dest not in _VALID_STAGE_DESTINATIONS:
                    raise ValueError(
                        f"stages[{i}].destination must be one of "
                        f"{sorted(_VALID_STAGE_DESTINATIONS)} (got '{dest}')"
                    )
                if int(s.get("every_n", 1)) <= 0:
                    raise ValueError(f"stages[{i}].every_n must be > 0")

        return cls(
            mode=raw_mode,
            remote_target=rt,
            checkpoint_every_n=every_n,
            state_multiplier=float(wc.get("state_multiplier", 3.0)),
            overhead_multiplier=float(wc.get("overhead_multiplier", 1.0)),
            persist_in_background_no_graph=bool(wc.get("persist_in_background_no_graph", False)),
            max_inflight=max_inflight,
            align_with=align,
            iterations_per_epoch=iters_per_epoch,
            comm_tag_base=int(wc.get("comm_tag_base", 8000)),
            storage_rank_id=_coerce_int_or_none(wc.get("storage_rank_id")),
            checkpoint_bytes_respect_scale=bool(wc.get("checkpoint_bytes_respect_scale", True)),
            seed=_coerce_int_or_none(wc.get("seed")),
            rank_overrides=dict(wc.get("rank_overrides") or {}),
            stages=stages,
            base_costs=base,
        )


# ----------------------------------------------------------------------------
# CheckpointWrapper
# ----------------------------------------------------------------------------
class CheckpointWrapper(Wrapper):
    """
    Injects checkpoint-related Chakra nodes at end-of-iteration boundaries.

    See module docstring for the full configuration reference.
    """

    def __init__(self, model, config: dict):
        self.model = model
        self.num_params = model.num_params
        self.config = config
        wc = config.get("wrapper", {}) or {}

        # Parse + validate once. ValueErrors here are user-visible.
        self.cfg = WrapperConfig.from_dict(wc)

        # Parallelism / model metadata.
        self.tp_size = config["parallelism"]["tp_size"]
        self.pp_size = config["parallelism"]["pp_size"]
        self.dp_size = config["parallelism"]["dp_size"]
        self.num_npus = self.dp_size * self.pp_size * self.tp_size
        self.scale = float(config["model"].get("scale", 1.0))

        # Cache hot-path model values.
        self._bytes_per_val = self.model.get_bytes_per_val()
        self._params_per_rank = self.num_params / (self.tp_size * self.pp_size)

        # Storage rank id (used by remote_*/storage and tiered/storage).
        self.storage_rank_id = (
            int(self.cfg.storage_rank_id)
            if self.cfg.storage_rank_id is not None
            else self.num_npus
        )

        # Per-rank RNG (deterministic if seed is set).
        self._rng_by_npu: Dict[int, random.Random] = {}

        # Per-rank background in-flight history (for max_inflight > 1).
        # Keyed by npu_id; deque of the last (max_inflight) bg_tail nodes.
        # We use a list because Python deque needs maxlen, but the wrapper
        # is also called with prev_write_bg from the orchestrator and we want
        # to interoperate with that. See _push_bg() / _bg_chain_parent().
        self._bg_history: Dict[int, List[ChakraNode]] = {}

        # Per-rank tiered-mode bookkeeping: last bg_tail per (stage_index).
        self._tiered_bg: Dict[Tuple[int, int], List[ChakraNode]] = {}

        # Back-compat attributes expected by other code/tests.
        self.mode = self.cfg.mode
        self.checkpoint_every_n = self.cfg.checkpoint_every_n
        self.state_multiplier = self.cfg.state_multiplier
        self.overhead_multiplier = self.cfg.overhead_multiplier
        self.drain_when_backpressure = bool(wc.get("drain_when_backpressure", False))
        self.persist_in_background_no_graph = self.cfg.persist_in_background_no_graph

        # Dispatch table: mode -> emit method.
        self._emitters: Dict[str, Callable[..., CheckpointInjection]] = {
            "sync": self._emit_sync,
            "async": self._emit_async,
            "remote_sync": self._emit_remote_sync,
            "remote_async": self._emit_remote_async,
            "checkfreq_like": self._emit_checkfreq_like,
            "pipelined": self._emit_pipelined,
            "tiered": self._emit_tiered,
        }

    # ------------------------------------------------------------------
    # Model passthroughs
    # ------------------------------------------------------------------
    def fwd(self, name: str, npu_id: int, layer: int, num_batches: int,
            pg_name: Optional[str] = None) -> List[ChakraNode]:
        return self.model.fwd(name, npu_id, layer, num_batches, pg_name)

    def bckwd(self, name: str, npu_id: int, layer: int, num_batches: int,
              pg_name: Optional[str] = None) -> List[ChakraNode]:
        return self.model.bckwd(name, npu_id, layer, num_batches, pg_name)

    # ------------------------------------------------------------------
    # Public properties expected by the orchestrator + tests
    # ------------------------------------------------------------------
    @property
    def checkpoint_boundary_all_nodes(self) -> bool:
        """Legacy hook: True for modes where the next iteration must fence
        on every returned node (not just nodes[0]). Kept so older orchestrator
        code still works; the new orchestrator instead reads
        ``CheckpointInjection.boundary_nodes`` directly."""
        return self.mode == "remote_sync"

    @property
    def uses_storage_rank(self) -> bool:
        """True if any emitted stage sends to ``self.storage_rank_id``. The
        orchestrator uses this to optionally register the storage rank as a
        comm group."""
        if self.mode in ("remote_sync", "remote_async") and self.cfg.remote_target == "storage":
            return True
        if self.mode == "tiered":
            return any(str(s.get("destination", "")).lower() == "storage" for s in self.cfg.stages)
        return False

    # ------------------------------------------------------------------
    # Cost / cadence helpers
    # ------------------------------------------------------------------
    def _is_checkpoint_iteration(self, iteration: int) -> bool:
        if self.cfg.align_with == "epoch":
            period = self.cfg.checkpoint_every_n * self.cfg.iterations_per_epoch
        else:
            period = self.cfg.checkpoint_every_n
        return iteration % period == 0

    def _bytes_to_write_for(self, state_multiplier: float) -> int:
        bytes_per_rank = self._params_per_rank * self._bytes_per_val * state_multiplier
        if self.cfg.checkpoint_bytes_respect_scale:
            bytes_per_rank *= self.scale
        return max(int(bytes_per_rank), 1)

    def _cost_model_for(self, npu_id: int) -> Tuple[StageCostModel, float]:
        """Return (per-rank cost model, per-rank state_multiplier)."""
        override = self.cfg.rank_overrides.get(npu_id)
        if override is None:
            override = self.cfg.rank_overrides.get(str(npu_id))
        if not override:
            return self.cfg.base_costs, self.cfg.state_multiplier
        state_mult = float(override.get("state_multiplier", self.cfg.state_multiplier))
        cost_fields = {f.name for f in dataclasses.fields(StageCostModel)}
        cost_kwargs = {k: v for k, v in override.items() if k in cost_fields}
        if cost_kwargs:
            costs = replace(self.cfg.base_costs, **cost_kwargs)
        else:
            costs = self.cfg.base_costs
        return costs, state_mult

    def _rng(self, npu_id: int) -> Optional[random.Random]:
        if self.cfg.base_costs.cost_jitter_pct <= 0:
            return None
        if npu_id not in self._rng_by_npu:
            seed = self.cfg.seed if self.cfg.seed is not None else 0
            self._rng_by_npu[npu_id] = random.Random((int(seed) << 16) ^ npu_id)
        return self._rng_by_npu[npu_id]

    def _bg_chain_parent(self, npu_id: int, fallback: Optional[ChakraNode]) -> Optional[ChakraNode]:
        """Return the bg-tail the next background phase should depend on,
        honouring ``max_inflight``. ``fallback`` is the prev_write_bg the
        orchestrator passed in (used when no internal history is tracked)."""
        mi = self.cfg.max_inflight
        if mi == 0:
            return None  # unlimited concurrency
        history = self._bg_history.get(npu_id, [])
        if not history:
            return fallback
        # If we've already filled the window, the oldest entry must complete
        # before the new one can start.
        if len(history) >= mi:
            return history[-mi]
        return None

    def _push_bg(self, npu_id: int, bg_node: ChakraNode) -> None:
        if self.cfg.max_inflight == 0:
            return
        hist = self._bg_history.setdefault(npu_id, [])
        hist.append(bg_node)
        # Keep at most max_inflight entries; the oldest one is the dep target.
        if len(hist) > self.cfg.max_inflight:
            del hist[: len(hist) - self.cfg.max_inflight]

    def _comm_tag(self, iteration: int) -> int:
        return self.cfg.comm_tag_base + iteration

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------
    def get_checkpoint_nodes(
        self,
        npu_id: int,
        parents: List[Optional[ChakraNode]],
        iteration: int = 0,
        prev_write_bg: Optional[ChakraNode] = None,
    ) -> CheckpointInjection:
        """Return the nodes to inject for ``iteration`` on ``npu_id``."""
        if self.mode != "tiered" and not self._is_checkpoint_iteration(iteration):
            return CheckpointInjection(nodes=[], bg_tail=prev_write_bg, snapshot_fence=None)
        valid_parents = [p for p in parents if p]
        if not valid_parents:
            return CheckpointInjection(nodes=[], bg_tail=prev_write_bg, snapshot_fence=None)

        emitter = self._emitters[self.mode]
        return emitter(npu_id=npu_id, valid_parents=valid_parents,
                       iteration=iteration, prev_write_bg=prev_write_bg)

    # ------------------------------------------------------------------
    # Per-mode emitters
    # ------------------------------------------------------------------
    def _emit_sync(self, npu_id: int, valid_parents: List[ChakraNode],
                   iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        snap_cost, snap_dur = costs.snapshot_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
        pers_cost, pers_dur = costs.persist_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
        total_cost = snap_cost + pers_cost
        if costs.sync_total_micros is not None:
            # Legacy override: replace summed duration, still scaled by overhead.
            total_dur = int(costs.sync_total_micros * self.cfg.overhead_multiplier)
        else:
            total_dur = snap_dur + pers_dur
        node = compute(
            flops=total_cost,
            tensor_size=total_cost,
            parents=valid_parents,
            name=_name(_N_SYNC, iteration, npu_id),
            duration_micros=max(1, total_dur),
        )
        return CheckpointInjection(nodes=[node], bg_tail=None, snapshot_fence=None)

    def _emit_checkfreq_like(self, npu_id: int, valid_parents: List[ChakraNode],
                             iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        return self._emit_cf_family(npu_id, valid_parents, iteration, prev_write_bg)

    def _emit_pipelined(self, npu_id: int, valid_parents: List[ChakraNode],
                        iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        # Same structure as checkfreq_like. The relevant knob is max_inflight,
        # which the user sets in config (default 1 = same as checkfreq_like;
        # set max_inflight: 4 to approximate PCcheck-style pipelining).
        return self._emit_cf_family(npu_id, valid_parents, iteration, prev_write_bg)

    def _emit_cf_family(self, npu_id: int, valid_parents: List[ChakraNode],
                        iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        nodes: List[ChakraNode] = []

        # Drain when a previous persist is still inflight (subject to max_inflight).
        bg_dep = self._bg_chain_parent(npu_id, prev_write_bg)
        if bg_dep is not None:
            drain = compute(
                flops=1,
                tensor_size=1,
                parents=valid_parents + [bg_dep],
                name=_name(_N_CF_DRAIN, iteration, npu_id),
                duration_micros=1,
            )
            nodes.append(drain)
            boundary_parents = [drain]
        else:
            boundary_parents = list(valid_parents)

        # Kickoff: cap defaults to checkfreq_like historical value (1e5).
        # Legacy convention: when kickoff_micros is 0, kickoff_cost is also 0
        # (used by CheckWork experiment drivers to make the
        # kickoff a graph-only fence with no simulated time).
        cap = costs.resolved_kickoff_cap(_CF_KICKOFF_CAP_DEFAULT)
        kickoff_dur = int(costs.kickoff_micros)
        kickoff_cost = 0 if kickoff_dur == 0 else min(int(raw_bytes), cap)
        kickoff = compute(
            flops=kickoff_cost,
            tensor_size=kickoff_cost,
            parents=boundary_parents,
            name=_name(_N_CF_KICKOFF, iteration, npu_id),
            duration_micros=kickoff_dur,
        )
        nodes.append(kickoff)

        # Snapshot (on critical path -- fences next weight update via orchestrator)
        snap_cost, snap_dur = costs.snapshot_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
        snapshot = compute(
            flops=snap_cost,
            tensor_size=snap_cost,
            parents=[kickoff],
            name=_name(_N_CF_SNAPSHOT, iteration, npu_id),
            duration_micros=snap_dur,
        )
        nodes.append(snapshot)

        if self.cfg.persist_in_background_no_graph:
            return CheckpointInjection(nodes=nodes, bg_tail=None, snapshot_fence=snapshot)

        # Persist (in background; chained for max_inflight via _push_bg).
        pers_cost, pers_dur = costs.persist_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
        persist = compute(
            flops=pers_cost,
            tensor_size=pers_cost,
            parents=[snapshot],
            name=_name(_N_CF_PERSIST, iteration, npu_id),
            duration_micros=pers_dur,
        )
        nodes.append(persist)
        self._push_bg(npu_id, persist)
        return CheckpointInjection(nodes=nodes, bg_tail=persist, snapshot_fence=snapshot)

    def _emit_async(self, npu_id: int, valid_parents: List[ChakraNode],
                    iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        nodes: List[ChakraNode] = []

        # Explicit DRAIN only when the user opted in via drain_when_backpressure.
        # Independent of that, max_inflight governs the bg chain (default 1).
        bg_dep = self._bg_chain_parent(npu_id, prev_write_bg)
        if self.drain_when_backpressure and bg_dep is not None and iteration > 0:
            drain = compute(
                flops=1,
                tensor_size=1,
                parents=valid_parents + [bg_dep],
                name=_name(_N_ASYNC_DRAIN, iteration, npu_id),
                duration_micros=1,
            )
            nodes.append(drain)
            boundary_parents = [drain]
        else:
            boundary_parents = list(valid_parents)

        write_bg_cost, write_bg_dur = costs.write_bg_cost_dur(
            raw_bytes, self.cfg.overhead_multiplier, rng
        )
        # Historical async kickoff cap = 1e6 (same magnitude as the conceptual
        # write_bg cap). Resolved via stage_costs.kickoff_flops_cap if set.
        cap = costs.resolved_kickoff_cap(_ASYNC_KICKOFF_CAP_DEFAULT)
        kickoff_cost = min(write_bg_cost, cap)
        kickoff = compute(
            flops=kickoff_cost,
            tensor_size=kickoff_cost,
            parents=boundary_parents,
            name=_name(_N_ASYNC_KICKOFF, iteration, npu_id),
            duration_micros=int(costs.kickoff_micros),
        )
        nodes.append(kickoff)

        write_bg_parents: List[ChakraNode] = [kickoff]
        if bg_dep is not None:
            write_bg_parents.append(bg_dep)
        write_bg = compute(
            flops=write_bg_cost,
            tensor_size=write_bg_cost,
            parents=write_bg_parents,
            name=_name(_N_ASYNC_WRITE_BG, iteration, npu_id),
            duration_micros=write_bg_dur,
        )
        nodes.append(write_bg)
        self._push_bg(npu_id, write_bg)
        return CheckpointInjection(nodes=nodes, bg_tail=write_bg, snapshot_fence=None)

    def _emit_remote_sync(self, npu_id: int, valid_parents: List[ChakraNode],
                          iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        cost_for_kickoff = int(raw_bytes * self.cfg.overhead_multiplier)
        cap = costs.resolved_kickoff_cap(_REMOTE_KICKOFF_CAP_DEFAULT)
        kickoff_cost = min(max(cost_for_kickoff, 1), cap)
        kickoff = compute(
            flops=kickoff_cost,
            tensor_size=kickoff_cost,
            parents=valid_parents,
            name=_name(_N_REMOTE_KICKOFF, iteration, npu_id),
            duration_micros=int(costs.kickoff_micros),
        )
        tag = self._comm_tag(iteration)
        target = self.cfg.remote_target

        if target == "ring":
            next_npu = (npu_id + 1) % self.num_npus
            prev_npu = (npu_id - 1 + self.num_npus) % self.num_npus
            snd = send(npu_id, next_npu, raw_bytes,
                       name=_name(_N_REMOTE_SYNC_RING_SEND, iteration, npu_id),
                       parents=[kickoff], comm_tag=tag)
            rcv = receive(prev_npu, npu_id, raw_bytes,
                          name=_name(_N_REMOTE_SYNC_RING_RECV, iteration, npu_id),
                          parents=[kickoff], comm_tag=tag)
            sync_join = compute(
                flops=1, tensor_size=1,
                parents=[snd, rcv],
                name=_name(_N_REMOTE_SYNC_JOIN, iteration, npu_id),
                duration_micros=1,
            )
            # Order preserved from pre-refactor: [sync_join, kickoff, snd, rcv]
            # so nodes[0] is the explicit join (kept for any legacy code).
            nodes = [sync_join, kickoff, snd, rcv]
            return CheckpointInjection(nodes=nodes, bg_tail=None, snapshot_fence=None,
                                       boundary_nodes=nodes)

        if target == "rank0":
            if npu_id == 0:
                nodes: List[ChakraNode] = [kickoff]
                for src in range(1, self.num_npus):
                    rcv = receive(src, 0, raw_bytes,
                                  name=_name(_N_REMOTE_SYNC_RANK0_RECV, iteration, 0, f"from{src}"),
                                  parents=[kickoff], comm_tag=tag)
                    nodes.append(rcv)
                return CheckpointInjection(nodes=nodes, bg_tail=None, snapshot_fence=None,
                                           boundary_nodes=nodes)
            snd = send(npu_id, 0, raw_bytes,
                       name=_name(_N_REMOTE_SYNC_RANK0_SEND, iteration, npu_id),
                       parents=[kickoff], comm_tag=tag)
            nodes = [kickoff, snd]
            return CheckpointInjection(nodes=nodes, bg_tail=None, snapshot_fence=None,
                                       boundary_nodes=nodes)

        # target == "storage"
        snd = send(npu_id, self.storage_rank_id, raw_bytes,
                   name=_name(_N_REMOTE_SYNC_STORAGE_SEND, iteration, npu_id),
                   parents=[kickoff], comm_tag=tag)
        nodes = [kickoff, snd]
        return CheckpointInjection(nodes=nodes, bg_tail=None, snapshot_fence=None,
                                   boundary_nodes=nodes)

    def _emit_remote_async(self, npu_id: int, valid_parents: List[ChakraNode],
                           iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        cost_for_kickoff = int(raw_bytes * self.cfg.overhead_multiplier)
        cap = costs.resolved_kickoff_cap(_REMOTE_KICKOFF_CAP_DEFAULT)
        kickoff_cost = min(max(cost_for_kickoff, 1), cap)
        tag = self._comm_tag(iteration)
        nodes: List[ChakraNode] = []

        # Drain only when user opted into back-pressure (legacy semantic).
        bg_dep = self._bg_chain_parent(npu_id, prev_write_bg)
        if self.drain_when_backpressure and bg_dep is not None and iteration > 0:
            drain = compute(
                flops=1,
                tensor_size=1,
                parents=valid_parents + [bg_dep],
                name=_name(_N_REMOTE_DRAIN, iteration, npu_id),
                duration_micros=1,
            )
            nodes.append(drain)
            boundary_parents = [drain]
        else:
            boundary_parents = list(valid_parents)

        kickoff = compute(
            flops=kickoff_cost,
            tensor_size=kickoff_cost,
            parents=boundary_parents,
            name=_name(_N_REMOTE_KICKOFF, iteration, npu_id),
            duration_micros=int(costs.kickoff_micros),
        )
        nodes.append(kickoff)

        upload_parents: List[ChakraNode] = [kickoff]
        if bg_dep is not None:
            upload_parents.append(bg_dep)
        target = self.cfg.remote_target

        if target == "storage":
            up = send(npu_id, self.storage_rank_id, raw_bytes,
                      name=_name(_N_REMOTE_ASYNC_RING_SEND, iteration, npu_id),
                      parents=upload_parents, comm_tag=tag)
            nodes.append(up)
            self._push_bg(npu_id, up)
            return CheckpointInjection(nodes=nodes, bg_tail=up, snapshot_fence=None)

        if target == "rank0":
            if npu_id == 0:
                last_rcv: Optional[ChakraNode] = None
                for src in range(1, self.num_npus):
                    rcv = receive(src, 0, raw_bytes,
                                  name=_name(_N_REMOTE_ASYNC_RANK0_RECV, iteration, 0, f"from{src}"),
                                  parents=[kickoff], comm_tag=tag)
                    nodes.append(rcv)
                    last_rcv = rcv
                if last_rcv is not None:
                    self._push_bg(npu_id, last_rcv)
                return CheckpointInjection(nodes=nodes, bg_tail=last_rcv, snapshot_fence=None)
            snd = send(npu_id, 0, raw_bytes,
                       name=_name(_N_REMOTE_ASYNC_RANK0_SEND, iteration, npu_id),
                       parents=upload_parents, comm_tag=tag)
            nodes.append(snd)
            self._push_bg(npu_id, snd)
            return CheckpointInjection(nodes=nodes, bg_tail=snd, snapshot_fence=None)

        # target == "ring"
        next_npu = (npu_id + 1) % self.num_npus
        prev_npu = (npu_id - 1 + self.num_npus) % self.num_npus
        up_snd = send(npu_id, next_npu, raw_bytes,
                      name=_name(_N_REMOTE_ASYNC_RING_SEND, iteration, npu_id),
                      parents=upload_parents, comm_tag=tag)
        up_rcv = receive(prev_npu, npu_id, raw_bytes,
                         name=_name(_N_REMOTE_ASYNC_RING_RECV, iteration, npu_id),
                         parents=valid_parents, comm_tag=tag)
        nodes.append(up_snd)
        nodes.append(up_rcv)
        self._push_bg(npu_id, up_snd)
        return CheckpointInjection(nodes=nodes, bg_tail=up_snd, snapshot_fence=None)

    def _emit_tiered(self, npu_id: int, valid_parents: List[ChakraNode],
                     iteration: int, prev_write_bg: Optional[ChakraNode]) -> CheckpointInjection:
        """
        Run each stage at its own cadence. Each stage produces one COMP_NODE
        (cost from its 'kind') plus optional comm nodes (from its 'destination').

        Stage dict keys:
          kind          "snapshot" | "persist" | "write_bg"
          every_n       int >= 1 (default 1)
          destination   "local" (default) | "ring" | "rank0" | "storage"
          on_critical_path  bool (default False); if True, next iter's first
                            compute fences on this stage.
        """
        costs, state_mult = self._cost_model_for(npu_id)
        rng = self._rng(npu_id)
        raw_bytes = self._bytes_to_write_for(state_mult)
        tag = self._comm_tag(iteration)
        nodes: List[ChakraNode] = []
        snapshot_fence: Optional[ChakraNode] = None
        boundary_nodes: List[ChakraNode] = []
        bg_tail: Optional[ChakraNode] = None

        for stage_idx, stage in enumerate(self.cfg.stages):
            every_n = int(stage.get("every_n", 1))
            if iteration % every_n != 0:
                continue
            kind = str(stage.get("kind", "")).lower()
            destination = str(stage.get("destination", "local")).lower()
            on_critical_path = bool(stage.get("on_critical_path", False))

            # Build per-stage chain dependency on previous bg of *this* stage.
            stage_key = (npu_id, stage_idx)
            stage_hist = self._tiered_bg.setdefault(stage_key, [])
            stage_bg_dep: Optional[ChakraNode] = None
            if self.cfg.max_inflight > 0 and stage_hist:
                if len(stage_hist) >= self.cfg.max_inflight:
                    stage_bg_dep = stage_hist[-self.cfg.max_inflight]

            comp_parents = list(valid_parents)
            if stage_bg_dep is not None:
                comp_parents.append(stage_bg_dep)

            if kind == "snapshot":
                cost, dur = costs.snapshot_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
            elif kind == "persist":
                cost, dur = costs.persist_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
            else:  # write_bg
                cost, dur = costs.write_bg_cost_dur(raw_bytes, self.cfg.overhead_multiplier, rng)
            comp = compute(
                flops=cost,
                tensor_size=cost,
                parents=comp_parents,
                name=_name(_N_TIERED_STAGE_COMP, iteration, npu_id, f"s{stage_idx}", kind),
                duration_micros=dur,
            )
            nodes.append(comp)
            if kind == "snapshot" and on_critical_path:
                # Same semantics as checkfreq_like snapshot fence.
                snapshot_fence = comp

            stage_tail: ChakraNode = comp
            if destination == "ring":
                nxt = (npu_id + 1) % self.num_npus
                prv = (npu_id - 1 + self.num_npus) % self.num_npus
                snd = send(npu_id, nxt, raw_bytes,
                           name=_name(_N_TIERED_STAGE_SEND, iteration, npu_id, f"s{stage_idx}", "ring"),
                           parents=[comp], comm_tag=tag)
                rcv = receive(prv, npu_id, raw_bytes,
                              name=_name(_N_TIERED_STAGE_RECV, iteration, npu_id, f"s{stage_idx}", "ring"),
                              parents=[comp], comm_tag=tag)
                nodes.extend([snd, rcv])
                stage_tail = snd
            elif destination == "rank0":
                if npu_id == 0:
                    last = None
                    for src in range(1, self.num_npus):
                        rcv = receive(src, 0, raw_bytes,
                                      name=_name(_N_TIERED_STAGE_RECV, iteration, 0, f"s{stage_idx}", f"from{src}"),
                                      parents=[comp], comm_tag=tag)
                        nodes.append(rcv)
                        last = rcv
                    if last is not None:
                        stage_tail = last
                else:
                    snd = send(npu_id, 0, raw_bytes,
                               name=_name(_N_TIERED_STAGE_SEND, iteration, npu_id, f"s{stage_idx}", "rank0"),
                               parents=[comp], comm_tag=tag)
                    nodes.append(snd)
                    stage_tail = snd
            elif destination == "storage":
                snd = send(npu_id, self.storage_rank_id, raw_bytes,
                           name=_name(_N_TIERED_STAGE_SEND, iteration, npu_id, f"s{stage_idx}", "storage"),
                           parents=[comp], comm_tag=tag)
                nodes.append(snd)
                stage_tail = snd
            # destination == "local": nothing more to add.

            stage_hist.append(stage_tail)
            if len(stage_hist) > max(1, self.cfg.max_inflight or 1):
                del stage_hist[: len(stage_hist) - max(1, self.cfg.max_inflight or 1)]

            if on_critical_path:
                # Boundary for next iter must wait on every node we just emitted
                # for this stage so the simulator schedules them on the path.
                boundary_nodes.append(stage_tail)
            else:
                bg_tail = stage_tail

        if not nodes:
            return CheckpointInjection(nodes=[], bg_tail=prev_write_bg, snapshot_fence=None)

        return CheckpointInjection(
            nodes=nodes,
            bg_tail=bg_tail,
            snapshot_fence=snapshot_fence,
            boundary_nodes=boundary_nodes or None,
        )

    # ------------------------------------------------------------------
    # Pass-through metadata for downstream code
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return self.model.get_name()

    def get_num_params(self) -> int:
        return self.model.get_num_params()

    def get_num_layers(self) -> int:
        return self.model.get_num_layers()

    def get_hidden_size(self) -> int:
        return self.model.get_hidden_size()

    def get_sequence_len(self) -> int:
        return self.model.get_sequence_len()

    def get_batch_size(self) -> int:
        return self.model.get_batch_size()

    def get_bytes_per_val(self) -> int:
        return self.model.get_bytes_per_val()
