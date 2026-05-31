# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from Model.Model import Model
from Orchestrator.Orchestrator import Orchestrator
from utils import add_dependencies, allreduce, receive, send, reset_id_counter
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)

# Modes that emit a snapshot node which must fence the *next* iteration's
# weight update. The wrapper returns snapshot_fence != None for these modes;
# the orchestrator wires it into the next iteration's DP all-reduce or last
# backward node.
_SNAPSHOT_FENCE_MODES = {"checkfreq_like", "pipelined", "tiered"}


class MegatronLM(Orchestrator):
    def __init__(self, 
        model: Model,
        config):
        self.model = model
        self.dp_size = config["parallelism"]["dp_size"]
        self.pp_size = config["parallelism"]["pp_size"]
        self.tp_size = config["parallelism"]["tp_size"]
        self.num_npus = self.dp_size * self.pp_size * self.tp_size
        self.num_microbatches = config["model"]["num_microbatches"]
        self.scale = config["model"]["scale"]
        self.num_iterations = int(config.get("num_iterations", 1))

    def generate_comm_groups(self):
        """Generate comm groups with numeric IDs (ASTRA-sim requires stoi-parseable keys)."""
        comm_groups_raw = defaultdict(list)
        self._pg_name_to_id = {}  # semantic name -> numeric id for ET pg_name attribute
        next_id = 1  # ASTRA-sim reserves id 0 for default comm group; requires id > 0

        # generate comm groups for data parallel groups
        for dp_group in range(self.dp_size):
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    npu_id = dp_group * (self.pp_size * self.tp_size) + (pp_stage * self.tp_size) + tp_shard
                    pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
                    tp_name = "" if self.tp_size <= 1 else f"_tp_{tp_shard}"
                    key = f"{pp_name}{tp_name}" if (pp_name or tp_name) else "_default"
                    comm_groups_raw[key].append(npu_id)
                    if key not in self._pg_name_to_id:
                        self._pg_name_to_id[key] = next_id
                        next_id += 1

        # generate comm groups for each tensor parallel group
        if self.tp_size > 1:
            for tp_group in range(self.num_npus // self.tp_size):
                tp_comm_group = []
                base = tp_group * self.tp_size
                for npu in range(self.tp_size):
                    tp_comm_group.append(base + npu)
                key = f"tp_{tp_group}"
                comm_groups_raw[key] = tp_comm_group
                if key not in self._pg_name_to_id:
                    self._pg_name_to_id[key] = next_id
                    next_id += 1

        # If the wrapper uses a virtual storage rank, register it as a
        # singleton comm group so downstream simulators that *require* every
        # referenced npu to exist can find it. ASTRA-sim historically ignored
        # unknown ranks for P2P, so this is a no-op for ASTRA-sim runs.
        if getattr(self.model, "uses_storage_rank", False):
            storage_rank = int(getattr(self.model, "storage_rank_id", self.num_npus))
            key = "checkpoint_storage"
            comm_groups_raw[key] = [storage_rank]
            if key not in self._pg_name_to_id:
                self._pg_name_to_id[key] = next_id
                next_id += 1

        # Convert to numeric string keys for ASTRA-sim (stoi-parseable)
        return {str(self._pg_name_to_id[k]): v for k, v in comm_groups_raw.items()}

    def _pg_name(self, semantic_name: str) -> str:
        """Map semantic pg name (e.g. pp_0) to numeric id string for ASTRA-sim."""
        key = semantic_name if semantic_name else "_default"
        return str(self._pg_name_to_id.get(key, 0))

    def exec(self) -> dict:
        # Each exec() starts a fresh node-id namespace so that two simulations
        # run in the same Python interpreter (e.g. inside a test runner) do
        # not produce nodes whose IDs collide with the previous run.
        reset_id_counter(0)
        num_params = self.model.num_params
        B = self.model.get_batch_size()
        S = self.model.get_sequence_len()
        d = self.model.get_hidden_size()
        b = self.model.get_bytes_per_val()

        layers_per_pipeline_stage = self.model.get_num_layers() // self.pp_size
        pp_comm_size = int((B*S*d*b * self.scale) / self.num_microbatches)
        dp_comm_size = int(self.scale * num_params * b / self.tp_size / self.pp_size)

        # print(f"Num params: {num_params:,.2f}")
        # print(f"Pipeline comm size: {pp_comm_size / 1024 / 1024:,.2f} MB")
        # print(f"DP comm size: {dp_comm_size / 1024 / 1024 / 1024:,.2f} GB")

        nodes = defaultdict(list)

        for dp_group in range(self.dp_size):
            for pp_stage in range(self.pp_size):
                for tp_shard in range(self.tp_size):
                    npu_id = dp_group * (self.pp_size * self.tp_size) + (pp_stage * self.tp_size) + tp_shard
                    tp_group = npu_id // self.tp_size
                    nodes[npu_id].append(GlobalMetadata(version="0.0.4"))

                    prev_comp = None
                    prev_write_bg = None  # background checkpoint node (async write / persist)
                    last_ckpt_nodes = None
                    pending_snapshot_fence = None  # checkfreq_like: snapshot(i) must complete before weight_update(i+1)

                    for iteration in range(self.num_iterations):
                        # Boundary for this iteration: whatever the wrapper
                        # told us last iter. For most modes that's just
                        # [prev_comp] (single-node fence); for remote_sync /
                        # tiered-on-critical-path it's the full list returned
                        # via CheckpointInjection.boundary_nodes.
                        boundary_deps = (
                            list(last_ckpt_nodes)
                            if (iteration > 0 and last_ckpt_nodes)
                            else ([prev_comp] if prev_comp else [])
                        )

                        # -------------
                        # Forward pass
                        # -------------
                        prev_rcv = None
                        for b in range(self.num_microbatches):
                            rcv_node = None
                            if pp_stage != 0 and self.pp_size > 1:
                                rcv_parents = [prev_rcv] if prev_rcv else []
                                if iteration > 0 and boundary_deps:
                                    rcv_parents.extend(boundary_deps)
                                tag = 1 + iteration * 1000 + b * 20
                                rcv_node = receive(npu_id - self.tp_size, npu_id, pp_comm_size, parents=rcv_parents if rcv_parents else None, name=f"COMM_RECV_NODE_FWD_iter{iteration}_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}", comm_tag=tag)
                                nodes[npu_id].append(rcv_node)
                                prev_rcv = rcv_node

                            for layer in range(layers_per_pipeline_stage):
                                current_layer = pp_stage * layers_per_pipeline_stage + layer
                                cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_iter{iteration}_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=self._pg_name(f"tp_{tp_group}"))
                                if layer == 0:
                                    fwd_deps = [x for x in [rcv_node] if x is not None] + boundary_deps
                                    if fwd_deps:
                                        add_dependencies(cmp_nodes[0], fwd_deps)
                                else:
                                    add_dependencies(cmp_nodes[0], [prev_comp])
                                for node in cmp_nodes:
                                    nodes[npu_id].append(node)
                                prev_comp = cmp_nodes[-1]

                            if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                                tag = 1 + iteration * 1000 + b * 20
                                snd_node = send(npu_id, npu_id + self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_iter{iteration}_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}", comm_tag=tag)
                                nodes[npu_id].append(snd_node)

                        # -------------
                        # Backward pass
                        # -------------
                        for b in range(self.num_microbatches):
                            bck_rcv_node = None
                            if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                                tag = 1 + iteration * 1000 + b * 20 + 10
                                bck_rcv_node = receive(npu_id + self.tp_size, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_iter{iteration}_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}", comm_tag=tag)
                                nodes[npu_id].append(bck_rcv_node)
                                prev_rcv = bck_rcv_node

                            for layer in range(layers_per_pipeline_stage):
                                current_layer = pp_stage * layers_per_pipeline_stage + layer
                                bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_iter{iteration}_b{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=self._pg_name(f"tp_{tp_group}"))
                                if layer == 0:
                                    bck_deps = [x for x in [bck_rcv_node, prev_comp] if x is not None]
                                    add_dependencies(bck_cmp_nodes[0], bck_deps)
                                else:
                                    add_dependencies(bck_cmp_nodes[0], [prev_comp])
                                for node in bck_cmp_nodes:
                                    nodes[npu_id].append(node)
                                prev_comp = bck_cmp_nodes[-1]

                            if pp_stage != 0 and self.pp_size > 1:
                                tag = 1 + iteration * 1000 + b * 20 + 10
                                bck_snd_node = send(npu_id, npu_id - self.tp_size, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_iter{iteration}_b{b}_dp{dp_group}pp{pp_stage}tp{tp_shard}", comm_tag=tag)
                                nodes[npu_id].append(bck_snd_node)

                        if self.dp_size > 1:
                            pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
                            tp_name = "" if self.tp_size <= 1 else f"_tp_{tp_shard}"
                            dp_comm_node = allreduce(dp_comm_size, parents=[prev_comp], pg_name=self._pg_name(f"{pp_name}{tp_name}"), name=f"COMM_COLL_NODE_DP_All-Reduce_iter{iteration}_dp{dp_group}pp{pp_stage}tp{tp_shard}")
                            if pending_snapshot_fence is not None:
                                add_dependencies(dp_comm_node, [pending_snapshot_fence])
                                pending_snapshot_fence = None
                            nodes[npu_id].append(dp_comm_node)
                            prev_comp = dp_comm_node
                        else:
                            # Weight update = last backward node; enforce snapshot fence before it
                            if pending_snapshot_fence is not None:
                                add_dependencies(prev_comp, [pending_snapshot_fence])
                                pending_snapshot_fence = None

                        if hasattr(self.model, "get_checkpoint_nodes"):
                            inj = self.model.get_checkpoint_nodes(
                                npu_id, parents=[prev_comp],
                                iteration=iteration, prev_write_bg=prev_write_bg,
                            )
                            # Support both the new CheckpointInjection
                            # dataclass and the legacy 3-tuple unpack.
                            if hasattr(inj, "nodes"):
                                ckpt_nodes = inj.nodes
                                bg_node = inj.bg_tail
                                snapshot_fence = inj.snapshot_fence
                                boundary_nodes_override = inj.boundary_nodes
                            else:
                                ckpt_nodes, bg_node, snapshot_fence = inj
                                boundary_nodes_override = None
                            for node in ckpt_nodes:
                                nodes[npu_id].append(node)
                            if ckpt_nodes:
                                # boundary_nodes_override lets the wrapper say
                                # "next iter must wait on all of these"
                                # (remote_sync, tiered with on_critical_path).
                                # Default: just the first emitted node, which
                                # is always the canonical boundary.
                                last_ckpt_nodes = (
                                    list(boundary_nodes_override)
                                    if boundary_nodes_override
                                    else [ckpt_nodes[0]]
                                )
                                prev_comp = ckpt_nodes[0]
                                prev_write_bg = bg_node
                                mode = getattr(self.model, "mode", None)
                                if mode in _SNAPSHOT_FENCE_MODES and snapshot_fence is not None:
                                    pending_snapshot_fence = snapshot_fence
                            else:
                                last_ckpt_nodes = None
                                # prev_write_bg unchanged (wrapper returns it as bg_node on non-checkpoint iters)
        return nodes
