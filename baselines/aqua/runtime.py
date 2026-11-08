"""Adapters around the official Aqua tensor/manager, not SwiftCache's allocator."""

import os
import time
import threading

import torch

from .vendor.official_aqua.controller.mmc import MemoryManagerClient
from .vendor.official_aqua.policies.dynamic_policy import dynamic_policy
from .vendor.official_aqua.policies.aqua_policy import tensor_home
from .vendor.official_aqua.datastructures.responsive_tensor import tensor_device
from .vendor.official_aqua.core.responsive_manager import responsive_manager


class Trace:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def log(self, event, **data):
        with self.lock:
            self.events.append(dict(event=event, time_ns=time.monotonic_ns(), **data))
            self.events = self.events[-10000:]

    def snapshot(self):
        with self.lock:
            return self.events.copy()


class Client(MemoryManagerClient):
    def __init__(self, host, port, source_gpu):
        super().__init__(host, port)
        self.source_gpu = source_gpu
        self.session.trust_env = False
        self.rpc_lock = threading.Lock()

    def _make_request(self, method, endpoint, payload):
        with self.rpc_lock:
            response = self.session.request(
                method,
                self.base_url + "/" + endpoint,
                json=payload,
                headers={"source-gpu": str(self.source_gpu)},
                timeout=15,
            )
        response.raise_for_status()
        return response.json()

    def status(self):
        with self.rpc_lock:
            response = self.session.get(self.base_url + "/status", timeout=15)
        response.raise_for_status()
        return response.json()


class Policy(dynamic_policy):
    """Preserve upstream purge/ack, fix allocation tracking and re-expansion.

    Upstream's expansion retries all tensors, including those already on a GPU.
    Only unallocated CPU tensors are eligible here. Release is explicit so
    completed/preempted requests cannot leave permanent coordinator leases.
    """

    def __init__(self, client, physical_to_logical, flush, trace):
        super().__init__("127.0.0.1", 0)
        self.mmc = client
        self.devices = physical_to_logical
        self.flush = flush
        self.trace = trace

    def _real_to_virtual_gid(self, address):
        physical = int(address.split(":")[1])
        if physical not in self.devices:
            raise ValueError("Aqua provider is not visible to this consumer")
        return f"cuda:{self.devices[physical]}"

    def add_rtensor(self, rt):
        return self.add_rtensors([rt])[0]

    def add_rtensors(self, rts):
        if not rts:
            return []
        allocation = self.mmc.malloc_nv_memory(
            sum(rt.get_size_in_bytes() for rt in rts)
        )
        self.rtensors.extend(rts)
        if allocation is None:
            return [tensor_home(tensor_device.DRAM, "cpu") for _ in rts]
        assert allocation["size"] == sum(rt.get_size_in_bytes() for rt in rts)
        self.allocation_ids_to_rt_tensors[allocation["allocation_id"]] = list(rts)
        address = self._real_to_virtual_gid(allocation["address"])
        return [tensor_home(tensor_device.GPU, address) for _ in rts]

    def get_rtensors_to_move(self):
        # The caller enforces inference safe points and polling frequency.
        mapping = super()._check_for_purging()
        if mapping:
            return mapping
        allocated = {
            rt for group in self.allocation_ids_to_rt_tensors.values() for rt in group
        }
        eligible = [
            rt
            for rt in self.rtensors
            if rt not in allocated and rt.active_device == tensor_device.DRAM
        ]
        if not eligible:
            return {}
        allocation = self.mmc.malloc_nv_memory(
            sum(rt.get_size_in_bytes() for rt in eligible)
        )
        if allocation is None:
            return {}
        self.allocation_ids_to_rt_tensors[allocation["allocation_id"]] = eligible
        address = self._real_to_virtual_gid(allocation["address"])
        return {rt: tensor_home(tensor_device.GPU, address) for rt in eligible}

    def done_moving_rtensors(self, mapping):
        if self.previous_purged_allocations:
            ids = self.previous_purged_allocations
            for ident in ids:
                for rt in self.allocation_ids_to_rt_tensors[ident]:
                    if (
                        rt.active_device != tensor_device.DRAM
                        or rt.gpu_tensor is not None
                    ):
                        raise RuntimeError(
                            "Cannot acknowledge reclamation before GPU storage is released"
                        )
            self.flush()
            self.trace.log("gpu_to_dram_complete", allocations=ids.copy())
        super().done_moving_rtensors(mapping)
        for rt, home in mapping.items():
            self.trace.log(
                "tensor_migrated",
                bytes=rt.get_size_in_bytes(),
                location=home.device_type.value,
            )

    def release(self, rt):
        # Do not decrement a shared allocation until every member is closed.
        self.rtensors.remove(rt)
        rt.gpu_tensor = None
        rt.active_device = tensor_device.DRAM
        self.flush()
        for ident, group in list(self.allocation_ids_to_rt_tensors.items()):
            if rt not in group:
                continue
            group.remove(rt)
            if not group:
                self.mmc.free_nv_memory(ident)
                del self.allocation_ids_to_rt_tensors[ident]
        rt.dram_tensor = None


class Store:
    def __init__(
        self,
        client,
        devices,
        consumer_device="cuda:0",
        host_limit_bytes=2 << 30,
        trace=None,
    ):
        self.client = client
        self.trace = trace or Trace()
        self.consumer_device = torch.device(consumer_device)
        self.host_limit_bytes = host_limit_bytes
        self.handles = set()
        self.bytes = 0
        self.devices = devices
        self.policy = Policy(client, devices, self.flush, self.trace)
        # Classmethod-based upstream managers otherwise share a process-global policy.
        self.manager = type(
            "IsolatedAquaManager", (responsive_manager,), {"_policy": self.policy}
        )

    def flush(self):
        for logical in self.devices.values():
            torch.cuda.synchronize(logical)
        torch.cuda.empty_cache()

    def can_fit(self, size):
        return self.bytes + size <= self.host_limit_bytes

    def put(self, packed, kind="request"):
        size = packed.numel() * packed.element_size()
        if not self.can_fit(size):
            raise MemoryError("Aqua host swap budget exhausted")
        self.flush()
        before = len(self.policy.rtensors)
        try:
            # Allocate backing storage with the official Aqua manager, then
            # write live KV directly to its selected GPU (or CPU fallback).
            template = torch.zeros(packed.shape, dtype=packed.dtype, device="cpu")
            rt = self.manager.to_responsive_tensor(template)
        except torch.cuda.OutOfMemoryError:
            # The lease was metadata only; roll it back before CPU fallback.
            created = self.policy.rtensors[before:]
            if len(created) != 1:
                raise
            rt = created[0]
            if rt.gpu_tensor is not None:
                raise
            for ident, group in list(self.policy.allocation_ids_to_rt_tensors.items()):
                if rt in group:
                    self.client.free_nv_memory(ident)
                    del self.policy.allocation_ids_to_rt_tensors[ident]
            self.flush()
            self.trace.log("allocation_oom_dram_fallback", bytes=size)
        self.handles.add(rt)
        self.bytes += size
        rt.to_torch_tensor().copy_(packed, non_blocking=False)
        self.flush()
        self.trace.log(
            "offload",
            kind=kind,
            bytes=size,
            location=rt.active_device.value,
            transfer="gpu_peer"
            if rt.active_device == tensor_device.GPU
            else "gpu_to_dram",
        )
        return rt

    def restore(self, rt):
        if rt not in self.handles:
            raise RuntimeError("Snapshot is closed")
        start = time.perf_counter_ns()
        result = rt.to_torch_tensor().to(
            self.consumer_device, copy=True, non_blocking=False
        )
        torch.cuda.synchronize(self.consumer_device)
        self.trace.log(
            "restore",
            bytes=rt.get_size_in_bytes(),
            source=rt.active_device.value,
            wall_ms=(time.perf_counter_ns() - start) / 1e6,
        )
        return result

    def close(self, rt):
        if rt not in self.handles:
            return
        size = rt.get_size_in_bytes()
        self.policy.release(rt)
        self.handles.remove(rt)
        self.bytes -= size
        self.trace.log("snapshot_closed", bytes=size)

    def respond(self):
        self.flush()
        self.manager.respond()

    def has_dram(self):
        return any(rt.active_device == tensor_device.DRAM for rt in self.handles)

    def close_all(self):
        for rt in list(self.handles):
            self.close(rt)
        assert not self.policy.allocation_ids_to_rt_tensors and self.bytes == 0


class Producer:
    """Upstream LLM-informer hysteresis: high=10, low=2 by default."""

    def __init__(self, client, producer, consumer, budget, trace, high=10, low=2):
        if not 0 <= low < high:
            raise ValueError("Producer low must be below high threshold")
        self.client = client
        self.id = producer
        self.budget = budget
        self.trace = trace
        self.high = high
        self.low = low
        self.state = "shared"
        self.capacity = budget
        client.offer_memory(producer, f"cuda:{producer}", budget)
        client._make_request(
            "POST", "mapping", dict(consumer=consumer, producer=producer)
        )

    def request_reclaim(self):
        if self.state == "shared":
            self.client.reclaim_request(self.id)
            self.state = "reclaiming"
            self.trace.log("producer_reclaim_requested")

    def poll(self, pending, local_kv_bytes=0):
        if self.state == "shared" and pending >= self.high:
            self.request_reclaim()
        if self.state == "reclaiming":
            status = self.client.reclaim_status(self.id)
            if status["can_reclaim"]:
                self.client.add_memory(self.id, f"cuda:{self.id}", -self.capacity)
                self.capacity = 0
                self.client.remove_reclaim_request(self.id)
                self.state = "private"
                self.trace.log("producer_reclaimed")
        if self.state == "private" and pending <= self.low and local_kv_bytes == 0:
            self.client.add_memory(self.id, f"cuda:{self.id}", self.budget)
            self.capacity = self.budget
            self.state = "shared"
            self.trace.log("producer_offered")
        return self.state

    def close(self):
        status = self.client.reclaim_status(self.id)
        if not status["can_reclaim"]:
            raise RuntimeError("Cannot stop a producer with live Aqua allocations")
        self.client.take_back_memory(self.id)
