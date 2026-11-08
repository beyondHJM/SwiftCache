from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import random
import pytest

from baselines.aqua.coordinator import Ledger, LedgerError
from baselines.aqua.engine import plan_cfs
from baselines.aqua.runtime import Policy, Trace, tensor_device


def ledger():
    m = Ledger()
    m.offer(2, "cuda:2", 100)
    m.map(0, 2)
    return m


def test_quarantine_and_ack_before_capacity_reuse():
    m = ledger()
    a = m.allocate(0, 60)
    m.reclaim(2)
    assert m.allocate(0, 1) is None and not m.reclaim_status(2)["can_reclaim"]
    with pytest.raises(LedgerError):
        m.resize(2, -100)
    with pytest.raises(LedgerError):
        m.finish_reclaim(2)
    with pytest.raises(LedgerError):
        m.free(a["allocation_id"], 9)
    assert m.responsive([a["allocation_id"]], 0) == [a["allocation_id"]]
    m.free(a["allocation_id"], 0)
    assert m.free(a["allocation_id"], 0)["already_released"]
    m.resize(2, -100)
    m.finish_reclaim(2)
    assert m.allocate(0, 1) is None
    m.resize(2, 100)
    assert m.allocate(0, 100)
    m.snapshot()


def test_concurrent_grants_never_overcommit():
    m = ledger()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: m.allocate(0, 10), range(100)))
    assert sum(x is not None for x in results) == 10
    assert m.snapshot()["providers"]["2"]["allocated"] == 100
    for a in results:
        if a:
            m.free(a["allocation_id"], 0)
    assert not m.snapshot()["allocations"]


def test_random_reclaim_and_resize_invariants():
    m = ledger()
    rng = random.Random(20261108)
    for _ in range(1000):
        status = m.snapshot()
        if rng.randrange(3) == 0 and status["allocations"]:
            m.free(rng.choice(status["allocations"])["allocation_id"], 0)
        elif rng.randrange(2):
            m.allocate(0, rng.randint(1, 30))
        else:
            m.reclaim(2)
            if m.reclaim_status(2)["can_reclaim"]:
                m.resize(2, -100)
                m.finish_reclaim(2)
                m.resize(2, 100)
        m.snapshot()


def test_cfs_reserves_decode_and_prefill_budget():
    def req(ident, prefill, pos, work, generated):
        return SimpleNamespace(
            ident=ident,
            prefill=prefill,
            pos=pos,
            prefill_work=work,
            generated=[0] * generated,
            arrival_ns=int(ident),
            token_ids=[1] * 500,
        )

    a = req("1", True, 100, 100, 0)
    b = req("2", True, 0, 0, 0)
    c = req("3", False, 500, 500, 4)
    d = req("4", False, 500, 500, 1)
    plan = plan_cfs([a, b, c, d], 128, 1, 64)
    assert plan[0][0] is b and plan[-1][0] is d
    assert sum(n for _, n in plan) <= 128
    assert all(n <= 64 for r, n in plan if r.prefill)


class FakeClient:
    def __init__(self):
        self.m = ledger()

    def malloc_nv_memory(self, n):
        return self.m.allocate(0, n)

    def responsive_reclaim(self, ids):
        return self.m.responsive(ids, 0)

    def free_nv_memory(self, ident):
        return self.m.free(ident, 0)


class Tensor:
    def __init__(self, size):
        self.size = size
        self.active_device = tensor_device.DRAM
        self.gpu_tensor = None
        self.dram_tensor = object()

    def get_size_in_bytes(self):
        return self.size


def test_policy_never_acks_live_gpu_tensor_or_promotes_it_twice():
    client = FakeClient()
    p = Policy(client, {2: 1}, lambda: None, Trace())
    a, b = Tensor(20), Tensor(30)
    homes = p.add_rtensors([a, b])
    for r in [a, b]:
        r.active_device = tensor_device.GPU
        r.gpu_tensor = object()
    assert all(h.device_type == tensor_device.GPU for h in homes)
    assert p.get_rtensors_to_move() == {}
    client.m.reclaim(2)
    mapping = p.get_rtensors_to_move()
    with pytest.raises(RuntimeError, match="before GPU storage"):
        p.done_moving_rtensors(mapping)
    assert not client.m.reclaim_status(2)["can_reclaim"]
    for r in [a, b]:
        r.active_device = tensor_device.DRAM
        r.gpu_tensor = None
    p.done_moving_rtensors(mapping)
    assert client.m.reclaim_status(2)["can_reclaim"]
    client.m.finish_reclaim(2)
    expanded = p.get_rtensors_to_move()
    assert set(expanded) == {a, b}
    for r in expanded:
        r.active_device = tensor_device.GPU
        r.gpu_tensor = object()
    p.done_moving_rtensors(expanded)
    p.release(a)
    p.release(b)
    assert not client.m.snapshot()["allocations"] and not p.rtensors
